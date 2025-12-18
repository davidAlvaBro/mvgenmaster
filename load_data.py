from pathlib import Path
import json

import numpy as np 
from PIL import Image
import cv2
import torch
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, Lambda, ConvertImageDtype


def load_dataset(args, config): 
    """
    Loads 'trajectory' and 'eval' frames from 'transforms_path', translates them to MVGenMaster's
    format and records these changes in a new transforms json.   
    """
    parent_dir=Path(args.working_dir) 
    transforms_path=Path(args.input_path)
    with open(transforms_path, 'r') as file:
        metadata = json.load(file)
    
    frames = metadata["trajectory"] + metadata["eval"]
    
    # NOTE I removed all the rescaling of images and camera parameters because it
    # is already taken care of in the controlnet pipeline. But if I need it back it 
    # is in this comment below. 
    # # Generated images must be factors of 64 (because of the VAE encoder) 
    # ratio_set = json.load(open(f"./{args.model_dir}/ratio_set.json", "r"))
    # ratio_dict = dict()
    # for h, w in ratio_set:
    #     ratio_dict[h / w] = [h, w]
    # ratio_list = list(ratio_dict.keys())
    # h_img, w_img, _ = reference_img.shape
    # print(f'Original image shape is height:{h_img}, width:{w_img}.')
    # ratio = h_img / w_img
    # sub = [abs(ratio - r) for r in ratio_list]
    # [h, w] = ratio_dict[ratio_list[np.argmin(sub)]]
    # print(f'Closest valid ratio is height:{h}, width:{w}.')
    # img = Image.open(img_pth)
    # img = img.resize((w, h), Image.LANCZOS if h < h_img else Image.BICUBIC)
    # depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
    # img = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img).unsqueeze(0)
    # Fix reference camera parameters 
    # ref_names = list(reference_cam.keys())
    # ref_intrinsic = np.array(reference_cam[ref_names[0]]["intrinsic"])
    # ref_extrinsic = np.array(reference_cam[ref_names[0]]["extrinsic"])
    # ref_h, ref_w = reference_cam[ref_names[0]]["h"], reference_cam[ref_names[0]]["w"]
    # ref_intrinsic[0] *= w/ref_w
    # ref_intrinsic[1] *= h/ref_h


    # Reference image 
    ref = args["trajectory_ref"]
    ref_cam = frames[ref] 
    img_pth = parent_dir / ref_cam["file_path"]
    img = cv2.imread(img_pth) # TODO check that this does not fuck anything
    h,w = img.shape[1:] # TODO check this

    # Reference depth 
    depth_pth = parent_dir / ref_cam["depth_path"]
    depth = np.load(depth_pth)
    # depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_img = Compose([ToTensor()])(depth).unsqueeze(0)

    # Load all frames 
    cameras = []
    intrinsics = []
    w2cs = [] 
    zoomed_intrinsics = []
    for i, cam in enumerate(frames): 
        # Update camera parameters (rescale only)
        h_scale = h / cam["h"]
        w_scale = w / cam["w"]
        cam["w"], cam["h"] = w, h
        cam["fl_x"], cam["cx"] = cam["fl_x"]*w_scale, cam["cx"]*w_scale
        cam["fl_y"], cam["cy"] = cam["fl_y"]*h_scale, cam["cy"]*h_scale
        # The crop is taken in the original frame, and thus must also be updated
        cam["crop_x_min"] = int(cam["crop_x_min"] * w_scale)
        cam["crop_x_max"] = int(cam["crop_x_max"] * w_scale)
        cam["crop_y_min"] = int(cam["crop_y_min"] * h_scale)
        cam["crop_y_max"] = int(cam["crop_y_max"] * h_scale)

        intrinsic = np.array([[cam["fl_x"], 0, cam["cx"]],
                        [0, cam["fl_y"], cam["cy"]],
                        [0, 0, 1]])
        F = np.array(cam["transform_matrix"])
        # MVGenMaster uses y+down and z+forward, but my camera extrinsics are in NeRF format
        flip_ynz = np.diag([1, -1, -1, 1])
        # On the left flips world z and y
        # On the right rotates camera matrix to point the opposite way 
        extrinsic = np.linalg.inv(flip_ynz @ F @ flip_ynz) # MVGenMaster uses w2c 

        # Also need the crop intrinsics (They are already the right proportions by construction)
        zoomed_intrinsic = np.array([[cam["zoomed_fl_x"], 0, cam["zoomed_cx"]],
                                     [0, cam["zoomed_fl_y"], cam["zoomed_cy"]],
                                     [0, 0, 1]])
        
        camera = {"crop_size": (cam["crop_y_max"] - cam["crop_y_min"], cam["crop_x_max"] - cam["crop_x_min"]),
                  "crop_coords": (cam["crop_y_min"], cam["crop_x_min"]),
                  "name": cam["file_path"],
                  "zoomed_idx": i,
                  "wide_idc": i}
        
        cameras.append[camera]
        intrinsics.append(intrinsic)
        w2cs.append(extrinsic)
        zoomed_intrinsics.append(zoomed_intrinsic)

    # Store updated metadata file 
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)   

    # MVGenMaster expectes (N, 3, 3) for cameras 
    extrinsics = torch.stack([torch.tensor(w2c, dtype=torch.float32) for w2c in w2cs], dim=0)
    intrinsics = torch.stack([torch.tensor(K, dtype=torch.float32) for K in intrinsics], dim=0)
    zoomed_intrinsics = torch.stack([torch.tensor(K, dtype=torch.float32) for K in zoomed_intrinsics], dim=0)

    # MVGen is trained on scenes of specific sizes? 
    # Only the depth map and the camera extrinsic translations defines world coordinates
    if config.camera_longest_side is not None:
        c2ws = extrinsics.inverse()
        max_scale = torch.max(c2ws[:, :3, -1], dim=0)[0]
        min_scale = torch.min(c2ws[:, :3, -1], dim=0)[0]
        max_size = torch.max(max_scale - min_scale).item()
        rescale = config.camera_longest_side / max_size if max_size > config.camera_longest_side else 1.0
        # The translation for w2c is proportional to c2w 
        extrinsics[:, :3, 3:4] *= rescale
        ref_depth = rescale*depth_img
    
    return {"intrinsics": intrinsics, 
            "extrinsics": extrinsics, 
            "zoomed_intrinsics": zoomed_intrinsics, 
            "cameras": cameras,
            "shape": (h,w), 
            "ref": ref,
            "ref_depth": ref_depth, 
            "ref_image": img}
    
