import argparse
import copy
import json
import os
import random
import time
from glob import glob
from pathlib import Path

import cv2
import imagesize
import numpy as np
import torch
import trimesh
from PIL import Image
from diffusers import AutoencoderKL
from easydict import EasyDict
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, Lambda, ConvertImageDtype
from tqdm import tqdm
from depth_pro.depth_pro import create_model_and_transforms
from depth_pro.utils import load_rgb

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from src.modules.cam_vis import add_scene_cam
from src.modules.position_encoding import global_position_encoding_3d
from src.modules.schedulers import get_diffusion_scheduler
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from camera import load_cameras

CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def np_points_padding(points):
    padding = np.ones_like(points)[..., 0:1]
    points = np.concatenate([points, padding], axis=-1)
    return points


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def save_16bit_png_depth(depth: np.ndarray, depth_png: str):
    # Ensure the numpy array's dtype is float32, then cast to float16, and finally reinterpret as uint16
    depth_uint16 = np.array(depth, dtype=np.float32).astype(np.float16).view(np.uint16)

    # Create a PIL Image from the 16-bit depth values and save it
    depth_pil = Image.fromarray(depth_uint16)

    if not depth_png.endswith(".png"):
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError

    try:
        depth_pil.save(depth_png)
    except:
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError

def load_dataset(args, config, reference_cam, target_cams, reference_img, depth_img):
    """
    Preprocess camera parameters and image size to fit the MVGenMaster network
    """
    # Valid ratios 
    ratio_set = json.load(open(f"./{args.model_dir}/ratio_set.json", "r"))
    ratio_dict = dict()
    for h, w in ratio_set:
        ratio_dict[h / w] = [h, w]
    ratio_list = list(ratio_dict.keys())
    h_img, w_img, _ = reference_img.shape
    print(f'Original image shape is height:{h_img}, width:{w_img}.')
    ratio = h_img / w_img
    sub = [abs(ratio - r) for r in ratio_list]
    [h, w] = ratio_dict[ratio_list[np.argmin(sub)]]
    print(f'Closest valid ratio is height:{h}, width:{w}.')
    # Resize image and depth 
    img = Image.fromarray(reference_img)
    img = img.resize((w, h), Image.LANCZOS if h < h_img else Image.BICUBIC)
    depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
    img = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img).unsqueeze(0)
    depth_img = Compose([ToTensor()])(depth_img).unsqueeze(0)

    # Fix reference camera parameters 
    ref_names = list(reference_cam.keys())
    ref_intrinsic = np.array(reference_cam[ref_names[0]]["intrinsic"])
    ref_extrinsic = np.array(reference_cam[ref_names[0]]["extrinsic"])
    ref_h, ref_w = reference_cam[ref_names[0]]["h"], reference_cam[ref_names[0]]["w"]
    ref_intrinsic[0] *= w/ref_w
    ref_intrinsic[1] *= h/ref_h

    # Fix target frames camera parameters 
    tar_names = list(target_cams.keys())
    tar_names.sort()
    tar_extrinsic = []
    tar_intrinsic = []
    for tar_name in tar_names:
        cam = target_cams[tar_name]
        intrinsic = cam["intrinsic"]
        extrinsic = cam["extrinsic"]
        tar_h, tar_w = cam["h"], cam["w"]
        intrinsic[0] *= w/tar_w
        intrinsic[1] *= h/tar_h
        tar_extrinsic.append(extrinsic)
        tar_intrinsic.append(intrinsic)

    # MVGenMaster expectes (N, 3, 3) for cameras 
    ref_extrinsic = torch.tensor(ref_extrinsic, dtype=torch.float32).unsqueeze(0)
    ref_intrinsic = torch.tensor(ref_intrinsic, dtype=torch.float32).unsqueeze(0)
    tar_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_intrinsic], dim=0)
    tar_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_extrinsic], dim=0)

    if config.camera_longest_side is not None:
        # MVGen is trained on scenes of specific sizes? 
        # Only the depth map and the camera extrinsic translations defines world coordinates
        extrinsic = torch.cat([ref_extrinsic, tar_extrinsic], dim=0)  # [N,4,4]
        c2ws = extrinsic.inverse()
        max_scale = torch.max(c2ws[:, :3, -1], dim=0)[0]
        min_scale = torch.min(c2ws[:, :3, -1], dim=0)[0]
        max_size = torch.max(max_scale - min_scale).item()
        rescale = config.camera_longest_side / max_size if max_size > config.camera_longest_side else 1.0
        # The translation for w2c is proportional to c2w 
        ref_extrinsic[:, :3, 3:4] *= rescale
        tar_extrinsic[:, :3, 3:4] *= rescale
        ref_depth = rescale*depth_img
    else:
        rescale = 1.0

    # Names that could be deleted
    camera_poses = {"h": h, "w": w, "intrinsic": ref_intrinsic[0].numpy().tolist(), "extrinsic": dict()}
    for i in range(len(ref_names)):
        camera_poses['extrinsic'][ref_names[i].split('.')[0].replace('_ref', '') + ".png"] = ref_extrinsic[i].numpy().tolist()
    for i in range(len(tar_names)):
        camera_poses['extrinsic'][tar_names[i].split('.')[0].replace('_ref', '') + ".png"] = tar_extrinsic[i].numpy().tolist()

    return {"ref_images": img, "ref_intrinsic": ref_intrinsic, "tar_intrinsic": tar_intrinsic,
            "ref_extrinsic": ref_extrinsic, "tar_extrinsic": tar_extrinsic, "ref_depth": ref_depth,
            "ref_names": ref_names, "tar_names": tar_names, "h": h, "w": w, "scale": rescale}


def eval(args, config, data, pipeline, data_args: dict):
    # Bookkeeping for GS pipeline (make a new transforms.json)
    parent_path = Path(config.save_path)
    file_names = [] 
    # names of images and aspect ratios might change 
    # We also have to do it for the evaluation frames so that we can messure PSNR drop
    w, h = data["w"], data["h"]
    for i, frame in enumerate(data_args["frames"] + data_args["eval"]): 
        if frame["file_path"] is not None: 
            file_name = frame["file_path"]
        else: 
            file_name = f"images/view{i}.png"
            frame["file_path"] = file_name 
        old_h, old_w = frame["h"], frame["w"]
        scale_h, scale_w = h / old_h, w / old_w
        frame["w"] = w
        frame["h"] = h
        frame["fl_x"] = frame["fl_x"] * scale_w
        frame["fl_y"] = frame["fl_y"] * scale_h
        frame["cx"] = frame["cx"] * scale_w
        frame["cy"] = frame["cy"] * scale_h

        file_names.append(str(parent_path / file_name))
    

    N_target = data['tar_intrinsic'].shape[0]
    gen_num = config.nframe - args.cond_num

    # Save reference image 
    ref_img = ToPILImage()((data['ref_images'][0] + 1) / 2)
    ref_img.save(file_names[data_args["ref"]])
    file_names_without_reference = file_names[:data_args["ref"]] + file_names[data_args["ref"]+1:]

    # Save depth map path in transforms.json
    depth_map_path = "ref_depth_map.npy"
    data_args["depth_map"] = str(depth_map_path)

    # Store the transforms.json for the GS pipeline 
    with open(parent_path / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data_args, f, ensure_ascii=False, indent=2)   

    with torch.no_grad(), torch.autocast("cuda"):
        h, w = data['ref_images'].shape[2], data['ref_images'].shape[3]
        image = torch.cat([data["ref_images"], torch.zeros((N_target, 3, h, w), dtype=torch.float32)], dim=0).to("cuda")
        intrinsic = torch.cat([data["ref_intrinsic"], data["tar_intrinsic"]], dim=0).to("cuda")
        extrinsic = torch.cat([data["ref_extrinsic"], data["tar_extrinsic"]], dim=0).to("cuda")
        if data["ref_depth"] is not None:
            depth = torch.cat([data["ref_depth"], torch.zeros((N_target, 1, h, w), dtype=torch.float32)], dim=0).to("cuda")
        else:
            depth = None

        nframe_new = N_target + args.cond_num
        config_copy = copy.deepcopy(config)
        config_copy.nframe = nframe_new
        generator = torch.Generator()
        generator = generator.manual_seed(args.seed)
        st = time.time()

        preds = pipeline(images=image, nframe=nframe_new, cond_num=args.cond_num,
                            key_rescale=args.key_rescale, height=h, width=w, intrinsics=intrinsic,
                            extrinsics=extrinsic, num_inference_steps=50, guidance_scale=args.val_cfg,
                            output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                            class_label=args.class_label, depth=depth, vae=pipeline.vae, generator=generator).images  # [f,h,w,c]
        print("Time used:", time.time() - st)
        preds_without_ref = preds[args.cond_num:] # TODO kinda want to see how the reference image looks
        preds_without_ref = (preds_without_ref * 255).astype(np.uint8)

        # Store images 
        for j in range(preds_without_ref.shape[0]):
            cv2.imwrite(file_names_without_reference[j], preds_without_ref[j, :, :, ::-1])

        # Run it again but with less applied noise 
        preds = pipeline(images=preds, nframe=nframe_new, cond_num=args.cond_num,
                            key_rescale=args.key_rescale, height=h, width=w, intrinsics=intrinsic,
                            extrinsics=extrinsic, num_inference_steps=50, guidance_scale=args.val_cfg,
                            output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                            class_label=args.class_label, depth=depth, vae=pipeline.vae, generator=generator, start_from_step=25).images  # [f,h,w,c]
        preds_without_ref = preds[args.cond_num:]
        preds_without_ref = (preds_without_ref * 255).astype(np.uint8)
        for j in range(preds_without_ref.shape[0]):
            cv2.imwrite(file_names_without_reference[j][:-4] + "_2.png", preds_without_ref[j, :, :, ::-1])

    return file_names 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build cam traj")
    # parser.add_argument("--working_dir", type=str, default="../data/test")
    # parser.add_argument("--input_path", type=str, default="../data/test/transforms.json")
    parser.add_argument("--working_dir", type=str, default="../data/success1/")
    parser.add_argument("--input_path", type=str, default="../data/success1/transforms.json")
    parser.add_argument("--added_img_path", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="check_points/pretrained_model", help="model directory.")
    parser.add_argument("--output_path", type=str, default="mvgen")
    parser.add_argument("--val_cfg", type=float, default=2.0)
    parser.add_argument("--key_rescale", type=float, default=None)
    parser.add_argument("--camera_longest_side", type=float, default=5.0)
    parser.add_argument("--nframe", type=int, default=28)
    parser.add_argument("--min_conf_thr", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument("--target_limit", type=int, default=None)
    # single-view parameters
    parser.add_argument("--center_scale", type=float, default=1.0)
    parser.add_argument("--elevation", type=float, default=5.0, help="the initial elevation angle")
    parser.add_argument("--d_theta", type=float, default=0.0, help="elevation rotation angle")
    parser.add_argument("--d_phi", type=float, default=45.0, help="azimuth rotation angle")
    parser.add_argument("--d_r", type=float, default=1.0, help="the distance from camera to the world center")
    parser.add_argument("--x_offset", type=float, default=0.0, help="up moving")
    parser.add_argument("--y_offset", type=float, default=0.0, help="left moving")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--foreground", action="store_true")

    args = parser.parse_args()
    config = EasyDict(OmegaConf.load(os.path.join(args.model_dir, "config.yaml")))
    if config.nframe != args.nframe:
        print(f"Extend nframe from {config.nframe} to {args.nframe}.")
        config.nframe = args.nframe
        if config.nframe > 28 and args.key_rescale is None:
            args.key_rescale = 1.2
        print("key rescale", args.key_rescale)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda"

    save_path = Path(args.working_dir) / args.output_path
    ref_img_folder = Path(args.working_dir) if args.added_img_path == "" else Path(args.working_dir) / args.added_img_path
    save_path.mkdir(parents=True, exist_ok=True)
    args.cond_num = 1

    with open(args.input_path, 'r') as file:
        data_args = json.load(file)

    # Cameras and reference image # TODO fix this mess...
    img, depth, ref_n, extrinsics, intrinsics, Hs, Ws, view_names = load_cameras(ref_img_folder, data_args)
    depth = torch.tensor(depth, device=device)
    h, w = Hs[ref_n], Ws[ref_n] # TODO check this is correct
    c2ws_all = [torch.tensor(ex, dtype=torch.float32) for ex in extrinsics]
    w2cs_all = [c2w.inverse() for c2w in c2ws_all]
    Ks = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    K_invs = Ks.inverse()

    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(torch.float32),
        ]
    )

    image = transform(img)

    # 3d sparse point cloud
    points2d = torch.stack(torch.meshgrid(torch.arange(w, dtype=torch.float32),
                                            torch.arange(h, dtype=torch.float32), indexing="xy"), -1).to(device)  # [h,w,2]
    points3d = points_padding(points2d).reshape(h * w, 3)  # [hw,3]
    points3d = (K_invs[ref_n] @ points3d.T * depth.reshape(1, h * w)).T
    colors = ((image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(h * w, 3)
    points3d = points3d.cpu().numpy()
    colors = colors.cpu().numpy()

    # save pointcloud 
    # TODO understand what the hell happenes here? 
    scene = trimesh.Scene()
    for i in range(len(c2ws_all)):
        add_scene_cam(scene, c2ws_all[i], CAM_COLORS[i % len(CAM_COLORS)], None, imsize=(512, 512), screen_width=0.03)

    pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    _ = pcd.export(f"{save_path}/pcd.ply")
    scene.export(file_obj=f"{save_path}/cameras.glb")

    # Each camera has its own intrinsics and extrinsics
    target_cams = {}
    for (h, w, e, i, n) in zip(Hs, Ws, w2cs_all, intrinsics, view_names): 
        target_cams[n] = {'h': h, 
                          'w': w,
                          'extrinsic': e,
                          'intrinsic': i}
    reference_cam = {view_names[ref_n] + "_ref": target_cams.pop(view_names[ref_n])}

    ### Step2: generate multi-view images ###
    # init model
    print("load model...")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path,
                                        subfolder="vae", local_files_only=True)
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                rank=0,
                                                model_cfg=config.model_cfg,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True,
                                                local_files_only=True)
    unet.requires_grad_(False)
    # load pretained weights
    weights = torch.load(f"{args.model_dir}/ema_unet.pt", map_location="cpu")
    unet.load_state_dict(weights)
    unet.eval()

    weight_dtype = torch.float16
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    scheduler = get_diffusion_scheduler(config, name="DDIM")
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipeline = pipeline.to(device)

    # load dataset
    args.dataset_dir = save_path
    config.save_path = save_path
    data = load_dataset(args, config, reference_cam, target_cams, img, depth.cpu().numpy())

    os.makedirs(f"{save_path}/images", exist_ok=True)
    results = eval(args, config, data, pipeline, data_args)
