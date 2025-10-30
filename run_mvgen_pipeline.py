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
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
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


def load_dataset(args, config, reference_cam, target_cam, reference_list, ref_numbers, depth_list):
    ratio_set = json.load(open(f"./{args.model_dir}/ratio_set.json", "r"))
    ratio_dict = dict()
    for h, w in ratio_set:
        ratio_dict[h / w] = [h, w]
    ratio_list = list(ratio_dict.keys())

    # load dataset
    print("Loading dataset...")
    intrinsic = np.array(reference_cam["intrinsic"])
    tar_names = list(target_cam["extrinsic"].keys())
    tar_names.sort()
    if args.target_limit is not None:
        tar_names = tar_names[:args.target_limit]
    tar_extrinsic = [np.array(target_cam["extrinsic"][k]) for k in tar_names]

    if args.cond_num == 1:
        reference_list = [reference_list[0]]
    elif args.cond_num == 2:
        reference_list = [reference_list[0], reference_list[-1]]
    elif args.cond_num == 3:
        reference_list = reference_list[:3]
    else:
        pass

    ref_images = []
    ref_names = []
    ref_extrinsic = []
    ref_intrinsic = []
    ref_depth = []
    h, w = None, None
    for i, im in enumerate(tqdm(reference_list, desc="loading reference images")):
        img = Image.open(im).convert("RGB")
        intrinsic_ = copy.deepcopy(intrinsic)
        im = f"view{str(ref_numbers[i]).zfill(3)}_ref"
        if im.split("/")[-1] in reference_cam["extrinsic"]:
            extrinsic_ = np.array(reference_cam["extrinsic"][im.split("/")[-1]])
        else:
            extrinsic_ = np.array(reference_cam["extrinsic"][im.split("/")[-1].split(".")[0]])
        ref_extrinsic.append(extrinsic_)
        ref_names.append(im.split('/')[-1])

        origin_w, origin_h = img.size

        # load monocular depth
        if config.model_cfg.get("enable_depth", False):
            depth = depth_list[i]
            depth = cv2.resize(depth, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
        else:
            depth = None

        if h is None or w is None:
            ratio = origin_h / origin_w
            sub = [abs(ratio - r) for r in ratio_list]
            [h, w] = ratio_dict[ratio_list[np.argmin(sub)]]
            print(f'height:{h}, width:{w}.')
        img = img.resize((w, h), Image.LANCZOS if h < origin_h else Image.BICUBIC)
        if depth is not None:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        new_w, new_h = img.size
        # rescale intrinsic
        intrinsic_[0, :] *= (new_w / reference_cam['w'])
        intrinsic_[1, :] *= (new_h / reference_cam['h'])

        img = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img)
        if depth is not None:
            depth = Compose([ToTensor()])(depth)

        ref_images.append(img)
        ref_intrinsic.append(intrinsic_)
        if depth is not None:
            ref_depth.append(depth)

    ref_images = torch.stack(ref_images, dim=0)
    tar_intrinsic = [ref_intrinsic[0]] * len(tar_extrinsic)
    ref_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in ref_intrinsic], dim=0)
    tar_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_intrinsic], dim=0)
    ref_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in ref_extrinsic], dim=0)
    tar_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_extrinsic], dim=0)

    # 外参t归一化
    if config.camera_longest_side is not None:
        extrinsic = torch.cat([ref_extrinsic, tar_extrinsic], dim=0)  # [N,4,4]
        c2ws = extrinsic.inverse()
        max_scale = torch.max(c2ws[:, :3, -1], dim=0)[0]
        min_scale = torch.min(c2ws[:, :3, -1], dim=0)[0]
        max_size = torch.max(max_scale - min_scale).item()
        rescale = config.camera_longest_side / max_size if max_size > config.camera_longest_side else 1.0
        ref_extrinsic[:, :3, 3:4] *= rescale
        tar_extrinsic[:, :3, 3:4] *= rescale
    else:
        rescale = 1.0

    if len(ref_depth) > 0:
        ref_depth = [r * rescale for r in ref_depth]
        ref_depth = torch.stack(ref_depth, dim=0)
    else:
        ref_depth = None

    camera_poses = {"h": h, "w": w, "intrinsic": ref_intrinsic[0].numpy().tolist(), "extrinsic": dict()}
    for i in range(len(ref_names)):
        camera_poses['extrinsic'][ref_names[i].split('.')[0].replace('_ref', '') + ".png"] = ref_extrinsic[i].numpy().tolist()
    for i in range(len(tar_names)):
        camera_poses['extrinsic'][tar_names[i].split('.')[0].replace('_ref', '') + ".png"] = tar_extrinsic[i].numpy().tolist()

    return {"ref_images": ref_images, "ref_intrinsic": ref_intrinsic, "tar_intrinsic": tar_intrinsic,
            "ref_extrinsic": ref_extrinsic, "tar_extrinsic": tar_extrinsic, "ref_depth": ref_depth,
            "ref_names": ref_names, "tar_names": tar_names, "h": h, "w": w}


def eval(args, config, data, pipeline, data_args: dict):
    # Bookkeeping for GS pipeline (make a new transforms.json)
    new_transform = data_args.copy()
    new_transform.pop("trajectory")
    new_transform["frames"] = data_args["trajectory"]
    parent_path = Path(config.save_path)
    file_names = [] 
    # names of images and aspect ratios might change 
    # We also have to do it for the evaluation frames so that we can messure PSNR drop
    for i, frame in enumerate(new_transform["frames"] + new_transform["eval"]): 
        if frame["file_path"] is not None: 
            file_name = frame["file_path"]
        else: 
            file_name = f"images/view{i}.png"
            frame["file_path"] = file_name 
        frame["w"] = data["w"]
        frame["h"] = data["h"]
        file_names.append(str(parent_path / file_name))
    

    N_target = data['tar_intrinsic'].shape[0]
    gen_num = config.nframe - args.cond_num

    # Save reference image 
    ref_img = ToPILImage()((data['ref_images'][0] + 1) / 2)
    ref_img.save(file_names[new_transform["trajectory_ref"]])
    file_names_without_reference = file_names[:new_transform["trajectory_ref"]] + file_names[new_transform["trajectory_ref"]+1:]

    with torch.no_grad(), torch.autocast("cuda"):
        # Only generating 'gen_num' new frames each iteration
        iter_times = N_target // gen_num
        if N_target % gen_num != 0:
            iter_times += 1
        for i in range(iter_times):
            current_views = np.arange(N_target)[i::iter_times].tolist()
            print(f"synthesis target views {current_views}...")
            h, w = data['ref_images'].shape[2], data['ref_images'].shape[3]
            gen_num_ = len(current_views)
            image = torch.cat([data["ref_images"], torch.zeros((gen_num_, 3, h, w), dtype=torch.float32)], dim=0).to("cuda")
            intrinsic = torch.cat([data["ref_intrinsic"], data["tar_intrinsic"][i::iter_times]], dim=0).to("cuda")
            extrinsic = torch.cat([data["ref_extrinsic"], data["tar_extrinsic"][i::iter_times]], dim=0).to("cuda")
            if data["ref_depth"] is not None:
                depth = torch.cat([data["ref_depth"], torch.zeros((gen_num_, 1, h, w), dtype=torch.float32)], dim=0).to("cuda")
                # Saving the depth map so that it can be used for a sparse point cloud in the GS pipeline 
                depth_ref_to_save = data["ref_depth"].detach().to("cpu").numpy().squeeze()
                depth_map_path = "ref_depth_map.npy"
                np.save(parent_path / depth_map_path, depth_ref_to_save)  
                new_transform["depth_map"] = str(depth_map_path)
            else:
                depth = None

            nframe_new = gen_num_ + args.cond_num
            config_copy = copy.deepcopy(config)
            config_copy.nframe = nframe_new
            generator = torch.Generator()
            generator = generator.manual_seed(args.seed)
            st = time.time()

            # Print the point cloud projection - TODO remove? also why is this not passed to the pipeline?
            if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
                color_warps = global_position_encoding_3d(config_copy, depth, intrinsic, extrinsic,
                                                          args.cond_num, nframe=nframe_new, device=device,
                                                          pe_scale=1 / 8, embed_dim=config.model_cfg.get("coord_dim", 192),
                                                          colors=image)[0]

                cv2.imwrite(f"{config.save_path}/warp{current_views}.png", color_warps[:, :, ::-1])

            preds = pipeline(images=image, nframe=nframe_new, cond_num=args.cond_num,
                             key_rescale=args.key_rescale, height=h, width=w, intrinsics=intrinsic,
                             extrinsics=extrinsic, num_inference_steps=50, guidance_scale=args.val_cfg,
                             output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                             class_label=args.class_label, depth=depth, vae=pipeline.vae, generator=generator).images  # [f,h,w,c]
            print("Time used:", time.time() - st)
            preds = preds[args.cond_num:] # TODO kinda want to see how the reference image looks
            preds = (preds * 255).astype(np.uint8)

            # Store images 
            for j in range(preds.shape[0]):
                cv2.imwrite(file_names_without_reference[current_views[j]], preds[j, :, :, ::-1])
    
    # Store the transforms.json for the GS pipeline 
    with open(parent_path / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(new_transform, f, ensure_ascii=False, indent=2)      

    return file_names 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build cam traj")
    parser.add_argument("--working_dir", type=str, default="../data/set2")
    parser.add_argument("--input_path", type=str, default="../data/set2/transforms.json")
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
    save_path.mkdir(parents=True, exist_ok=True)
    args.cond_num = 1

    with open(args.input_path, 'r') as file:
        data_args = json.load(file)

    # Cameras and reference image
    image, img_pth, ref_n, extrinsics, intrinsics = load_cameras(Path(args.working_dir), data_args)
    h, w, _ = image.shape
    c2ws_all = [torch.tensor(ex, dtype=torch.float32) for ex in extrinsics]
    w2cs_all = [c2w.inverse() for c2w in c2ws_all]

    # Get depth for 3d point cloud 
    depth_pro_model, transform = create_model_and_transforms(device=torch.device("cuda"))
    depth_pro_model.eval()
    image = transform(image)    
    prediction = depth_pro_model.infer(image, f_px=intrinsics[ref_n][0,0]) # Depth model wants focal length
    depth = prediction["depth"]  # Depth in [m].

    K = torch.tensor(intrinsics[ref_n], dtype=torch.float32, device=device)
    K_inv = K.inverse()

    # 3d sparse point cloud
    points2d = torch.stack(torch.meshgrid(torch.arange(w, dtype=torch.float32),
                                            torch.arange(h, dtype=torch.float32), indexing="xy"), -1).to(device)  # [h,w,2]
    points3d = points_padding(points2d).reshape(h * w, 3)  # [hw,3]
    points3d = (K_inv @ points3d.T * depth.reshape(1, h * w)).T
    colors = ((image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(h * w, 3)
    points3d = points3d.cpu().numpy()
    colors = colors.cpu().numpy()

    # save pointcloud and cameras
    scene = trimesh.Scene()
    for i in range(len(c2ws_all)):
        add_scene_cam(scene, c2ws_all[i], CAM_COLORS[i % len(CAM_COLORS)], None, imsize=(512, 512), screen_width=0.03)

    pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    _ = pcd.export(f"{save_path}/pcd.ply")
    scene.export(file_obj=f"{save_path}/cameras.glb")

    reference_cam = {"h": h, "w": w, "intrinsic": K.tolist()}
    reference_cam["extrinsic"] = dict()
    target_cam = copy.deepcopy(reference_cam)
    reference_cam["extrinsic"][f"view{str(ref_n).zfill(3)}_ref"] = w2cs_all[ref_n].tolist()

    for i in range(len(w2cs_all)):
        if i != ref_n:
            target_cam["extrinsic"][f"view{str(i).zfill(3)}"] = w2cs_all[i].tolist()

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
    data = load_dataset(args, config, reference_cam, target_cam, [img_pth], [ref_n], [depth.cpu().numpy()])

    os.makedirs(f"{save_path}/images", exist_ok=True)
    results = eval(args, config, data, pipeline, data_args)

    clip = ImageSequenceClip(results, fps=15)
    clip.write_videofile(f"{config.save_path}/output.mp4", fps=15)
