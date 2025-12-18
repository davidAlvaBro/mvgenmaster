import argparse
import copy
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor, Compose, Normalize

from src.modules.schedulers import get_diffusion_scheduler
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline

from load_data import load_dataset

def eval(args, config, data, pipeline):
    # Get necessary data 
    cameras = data["cameras"]
    intrinsics = data["intrinsics"]
    zoomed_intrinsics = data["zoomed_intrinsics"]
    extrinsics = data["extrinsics"]

    # Extract reference cams 
    ref_cam = cameras[data["ref"]] # TODO check this works with multiple if I ever need it 
    ref_intrinsics = zoomed_intrinsics[ref_cam["zoomed_idx"]]
    ref_extrinsics = extrinsics[ref_cam["wide_idx"]]
    ref_depth = data["ref_depth"]
    img = data["ref_image"]
    (h,w) = data["shape"]
    n_to_gen = len(intrinsics)
    image_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ref_img = image_transform(img).unsqueeze(0)

    with torch.no_grad(), torch.autocast("cuda"):
        image = torch.cat([ref_img, torch.zeros((n_to_gen, 3, h, w), dtype=torch.float32)], dim=0).to("cuda")
        intrinsic = torch.cat([ref_intrinsics.unsqueeze(0), intrinsics], dim=0).to("cuda")
        extrinsic = torch.cat([ref_extrinsics.unsqueeze(0), extrinsics], dim=0).to("cuda")
        depth = torch.cat([ref_depth, torch.zeros((n_to_gen, 1, h, w), dtype=torch.float32)], dim=0).to("cuda")

        # Setting these, but they should not even be parameters 
        key_rescale = 1.2
        class_label = 0 

        nframe_new = n_to_gen + args.cond_num
        config_copy = copy.deepcopy(config)
        config_copy.nframe = nframe_new
        generator = torch.Generator()
        generator = generator.manual_seed(args.seed)

        wide_predictions = pipeline(images=image, nframe=nframe_new, cond_num=args.cond_num,
                            key_rescale=key_rescale, height=h, width=w, intrinsics=intrinsic,
                            extrinsics=extrinsic, num_inference_steps=70, guidance_scale=args.val_cfg,
                            output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                            class_label=class_label, depth=depth, vae=pipeline.vae, generator=generator,  start_from_step=None).images  # [f,h,w,c]
        wide_no_cond = wide_predictions[args.cond_num:] # TODO kinda want to see how the reference image looks
        wide_no_cond = (wide_no_cond * 255).astype(np.uint8)

        # Crop predicted images to "zoomed in" area 
        cropped_preds = np.zeros_like(wide_no_cond)
        for cam in cameras:
            crop_coords = cam["crop_coords"]
            crop_size = cam["crop_size"]
            crop = wide_no_cond[cam["wide_idx"], crop_coords[0]:crop_coords[0] + crop_size[0], crop_coords[1]:crop_coords[1] + crop_size[1]]
            cropped_preds[cam["wide_idx"]] = cv2.resize(crop, (w,h))
        
        new_in = torch.stack([image[0]] + [image_transform(pred).to("cuda") for pred in cropped_preds])
        intrinsic = torch.cat([ref_intrinsics.unsqueeze(0), zoomed_intrinsics], dim=0).to("cuda")

        # Run it again but with less applied noise 
        zoomed_pred = pipeline(images=new_in, nframe=nframe_new, cond_num=args.cond_num,
                            key_rescale=key_rescale, height=h, width=w, intrinsics=intrinsic,
                            extrinsics=extrinsic, num_inference_steps=80, guidance_scale=args.val_cfg,
                            output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                            class_label=class_label, depth=depth, vae=pipeline.vae, generator=generator, start_from_step=10).images  # [f,h,w,c]
        zoomed_pred_no_cond = zoomed_pred[args.cond_num:]
        zoomed_pred_no_cond = (zoomed_pred_no_cond * 255).astype(np.uint8)
        
        # NOTE Works better if I simply let the reference crop be generated... 
        # zoomed_pred_no_cond[ref_cam["zoomed_idx"]] = img
        
        # If store everthing save intermediate stages 
        if args.log_everything: 
            root_path = config.save_path
            (root_path / "zoom").mkdir(parents=True, exist_ok=True)
            (root_path / "crop").mkdir(parents=True, exist_ok=True)
            (root_path / "wide").mkdir(parents=True, exist_ok=True)
            for cam in cameras:
                name = Path(cam["name"]).name
                cv2.imwrite(root_path / "zoom" / name, zoomed_pred_no_cond[cam["wide_idx"], :, :, ::-1])
                cv2.imwrite(root_path / "crop" / name, cropped_preds[cam["wide_idx"], :, :, ::-1])
                cv2.imwrite(root_path / "wide" / name, wide_no_cond[cam["wide_idx"], :, :, ::-1])

        # Put in the newly inpainted people
        for cam in cameras:
            crop_coords = cam["crop_coords"]
            crop_size = cam["crop_size"]
            crop = cv2.resize(zoomed_pred_no_cond[cam["wide_idx"]], (crop_size[1], crop_size[0]))
            wide_no_cond[cam["wide_idx"], crop_coords[0]:crop_coords[0] + crop_size[0], crop_coords[1]:crop_coords[1] + crop_size[1]] = crop
            cv2.imwrite( config.save_path / cam["name"], wide_no_cond[cam["wide_idx"], :, :, ::-1])




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
    parser.add_argument("--log_everything", action='store_true', help="If set also saves the non-inpainted image, the crop and inpainting alone")
    # TODO are these even relevant at all? Look at debugging and kill them
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class_label", type=int, default=0)

    args = parser.parse_args()
    config = EasyDict(OmegaConf.load(os.path.join(args.model_dir, "config.yaml")))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda"

    save_path = Path(args.working_dir) / args.output_path
    save_path.mkdir(parents=True, exist_ok=True)
    args.cond_num = 1


    # Load models
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
    data = load_dataset(args=args, config=config)

    os.makedirs(f"{save_path}/images", exist_ok=True)
    eval(args, config, data, pipeline)
