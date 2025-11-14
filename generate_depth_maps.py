import argparse
import json
from pathlib import Path

from tqdm import tqdm 
import numpy as np
import torch

from depth_pro.depth_pro import create_model_and_transforms
from depth_pro.utils import load_rgb



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build cam traj")
    # parser.add_argument("--working_dir", type=str, default="../data/test")
    # parser.add_argument("--input_path", type=str, default="../data/test/transforms.json")
    parser.add_argument("--working_dir", type=str, default="../data/pointcloud/")
    parser.add_argument("--input_path", type=str, default="../data/pointcloud/transforms.json")
    args = parser.parse_args()
    
    device = "cuda"

    working_dir = Path(args.working_dir) 
    (working_dir / "depths").mkdir(parents=True, exist_ok=True)

    with open(args.input_path, 'r') as file:
        metadata = json.load(file)


    # Get depth for 3d point cloud 
    depth_pro_model, transform = create_model_and_transforms(device=torch.device("cuda"))
    depth_pro_model.eval()
    for frame in tqdm(metadata["frames"]): 
        K = np.array([[frame["fl_x"], 0, frame["cx"]], 
                      [0, frame["fl_y"], frame["cy"]], 
                      [0, 0, 1]])
        
        image_name = frame["file_path"] 
        image, _, _ = load_rgb(working_dir / image_name)
        image = transform(image)

        prediction = depth_pro_model.infer(image, f_px=K[0,0])
        depth = prediction["depth"]

        depth_file_name = Path("depths") / (Path(image_name).stem + (".npy"))
        frame["depth_path"] = str(depth_file_name)
        np.save(working_dir / depth_file_name, depth.cpu().numpy())

    with open(args.input_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)