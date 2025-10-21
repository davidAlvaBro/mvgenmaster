from pathlib import Path
import json

import numpy as np 

from depth_pro.utils import load_rgb

def load_cameras(camera_path: Path): 
    """
    This method loads in all camera to world matrices for a given .json,
    then translates them to 'first camera view' to 'other camera view'.
    """
    with open(camera_path, 'r') as file:
        args = json.load(file)
    
    extrinsics = []
    intrinsics = []
    for cam in args["frames"]: 
        K = np.array([[cam["fl_x"], 0, cam["cx"]],
                      [0, cam["fl_y"], cam["cy"]],
                      [0, 0, 1]])
        F = np.array(cam["transform_matrix"])
        flip_y = np.diag([1, -1, 1])
        flip_z = np.diag([1, 1, -1])
        # F[:3,:3] = F[:3,:3] @ np.diag([-1,1,-1]) # my view convention is flipped 
        F[:3,:3] = flip_y @ F[:3,:3] @ flip_y
        F[:3,3] = flip_y @ F[:3,3]
        F[:3,:3] = flip_z @ F[:3,:3] @ flip_z
        F[:3,3] = flip_z @ F[:3,3]
        extrinsics.append(F) 
        intrinsics.append(K)
    
    # The reference image is selected 
    ref_n = args["ref_img"]
    ref_cam = args["frames"][ref_n] 
    img_pth = ref_cam["file_path"]
    image, _, _ = load_rgb(img_pth)
    return image, img_pth, ref_n, extrinsics, intrinsics
    # ref_extrinsics = extrinsics.pop(ref_n)
    # ref_intrinsics = intrinsics.pop(ref_n)

    # Calculate c2c extrinsics for other views than the reference image
    # c2c = [ref_extrinsics @ np.linalg.inv(ex) for ex in extrinsics] # or should I output this? 

    # return image, ref_extrinsics, ref_intrinsics, extrinsics, intrinsics