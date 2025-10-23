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
    # The reference image is selected 
    ref = args["ref"]
    ref_cam = args["frames"][ref] 
    img_pth = ref_cam["file_path"]
    image, _, _ = load_rgb(img_pth)
    ref_intrinsics = np.array([[ref_cam["fl_x"], 0, ref_cam["cx"]],
                      [0, ref_cam["fl_y"], ref_cam["cy"]],
                      [0, 0, 1]])
    ref_n = args["trajectory_ref"]

    extrinsics = []
    for cam in args["trajectory"]: 
        F = np.array(cam)
        # MVGenMaster uses y+down and z+forward, but my camera extrinsics are in NeRF format
        flip_ynz = np.diag([1, -1, -1, 1])
        # On the left flips world z and y
        # On the right rotates camera matrix to point the opposite way 
        extrinsics.append(flip_ynz @ F @ flip_ynz) 
    
    # NOTE : Since these are generated camera views I simply copy the reference cameras intrinsics 
    intrinsics = [ref_intrinsics]*len(extrinsics)
    
    return image, img_pth, ref_n, extrinsics, intrinsics