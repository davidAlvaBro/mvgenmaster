from pathlib import Path

import numpy as np 
from PIL import Image

def load_cameras(parent_dir: Path, args: dict): 
    """
    This method loads in all camera to world matrices for a given .json,
    along with the reference image
    """
    # The reference image is selected 
    ref = args["ref"]
    ref_cam = args["frames"][ref] 
    img_pth = parent_dir / ref_cam["file_path"]
    img_pil = Image.open(img_pth)
    image = np.array(img_pil)
    depth_pth = parent_dir / ref_cam["depth_path"]
    depth = np.load(depth_pth)

    intrinsics = [] 
    extrinsics = []
    Hs = []
    Ws = []
    names = []
    for i, cam in enumerate(args["frames"] + args["eval"]): 
        Hs.append(cam["h"])
        Ws.append(cam["w"])
        intrinsic = np.array([[cam["fl_x"], 0, cam["cx"]],
                        [0, cam["fl_y"], cam["cy"]],
                        [0, 0, 1]])
        F = np.array(cam["transform_matrix"])
        # MVGenMaster uses y+down and z+forward, but my camera extrinsics are in NeRF format
        flip_ynz = np.diag([1, -1, -1, 1])
        # On the left flips world z and y
        # On the right rotates camera matrix to point the opposite way 
        extrinsics.append(flip_ynz @ F @ flip_ynz)
        intrinsics.append(intrinsic) 
        names.append(f"view{str(i).zfill(3)}")

    return image, depth, ref, extrinsics, intrinsics, Hs, Ws, names
