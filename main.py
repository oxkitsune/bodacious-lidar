from loader import Pointcloud
import visualize

name = "0104Amstel"

cloud = Pointcloud(verbose=True)
(cloud
    .load_from_e57(f"{name}.e57")
    .downscale(0.05)
    .denoise()
    .filter_main_cluster()
    .save_to_ply(f"{name}.ply"))

planes, inlier_points = cloud.find_planes(min_inliers=1000, inlier_distance_threshold=0.10, ransac_n=3, save=True, save_path=name)

# load pointcloud from already processed .ply
# cloud = Pointcloud(verbose=True)
# cloud.load_from_ply(f"{name}.e57")

# load ransac results from .npz file
# import numpy as np
# data = np.load(f"{name}.npz", allow_pickle=True)
# inlier_points = data["inlier_points"]

visualize.draw_planes_in_one(inlier_points)
