from loader import Pointcloud
import visualize
from render import render_magic

name = "0104Amstel"

# cloud = Pointcloud(verbose=True)
# (cloud
#     .load_from_e57(f"{name}.e57")
#     .downscale(0.05)
#     .denoise()
#     .filter_main_cluster()
#     .save_to_ply(f"{name}.ply"))

# planes, inlier_points = cloud.find_planes(min_inliers=1000, inlier_distance_threshold=0.10, ransac_n=3, save=True, save_path=name)
 
# load pointcloud from already processed .ply
cloud = Pointcloud(verbose=True)
cloud.load_from_ply(f"{name}.ply")

chunks = cloud.chunks(chunk_size=2.5)
chunkresults = cloud.find_planes_chunked(chunks, min_inliers=100, inlier_distance_threshold=0.10, ransac_n=3)

planes, inlier_points = cloud.merge_chunk_planes(chunkresults, thresh=0.01)

# print(inlier_points)

# load ransac results from .npz file
# import numpy as np
# data = np.load(f"{name}.npz", allow_pickle=True)
# inlier_points = data["inlier_points"]
# planes = data["planes"]

# plane_0 = np.array(planes[0][:3], np.float)
# print(np.linalg.norm(planes[0][:3], axis=0))

# plane_0 = plane_0 / np.linalg.norm(planes[0][:3])

# print(plane_0)

# plane_idx = 0
# render_magic(inlier_points[plane_idx], planes[plane_idx][:3])
# visualize.draw_plane(inlier_points[plane_idx], planes[plane_idx][:3])
# print(chunkresults.flatten())

# visualize.draw_planes_in_one(chunkresults.flatten()[1])
visualize.draw_planes_in_one(inlier_points)