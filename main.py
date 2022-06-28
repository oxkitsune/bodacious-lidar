from loader import Pointcloud, cluster_results
import visualize
from render import render_magic
import numpy as np
import pickle

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
cloud = Pointcloud(verbose=False)
cloud.load_from_ply(f"{name}.ply")

chunks = cloud.chunks(chunk_size=1)
chunked_result = cloud.find_planes_chunked(chunks, min_inliers=10, inlier_distance_threshold=0.01, ransac_n=3)
result = cloud.merge_chunk_planes(chunked_result, thresh=0.95)

result = cluster_results(result, eps=0.35, min_points=100)

# load ransac results from .npz file
# import numpy as np
# data = np.load(f"{name}.npz", allow_pickle=True)
# inlier_points = data["inlier_points"]
# planes = data["planes"]

# renders all planes in their own visualization
# visualize.draw_planes(result.points, max_amount=2)

# renders all planes in their own visualization with the other planes greyed out
# visualize.draw_planes_highlighted(result_split_points, index=0)

# renders all planes in one visualization
# visualize.draw_planes_in_one(result.points)

# generates views for every plane and saves result as a .PNG
visualize.generate_views(result, max_images=50)
