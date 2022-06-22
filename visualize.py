import numpy as np
import open3d as o3d
from loader import Pointcloud


def draw_planes(inlier_points, max_amount=10):
    # Pass xyz to o3d pc and visualize
    for i, curr_inliers in enumerate(inlier_points):
        if i >= max_amount:
            break

        # color = np.random.choice(range(256), size=3)/255
        # colors = np.repeat([color], len(inliers[i]), axis=0)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = Pointcloud.np_to_o3d(curr_points)

        o3d.visualization.draw_geometries([pcd])


def draw_planes_in_one(inlier_points):
    # Pass xyz to o3d pc and visualize
    pcds = []
    for i, curr_inliers in enumerate(inlier_points):
        pcd = Pointcloud.np_to_o3d(curr_inliers)

        color = np.random.choice(range(256), size=3)/255
        colors = np.repeat([color], len(curr_inliers), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)
