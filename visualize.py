import ast
import numpy as np
import pye57
import open3d as o3d

# ############# TEMP #############
DOWNSAMPLE = 100

e57 = pye57.E57("0104Amstel.e57")

# read scan at index 0
data = e57.read_scan(0, ignore_missing_fields=True)

# 'data' is a dictionary with the point types as keys
points = np.array([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).T

# downsample
points = points[::DOWNSAMPLE]
# ############# TEMP #############

with open("ransac.txt", "r") as file:
    planes = next(file)
    inliers = next(file)

planes = ast.literal_eval(planes)
inliers = ast.literal_eval(inliers)


# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
for i, curr_inliers in enumerate(inliers[:3]):
    color = np.random.choice(range(256), size=3)/255
    pcd = o3d.geometry.PointCloud()

    # colors = np.repeat([color], len(inliers[i]), axis=0)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.points = o3d.utility.Vector3dVector(points[inliers[i]])
    o3d.visualization.draw_geometries([pcd])
                        