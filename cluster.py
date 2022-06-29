import numpy as np
from ransac import RansacPointResult
from loader import Pointcloud

def cluster_results(result, eps=0.5, min_points=25):
    """Clusters RANSAC results to separate similar-angled planes that are far away from each other

    :param result: The planes and their corresponding points from performing RANSAC
    :type pointcloud: RansacPointResult
    :param eps: The density parameter used to find neighbors
    :type eps: float
    :param min_points: Minimum points in a cluster
    :type min_points: int

    :return: The separated planes and their containing points
    :rtype: RansacPointResult
    """
    new_result = RansacPointResult([], [])
    
    for i in range(len(result.points)):
        pts = np.array(result.points[i])
        pcd = Pointcloud.np_to_o3d(pts)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        labelset = set(labels)
        if -1 in labelset:  # remove noise label
            labelset.remove(-1)

        for label in labelset:
            filtered_points = pts[labels == label, :]
            new_result.planes.append(result.planes[i])
            new_result.points.append(filtered_points)

    return new_result
