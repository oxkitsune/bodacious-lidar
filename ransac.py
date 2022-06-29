import numpy as np
from dataclasses import dataclass
from loader import Pointcloud
from chunk import Point


Plane = list[int]

@dataclass
class RansacResult:
    # variables of plane formula Ax + By + Cz = D for each found plane
    planes: list[Plane]
    # all inlier point indices for each found plane
    points_idx: list[int]


@dataclass
class RansacPointResult:
    # variables of plane formula Ax + By + Cz = D for each found plane
    planes: list[Plane]
    # all inlier points for each found plane
    points: list[Point]


# def find_planes(Pointcloud, min_inliers=100, inlier_distance_threshold=0.1, ransac_n=3, num_iterations=1000, save=False, save_path=None):
def find_planes(pointcloud, min_inliers=100, inlier_distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """Finds all planes and their corresponding inlier points in a pointcloud, using the RANSAC algorithm

    :param pointcloud: The pointcloud to find planes in
    :type pointcloud: Pointcloud
    :param min_inliers: The minimum amount of points the planes should have
    :type min_inliers: int
    :param inlier_distance_threshold: The distance from a plane in which a point should qualify as an inlier
    :type inlier_distance_threshold: float
    :param ransac_n: The number of points to be initially considered inliers in each RANSAC iteration
    :type ransac_n: int
    :param num_iterations: The number of iterations a plane should be fitted 
    :type num_iterations: int

    :return: The planes and their containing points
    :rtype: RansacPointResult
    """
    result = RansacPointResult([], [])

    points = pointcloud.pcd
    np_points = Pointcloud.o3d_to_np(pointcloud.pcd)

    while True:
        # not enough points to perform RANSAC
        if len(np_points) < ransac_n:
            break

        # perform RANSAC
        best_eq, best_inliers = points.segment_plane(distance_threshold=inlier_distance_threshold,
            ransac_n=ransac_n, num_iterations=num_iterations)

        # exit early when we found the plane with less than the minimum amount of inliers
        if len(best_inliers) < min_inliers:
            break

        result.planes.append(best_eq)
        result.points.append(np_points[best_inliers])

        np_points = np.delete(np_points, best_inliers, axis=0)
        points = Pointcloud.np_to_o3d(np_points)

        if pointcloud.verbose:
            print(f"Found plane with {len(best_inliers)} inliers!")

    # if save:
    #     if not save_path:
    #         save_path = "./results"

    #     np.savez_compressed(save_path, planes=np.array(planes), inlier_points=np.array(inlier_points, dtype=object))

    return result


def find_planes_chunked(pointcloud, chunks, min_inliers=100, inlier_distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """Finds all planes and their corresponding inlier points in a chunked pointcloud, using the RANSAC algorithm

    :param pointcloud: The pointcloud to find planes in
    :type pointcloud: Pointcloud
    :param chunks: A 3D numpy array with at each index [x, y, z] that corresponds to a chunk coordinate, either None or a list of point indices inside that chunk
    :type chunks: np.ndarray
    :param min_inliers: The minimum amount of points the planes should have
    :type min_inliers: int
    :param inlier_distance_threshold: The distance from a plane in which a point should qualify as an inlier
    :type inlier_distance_threshold: float
    :param ransac_n: The number of points to be initially considered inliers in each RANSAC iteration
    :type ransac_n: int
    :param num_iterations: The number of iterations a plane should be fitted 
    :type num_iterations: int

    :return: A 3D numpy array with at each index [x, y, z] that corresponds to a chunk coordinate, either None or a :class:`RansacPointResult`
    :type chunks: The planes and their containing points
    :rtype: np.ndarray
    """
    pcd = Pointcloud.o3d_to_np(pointcloud.pcd)

    chunked_result = np.empty_like(chunks)

    for x, x_chunk in enumerate(chunks):
        for y, y_chunk in enumerate(x_chunk):
            for z, z_chunk in enumerate(y_chunk):
                if z_chunk is None:
                    continue

                pointcloud.pcd = Pointcloud.np_to_o3d(pcd[z_chunk])
                chunked_result[x, y, z] = find_planes(pointcloud, min_inliers, inlier_distance_threshold, ransac_n, num_iterations)

    pointcloud.pcd = Pointcloud.np_to_o3d(pcd)
    return chunked_result
