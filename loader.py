import numpy as np
import pye57
import open3d as o3d


class Pointcloud:
    def __init__(self, verbose=False):
        self.path = None
        self.pcd = None
        self.verbose = verbose

    def o3d_to_np(o3d_pcd):
        return np.asarray(o3d_pcd.points)

    def np_to_o3d(np_pcd):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
        return pcd

    def load_from_e57(self, path):
        if path.endswith(".e57"):
            path = path[:-4]

        e57 = pye57.E57(f"{path}.e57")

        # read scan at index 0
        data = e57.read_scan(0, ignore_missing_fields=True)

        # creates (n, 3) array of points
        points = np.array([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).T

        # converts it to a o3d pointcloud
        self.pcd = Pointcloud.np_to_o3d(points)

        if self.verbose:
            print(f"Loaded pointcloud from {path}.e57.")
        return self

    def load_from_ply(self, path):
        if path.endswith(".ply"):
            path = path[:-4]
        self.pcd = o3d.io.read_point_cloud(f"{path}.ply")

        if self.verbose:
            print(f"Loaded pointcloud from {path}.ply.")  
        return self

    def save_to_ply(self, path):
        if path.endswith(".ply"):
            path = path[:-4]

        if not path:
            path = self.path

        o3d.io.write_point_cloud(f"{path}.ply", self.pcd)
        if self.verbose:
            print(f"Saved file to {path}.ply.")
        return self

    def downscale(self, resolution=0.1):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=resolution)
        if self.verbose:
            print(f"Downscaled pointcloud to {resolution}m resolution.")
        return self

    def denoise(self, num_neighbors=20, std_ratio=2):
        pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=num_neighbors,
                                                      std_ratio=std_ratio)
        self.pcd = pcd
        if self.verbose:
            print("Finished pointcloud denoising.")
        return self

    def filter_main_cluster(self, eps=0.5, min_points=100):
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        pcd = Pointcloud.o3d_to_np(self.pcd)
        pcd = pcd[labels == 0]
        self.pcd = Pointcloud.np_to_o3d(pcd)
        if self.verbose:
            print(f"Finished filtering out pointcloud noise using clustering. Found {labels.max()} clusters.")
        return self

    def find_planes(self, min_inliers=100, inlier_distance_threshold=0.1, ransac_n=3, num_iterations=1000, save=False, save_path=None):
        planes = []  # variables of plane formula Ax + By + Cz = D for each found plane
        inlier_points = []  # all inlier points for each found plane

        points = self.pcd
        np_points = Pointcloud.o3d_to_np(self.pcd)

        while True:
            # RANSAC
            best_eq, best_inliers = points.segment_plane(distance_threshold=inlier_distance_threshold,
                ransac_n=ransac_n, num_iterations=num_iterations)
            if len(best_inliers) < min_inliers:
                break

            planes.append(best_eq)
            inlier_points.append(np_points[best_inliers])

            np_points = np.delete(np_points, best_inliers, axis=0)
            points = Pointcloud.np_to_o3d(np_points)

            if self.verbose:
                print(f"Found plane with {len(best_inliers)} inliers!")

        if save:
            if not save_path:
                save_path = "./results"

            np.savez_compressed(save_path, planes=np.array(planes), inlier_points=np.array(inlier_points, dtype=object))    

        return planes, inlier_points
