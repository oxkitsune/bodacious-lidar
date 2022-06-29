import pye57
import numpy as np
import open3d as o3d

class Pointcloud:
    """An intermediate representation for loading and editing pointclouds

    :param verbose: Whether the pointcloud should log messages to stdout
    :type verbose: bool
    """
    def __init__(self, verbose=False):
        self.pcd = None
        self.verbose = verbose

    @classmethod
    def from_e57(cls, path, verbose=False, *, compress=lambda x: x.astype('f')):
        """Loads a pointcloud from a `.e57` file

        :param path: The path to the file
        :type path: string
        :param verbose: Whether the pointcloud should log messages to stdout
        :type verbose: bool

        :rtype: self
        """
        cloud = Pointcloud(verbose=verbose)

        if path.endswith(".e57"):
            path = path[:-4]

        e57 = pye57.E57(f"{path}.e57")

        # read scan at index 0
        header = e57.get_header(0)

        # read and compress each dimension seperately
        def read_field(field):
            buffers = pye57.libe57.VectorSourceDestBuffer()
            data, buffer = e57.make_buffer(field, header.point_count)

            buffers.append(buffer)
            header.points.reader(buffers).read()

            return compress(data)

        # creates (n, 3) array of points
        points = np.array([
            read_field("cartesianX"),
            read_field("cartesianY"),
            read_field("cartesianZ"),
        ]).T

        # converts it to a o3d pointcloud
        cloud.pcd = Pointcloud.np_to_o3d(points)

        if verbose:
            print(f"Loaded pointcloud from {path}.e57.")
        return cloud

    @classmethod
    def from_ply(cls, path, verbose=False):
        """Loads a pointcloud from a `.ply` file

        :param path: The path to the file
        :type path: string
        :param verbose: Whether the pointcloud should log messages to stdout
        :type verbose: bool

        :rtype: self
        """
        cloud = Pointcloud(verbose=verbose)

        if path.endswith(".ply"):
            path = path[:-4]

        cloud.pcd = o3d.io.read_point_cloud(f"{path}.ply")

        if verbose:
            print(f"Loaded pointcloud from {path}.ply.")  
        return cloud

    @staticmethod
    def o3d_to_np(o3d_pcd):
        """Converts an o3d pointcloud to a corresponding numpy representation

        :param o3d_pcd: The path to the file
        :type o3d_pcd: open3d.geometry.PointCloud

        :return: The points stored as an (n, 3) numpy array.
        :rtype: np.ndarray
        """
        return np.asarray(o3d_pcd.points)

    @staticmethod
    def np_to_o3d(np_pcd):
        """Converts a numpy array pointcloud to a corresponding o3d representation

        :param np_pcd: The points stored as an (n, 3) numpy array
        :type np_pcd: np.ndarray

        :return: The points stored as an open3d pointcloud
        :rtype: open3d.geometry.PointCloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
        return pcd

    def save_to_ply(self, path):
        """Converts a numpy array pointcloud to a corresponding o3d representation

        :param path: The path to the save location for the file
        :type path: string

        :rtype: self
        """
        if path.endswith(".ply"):
            path = path[:-4]

        o3d.io.write_point_cloud(f"{path}.ply", self.pcd)

        if self.verbose:
            print(f"Saved file to {path}.ply.")
        return self

    def downscale(self, resolution=0.1):
        """Scales down the pointcloud

        :param resolution: The resolution to which the cloud should be downscaled
        :type resolution: float

        :rtype: self
        """
        self.pcd = self.pcd.voxel_down_sample(voxel_size=resolution)

        if self.verbose:
            print(f"Downscaled pointcloud to {resolution}m resolution.")
        return self

    def denoise(self, num_neighbors=20, std_ratio=2):
        """Denoises the pointcloud using a statistical outlier removal approach

        :param num_neighbors: The resolution to which the cloud should be downscaled
        :type num_neighbors: int

        :param std_ratio: Sets the threshold level based on the standard deviation of the
         average distances across the point cloud.The lower this number the more aggressive the filter will be
        :type std_ratio: float

        :rtype: self
        """
        pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=num_neighbors,
                                                      std_ratio=std_ratio)
        self.pcd = pcd

        if self.verbose:
            print("Finished pointcloud denoising.")
        return self

    def filter_main_cluster(self, eps=1, min_points=100):
        """Cluster the pointcloud using the DBSCAN algorithm to filter out scan blobs that are far removed from the house

        :param eps: The density parameter used to find neighbors
        :type eps: float
        :param min_points: Minimum points in a cluster
        :type min_points: int

        :rtype: self
        """
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        pcd = Pointcloud.o3d_to_np(self.pcd)
        pcd = pcd[labels == 0]
        self.pcd = Pointcloud.np_to_o3d(pcd)

        if self.verbose:
            print(f"Finished filtering out pointcloud noise using clustering. Found {labels.max()} clusters.")
        return self
