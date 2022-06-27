import numpy as np
import pye57
import open3d as o3d

from tqdm import tqdm 
from tqdm.contrib import tzip

from dataclasses import dataclass

@dataclass
class RansacResult:
    planes: list[list[int]]  # list[list[a, b, c, d]]
    points: list[np.ndarray]  # list[[x, y, z]]


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

    def chunks(self, chunk_size=2.5):
        pcd = Pointcloud.o3d_to_np(self.pcd)
        min_coords = pcd.min(axis=0)
        max_coords = pcd.max(axis=0)

        # calculate chunk sizes
        chunk_sizes = np.ceil((max_coords - min_coords) / chunk_size).astype(int)
        chunks = np.empty(chunk_sizes, dtype=np.ndarray)

        for i, point in enumerate(pcd):
            x_chunk, y_chunk, z_chunk = (point / chunk_size).round().astype(int)

            if chunks[x_chunk, y_chunk, z_chunk] is None:
                chunks[x_chunk, y_chunk, z_chunk] = np.array([i])
            else:
                chunks[x_chunk, y_chunk, z_chunk] = np.append(chunks[x_chunk, y_chunk, z_chunk], i)

        if self.verbose:
            print("Finished chunking pointcloud!")

        return chunks

    def find_planes_chunked(self, chunks, min_inliers=100, inlier_distance_threshold=0.1, ransac_n=3, num_iterations=1000, save=False, save_path=None):
        pcd = Pointcloud.o3d_to_np(self.pcd)
        chunked_result = np.empty_like(chunks)

        for x, x_chunk in enumerate(chunks):
            for y, y_chunk in enumerate(x_chunk):
                for z, z_chunk in enumerate(y_chunk):
                    if z_chunk is None:
                        continue

                    self.pcd = Pointcloud.np_to_o3d(pcd[z_chunk])
                    new_planes, new_inlier_points = self.find_planes(min_inliers, inlier_distance_threshold, ransac_n, num_iterations, save, save_path)
                    
                    chunked_result[x, y, z] = RansacResult(new_planes, new_inlier_points)

        self.pcd = Pointcloud.np_to_o3d(pcd)
        return chunked_result

    def merge_chunk_planes(self, chunked_result, thresh=0.9):
        max_x, max_y, max_z = chunked_result.shape

        result = None

        for x in tqdm(range(max_x)):
            for y in range(max_y):
                for z in range(max_z):
                    curr = chunked_result[x, y, z]

                    if curr is None:
                        # print("Curr is None")
                        continue

                    if result is None:
                        # print("Result is None")
                        result = curr
                        continue
                    
                    to_add = set(range(len(curr.planes)))
                    while to_add:
                        j = to_add.pop()
                        found_similar = False
                        for i in range(len(result.planes)):
                            # similar, merge the planes
                            if Pointcloud._similar(result.planes[i], curr.planes[j], thresh):
                                found_similar = True
                                result.planes[i] = ((result.planes[i] * len(result.points[i]) + curr.planes[j] * len(curr.points[j]))
                                    / (len(result.points[i]) + len(curr.points[j])))  # weighted average
                                result.points[i] = np.append(result.points[i], curr.points[j], axis=0)

                        # dissimilar, append the plane
                        if not found_similar:
                            result.planes.append(curr.planes[j])
                            result.points.append(curr.points[j])

        return result

    # def merge_chunk_planes(self, chunked_result, thresh=0.9):
    #     max_x, max_y, max_z = chunked_result.shape

    #     curr_x = np.random.randint(max_x)
    #     curr_y = np.random.randint(max_y)
    #     curr_z = np.random.randint(max_z)
    #     while type(chunked_result[curr_x, curr_y, curr_z]) == int:
    #         curr_x = np.random.randint(max_x)
    #         curr_y = np.random.randint(max_y)
    #         curr_z = np.random.randint(max_z)

    #     finished = np.zeros_like(chunked_result).astype(bool)

    #     res_planes, res_inliers_pts = chunked_result[curr_x, curr_y, curr_z]

    #     planes, inlier_points = self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x, curr_y, curr_z, finished, thresh, 0)
    #     return planes, inlier_points


    # def _merge_chunk_planes(self, chunked_result, res_planes, res_inliers_pts, curr_x, curr_y, curr_z, finished, thresh, depth):
    #     # print(f"{depth=}, len res_planes={len(res_planes)}, len new_chunk_len={len(chunked_result[curr_x, curr_y, curr_z])}")
    #     print((finished == True).sum())

    #     if (curr_x < 0 or curr_x >= chunked_result.shape[0]
    #        or curr_y < 0 or curr_y >= chunked_result.shape[1]
    #        or curr_z < 0 or curr_z >= chunked_result.shape[2]):
    #         print("out of bounds, returning")
    #         return

    #     if finished[curr_x, curr_y, curr_z]:
    #         print(f"already finished {curr_x} {curr_y} {curr_z}")
    #         return

    #     if type(chunked_result[curr_x, curr_y, curr_z]) != int:  #TODO: cringe
    #         to_add = set()
    #         for plane, inlier_pts in zip(res_planes, res_inliers_pts):
    #             # print("outer loop")
    #             for i, (curr_plane, curr_inlier_pts) in enumerate(zip(*chunked_result[curr_x, curr_y, curr_z])):
    #                 # print("inner loop")

    #                 if Pointcloud._similar(plane, curr_plane, thresh):
    #                     # create plane that's a weighted average of the two 
    #                     plane = [(plane * len(inlier_pts) + curr_plane * len(curr_inlier_pts)
    #                         / (len(inlier_pts) + len(curr_inlier_pts)))]
    #                     inlier_pts = [np.append(inlier_pts, curr_inlier_pts, axis=0).tolist()]
    #                     # print("similar")
    #                     # print(inlier_pts)
    #                 else:
    #                     # print("not similar")
    #                     # print(i, len(curr_inlier_pts))
    #                     to_add.add(i)

    #         if len(to_add) != 0:
    #             print(f"{to_add=}")
    #             res_planes += [chunked_result[curr_x, curr_y, curr_z][0][i] for i in to_add]
    #             res_inliers_pts += [chunked_result[curr_x, curr_y, curr_z][1][i] for i in to_add]

    #     finished[curr_x, curr_y, curr_z] = True

    #     self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x + 1, curr_y, curr_z, finished, thresh, depth+1)
    #     self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x - 1, curr_y, curr_z, finished, thresh, depth+1)
    #     self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x, curr_y + 1, curr_z, finished, thresh, depth+1)
    #     self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x, curr_y - 1, curr_z, finished, thresh, depth+1)
    #     self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x, curr_y, curr_z + 1, finished, thresh, depth+1)
    #     self._merge_chunk_planes(chunked_result, res_planes, res_inliers_pts, curr_x, curr_y, curr_z - 1, finished, thresh, depth+1)

    #     print("returning finalresults")
    #     return res_planes, res_inliers_pts

    def _similar(plane1, plane2, thresh=0.9):
        curr_normal = plane1[:3]
        nb_normal = plane2[:3]

        # find cosine similarity between the planes
        sim = np.dot(curr_normal, nb_normal) / (np.linalg.norm(curr_normal) * np.linalg.norm(nb_normal))

        # sometimes planes will not merge
        # might be because ransac fit normals are facing
        # opposite directions
        sim = np.abs(sim) #TODO: find out if this truly fixes it
        # print(sim)

        if np.isnan(sim):
            print(curr_normal, nb_normal)

        return sim >= thresh

    def load_from_e57(self, path, *, compress=lambda x: x.astype('f')):
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
            if len(np_points) < ransac_n:
                break

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
