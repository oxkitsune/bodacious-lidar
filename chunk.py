import numpy as np
import numpy.typing as npt
from loader import Pointcloud

Point = np.ndarray

def chunk_pointcloud(pointcloud, chunk_size=2.5):
    """Split up pointcloud into cubic chunks

    :param pointcloud: The pointcloud to chunk
    :type pointcloud: Pointcloud
    :param chunk_size: The size of each chunk in meters
    :type chunk_size: float

    :return: A 3D numpy array with at each index [x, y, z] that corresponds to a chunk coordinate, either None or a list of point indices inside that chunk
    :rtype: np.ndarray
    """
    points = Pointcloud.o3d_to_np(pointcloud.pcd)

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # calculate chunk sizes
    chunk_sizes = np.ceil((max_coords - min_coords) / chunk_size).astype(int)
    chunks = np.empty(chunk_sizes, dtype=np.ndarray)

    for i, point in enumerate(points):
        x_chunk, y_chunk, z_chunk = (point / chunk_size).round().astype(int)

        if chunks[x_chunk, y_chunk, z_chunk] is None:
            chunks[x_chunk, y_chunk, z_chunk] = np.array([i])
        else:
            chunks[x_chunk, y_chunk, z_chunk] = np.append(chunks[x_chunk, y_chunk, z_chunk], i)

    if pointcloud.verbose:
        print("Finished chunking pointcloud!")

    return chunks
