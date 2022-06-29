import numpy as np
from ransac import RansacPointResult

def merge_chunk_planes(chunked_result, thresh=0.9):
    """Merges chunked RANSAC results together into a single :class:`RansacPointResult` using the cosine similarity of the planes

    :param chunked_result: A 3D numpy array with at each index [x, y, z] that corresponds to a chunk coordinate, either None or a :class:`RansacPointResult`
    :type chunked_result: np.ndarray
    :param thresh: The threshold similarity for the planes to be merged together
    :type thresh: float

    :return: All planes and their corresponding points, where similar planes have been merged together
    :rtype: RansacPointResult
    """
    max_x, max_y, max_z = chunked_result.shape
    result = None

    for x in range(max_x):
        for y in range(max_y):
            for z in range(max_z):
                curr = chunked_result[x, y, z]

                # Current chunk contains no planes
                if curr is None:
                    continue

                if result is None:
                    result = curr
                    continue
                
                to_add = set(range(len(curr.planes)))
                while to_add:
                    # try to merge plane with an existing one
                    j = to_add.pop()
                    found_similar = False
                    for i in range(len(result.planes)):
                        # similar, merge the planes
                        if _similar(result.planes[i], curr.planes[j], thresh):
                            found_similar = True

                            result.planes[i] = ((result.planes[i] * len(result.points[i]) + curr.planes[j] * len(curr.points[j]))
                                / (len(result.points[i]) + len(curr.points[j])))  # weighted average
                            result.points[i] = np.append(result.points[i], curr.points[j], axis=0)
                            break

                    # dissimilar, append the plane seperately
                    if not found_similar:
                        result.planes.append(curr.planes[j])
                        result.points.append(curr.points[j])

    return result


def _similar(plane1, plane2, thresh=0.9):
    """Find out if planes are similar

    :rtype: bool
    """
    curr_normal = plane1[:3]
    nb_normal = plane2[:3]

    # find cosine similarity between the planes
    sim = np.dot(curr_normal, nb_normal) / (np.linalg.norm(curr_normal) * np.linalg.norm(nb_normal))

    # sometimes planes will not merge
    # might be because ransac fit normals are facing
    # opposite directions
    sim = np.abs(sim) #TODO: find out if this truly fixes it
    # print(sim)

    return sim >= thresh
