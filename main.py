from loader import Pointcloud
from chunk import chunk_pointcloud
from ransac import find_planes_chunked
from merge import merge_chunk_planes
from cluster import cluster_results
from visualize import generate_views

cloud = Pointcloud.from_ply("data/0104Amstel.ply")
chunks = chunk_pointcloud(cloud)
planes = find_planes_chunked(cloud, chunks)
planes = merge_chunk_planes(planes)
planes = cluster_results(planes)

generate_views(planes)