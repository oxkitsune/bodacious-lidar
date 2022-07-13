from loader import Pointcloud
from chunk import chunk_pointcloud
from ransac import find_planes_chunked
from merge import merge_chunk_planes
from cluster import cluster_results
from visualize import generate_views, draw_planes_in_one

import argparse
import copy
from pathlib import Path

# Instantiate the parser
parser = argparse.ArgumentParser()

# Required positional argument
parser.add_argument('path', type=str,
                    help='The path to a pointcloud file')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--e57', action='store_true', help='If the file is an .e57 file')
group.add_argument('--ply', action='store_true', help='If the file is a .ply file')

args = parser.parse_args()

if args.ply:
    cloud = Pointcloud.from_ply(args.path)
else:
    cloud = Pointcloud.from_e57(args.path)
    cloud = cloud.denoise()
    cloud = cloud.downscale()
    cloud = cloud.save_to_ply(f"{Path(args.path).stem}.ply")

chunks = chunk_pointcloud(cloud)
planes = find_planes_chunked(cloud, chunks)
planes = merge_chunk_planes(planes)
planes = cluster_results(planes)


Path("./images").mkdir(parents=True, exist_ok=True)
generate_views(copy.deepcopy(planes))

draw_planes_in_one(planes.points)