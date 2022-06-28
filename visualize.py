import numpy as np
import open3d as o3d
from loader import Pointcloud
from tqdm.contrib import tzip


def draw_planes(inlier_points, max_amount=10):
    # Pass xyz to o3d pc and visualize
    for i, curr_inliers in enumerate(inlier_points):
        if i >= max_amount:
            break

        # color = np.random.choice(range(256), size=3)/255
        # colors = np.repeat([color], len(inliers[i]), axis=0)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = Pointcloud.np_to_o3d(curr_inliers)

        o3d.visualization.draw_geometries([pcd])


def draw_plane(inliers, normal):
    pcd = Pointcloud.np_to_o3d(inliers)
    cloud = Pointcloud(verbose=True)
    cloud.pcd = pcd
    (cloud
        .denoise(std_ratio=1.8)
        .filter_main_cluster(eps=0.3))

    pcd = cloud.pcd

    o3d.visualization.draw_geometries([pcd], lookat=normal, front=-normal, up=[normal[1], -normal[0], normal[2]], zoom=0.75)


def draw_planes_in_one(inlier_points):
    # Pass xyz to o3d pc and visualize
    pcds = []
    for i, curr_inliers in enumerate(inlier_points):
        pcd = Pointcloud.np_to_o3d(curr_inliers)

        color = np.random.choice(range(256), size=3)/255
        colors = np.repeat([color], len(curr_inliers), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)


def draw_planes_highlighted(inlier_points, index):
    # Pass xyz to o3d pc and visualize
    inlier_points.sort(key=len, reverse=True)

    pcds = []
    for i, curr_inliers in enumerate(inlier_points):
        pcd = Pointcloud.np_to_o3d(curr_inliers)

        if i != index:
            color = [0.9, 0.9, 0.9]
            colors = np.repeat([color], len(curr_inliers), axis=0)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)


def generate_views(results, max_images=50):
    # top 50 planes with the most points
    indices = np.argsort([len(pts) for pts in results.points])[::-1][:max_images]

    planes = np.array(results.planes)[indices]
    points = np.array(results.points, dtype=np.object)[indices]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for i, (curr_plane, curr_inliers) in enumerate(tzip(planes, points)):
        # print(f"{i}: Loading plane {curr_plane=} with {len(curr_inliers)} points")
        normal = curr_plane[:3]
        center = points[i].mean(axis=0)

        # center all points
        curr_inliers -= center
        pcd = Pointcloud.np_to_o3d(curr_inliers)

        # creates line to display normal
        pointy = np.array([points[i].mean(axis=0), normal + points[i].mean(axis=0)])
        lines = np.array([[0, 1]])
        colors = np.array([[1, 0, 0]])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pointy)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(pcd)
        vis.add_geometry(line_set)

        view = vis.get_view_control()
        view.change_field_of_view(step=-55.0) # Standard is 60, 5 is minimum
        view.set_front(normal)
        view.set_up([normal[0], normal[2], normal[1]])
        view.set_lookat(-normal)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(f"images/img_{i}.png")
        vis.clear_geometries()
    vis.destroy_window()
    