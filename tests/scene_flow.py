"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: view scene flow dataset after preprocess.
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from scripts.utils.mics import HDF5Data, flow_to_rgb
from scripts.utils.o3d_view import MyVisualizer

VIEW_FILE = f"{BASE_DIR}/assets/view/av2.json"

def check_flow(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    flow_mode: str = "flow", # "flow", "flow_est"
    start_id: int = -1,
    point_size: float = 3.0,
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if flow_mode == 'flow' else f'{flow_mode} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(0, len(dataset)))):
        # for easy stop and jump to any id, and save same id always from 0.
        if data_id < start_id and start_id != -1:
            continue
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        gm0 = data['gm0']
        pc0 = data['pc0'][~gm0]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([1.0, 0.0, 0.0]) # red: pc0

        pc1 = data['pc1']
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3][~data['gm1']])
        pcd1.paint_uniform_color([0.0, 1.0, 0.0]) # green: pc1

        pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + pose_flow) # if you want to check pose_flow
        pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + data[flow_mode][~gm0])
        pcd2.paint_uniform_color([0.0, 0.0, 1.0]) # blue: pc0 + flow
        o3d_vis.update([pcd, pcd1, pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

def vis(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    flow_mode: str = "flow", # "flow", "flow_est"
    start_id: int = -1,
    point_size: float = 2.0,
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if flow_mode == 'flow' else f'{flow_mode} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    # opt.background_color = np.asarray([216, 216, 216]) / 255.0
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    # opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(0, len(dataset)))):
        # for easy stop and jump to any id, and save same id always from 0.
        if data_id < start_id and start_id != -1:
            continue
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0

        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        flow = data[flow_mode] - pose_flow # ego motion compensation here.
        flow_color = flow_to_rgb(flow) / 255.0
        is_dynamic = np.linalg.norm(flow, axis=1) > 0.1
        flow_color[~is_dynamic] = [1, 1, 1]
        flow_color[gm0] = [1, 1, 1]

        black_color = np.zeros_like(flow_color)
        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc0[:, :3][~gm0])
        # pcd.colors = o3d.utility.Vector3dVector(flow_color[~gm0])
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(flow_color)
        o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(check_flow)
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")