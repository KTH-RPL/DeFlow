"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: view scene flow dataset after preprocess.
"""

from pathlib import Path
import numpy as np
import fire, time
from av2.geometry.se3 import SE3

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from scripts.utils.mics import HDF5Data, flow_to_rgb

def vis(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor",
    flow_mode: str = "flow", # "flow", "flow_est"
):
    import open3d as o3d
    from scripts.utils.o3d_view import ViewControl, color_map

    dataset = HDF5Data(data_dir, flow_view=True)
    
    viz = o3d.visualization.VisualizerWithKeyCallback()
    viz.create_window(window_name=f"view {'ground truth flow' if flow_mode == 'flow' else 'result flow'}, press n for next frame")
    opt = viz.get_render_option()
    # opt.background_color = np.asarray([216, 216, 216]) / 255.0
    opt.background_color = np.asarray([76/255, 86/255, 106/255])
    opt.point_size = 2.0
    pcd = o3d.geometry.PointCloud()
    viz.add_geometry(pcd)
    ctr = viz.get_view_control()
    o3d_vctrl = ViewControl(ctr, f"{BASE_DIR}/assets/view/av2.json")

    class thread:
        def __init__(self):
            self.id = 3
            # self.id = dataset.get_id_from_scene("08734a1b-0289-3aa3-a6ba-8c7121521e26", "315971469760093000")
            
            
            self.pre_cluster_centers, self.pre_labels = [], []
        def next_frame(self, viz):
            # if self.id > 100:
            #     return
            data = dataset[self.id]
            now_scene_id = data['scene_id']
            pc0 = data['pc0']
            gm0 = data['gm0']
            pose0 = data['pose0']
            pose1 = data['pose1']
            ego_pose = np.linalg.inv(pose1) @ pose0
            pose_se3 = SE3(rotation=ego_pose[:3,:3], translation=ego_pose[:3,3])
            
            pose_flow = pose_se3.transform_point_cloud(pc0[:, :3]) - pc0[:, :3]
            flow = data[flow_mode]
            flow_color = flow_to_rgb(flow) / 255.0
            is_dynamic = np.linalg.norm(flow - pose_flow, axis=1) > 0.05
            flow_color[~is_dynamic] = [1, 1, 1]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3][~gm0])
            pcd.colors = o3d.utility.Vector3dVector(flow_color[~gm0])
            
            
            viz.clear_geometries()
            viz.add_geometry(pcd)
            # need set view again, check this issue: https://github.com/isl-org/Open3D/issues/2219
            o3d_vctrl.read_viewTfile(f"{BASE_DIR}/assets/view/av2.json")
            viz.update_geometry(pcd)
            viz.update_renderer()
            print(f"id: {self.id}, next frame..")
            self.id = self.id+1

    myThread = thread()
    myThread.next_frame(viz)
    # click 'N' to next frame
    # viz.register_key_callback(ord('N'), myThread.next_frame)
    viz.register_animation_callback(myThread.next_frame)
    viz.run()
    viz.destroy_window()

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")