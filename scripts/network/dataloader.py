"""
# Created: 2023-11-04 15:52
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Torch dataloader for the dataset we preprocessed.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py, os, pickle, argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path

def collate_fn_pad(batch):
    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground= [], []
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['pc0'][~batch[i]['gm0']])
        pc1_after_mask_ground.append(batch[i]['pc1'][~batch[i]['gm1']])
    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)

    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))]
    }

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from scripts.utils import pcdpy3
from linefit import ground_seg
def xyzqwxyz_to_matrix(xyzqwxyz: list):
    """
    input: xyzqwxyz: [x, y, z, qx, qy, qz, qw] a list of 7 elements
    """
    rotation = R.from_quat([xyzqwxyz[4], xyzqwxyz[5], xyzqwxyz[6], xyzqwxyz[3]]).as_matrix()
    pose = np.eye(4).astype(np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = xyzqwxyz[:3]
    return pose

def inv_pose_matrix(pose):
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])
    return inv_pose

class DynamicMapData(Dataset):
    def __init__(self, directory):
        super(DynamicMapData, self).__init__()
        self.scene_id = directory.split("/")[-1]
        self.directory = Path(directory) / "pcd"
        self.pcd_files = [os.path.join(self.directory, f) for f in sorted(os.listdir(self.directory)) if f.endswith('.pcd')]

        # FIXME: ground segmentation config: av2, kitti, semindoor; hard code here
        if self.scene_id == "av2":
            ground_config = f"{BASE_DIR}/conf/groundseg/av2.toml"
        elif self.scene_id == "semindoor":
            ground_config = f"{BASE_DIR}/conf/groundseg/semindoor.toml"
        else:
            ground_config = f"{BASE_DIR}/conf/groundseg/kitti.toml"

        self.groundseg = ground_seg(ground_config)

    def __len__(self):
        return len(self.pcd_files)
    
    def __getitem__(self, index_):
        res_dict = {
            'scene_id': self.scene_id,
            'timestamp': self.pcd_files[index_].split("/")[-1].split(".")[0],
        }
        pcd_ = pcdpy3.PointCloud.from_path(self.pcd_files[index_])
        pc0 = pcd_.np_data[:,:3]
        pose0 = xyzqwxyz_to_matrix(list(pcd_.viewpoint))

        
        if index_ + 1 == len(self.pcd_files):
            index_ = index_ - 2
        pcd_ = pcdpy3.PointCloud.from_path(self.pcd_files[index_+1])
        pc1 = pcd_.np_data[:,:3]
        pose1 = xyzqwxyz_to_matrix(list(pcd_.viewpoint))

        inv_pose0 = inv_pose_matrix(pose0)
        ego_pc0 = pc0 @ inv_pose0[:3, :3].T + inv_pose0[:3, 3]
        # pcdpy3.save_pcd(f"{BASE_DIR}/ego_pc0.pcd", ego_pc0)
        # sys.exit(0)
        gm0 = np.array(self.groundseg.run(ego_pc0[:,:3]))

        inv_pose1 = inv_pose_matrix(pose1)
        ego_pc1 = pc1 @ inv_pose1[:3, :3].T + inv_pose1[:3, 3]
        gm1 = np.array(self.groundseg.run(ego_pc1[:,:3]))

        res_dict['pc0'] = torch.tensor(ego_pc0.astype(np.float32))
        res_dict['gm0'] = torch.tensor(gm0.astype(np.bool_))
        res_dict['pose0'] = torch.tensor(pose0)
        res_dict['pc1'] = torch.tensor(ego_pc1.astype(np.float32))
        res_dict['gm1'] = torch.tensor(gm1.astype(np.bool_))
        res_dict['pose1'] = torch.tensor(pose1)
        res_dict['world_pc0'] = torch.tensor(pc0.astype(np.float32))
        return res_dict

class HDF5Dataset(Dataset):
    def __init__(self, directory, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval
            with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                self.eval_data_index = pickle.load(f)
                
        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp])
        else:
            scene_id, timestamp = self.data_index[index_]
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_:
                index_ = index_ - 1
            # get the data again
            scene_id, timestamp = self.data_index[index_]

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            pc0 = torch.tensor(f[key]['lidar'][:])
            gm0 = torch.tensor(f[key]['ground_mask'][:])
            pose0 = torch.tensor(f[key]['pose'][:])

            next_timestamp = str(self.data_index[index_+1][1])
            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:])
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])
            # if pc0[~gm0].shape[0] == 0:
            #     print(f"\nWarning: empty point cloud! scene_id: [{scene_id}, {timestamp}], Check the data!\n")
            #     return self.__getitem__(index_+1)
            # elif pc1[~gm1].shape[0] == 0:
            #     print(f"\nWarning: empty point cloud! scene_id: [{scene_id}, {next_timestamp}], Check the data!\n")
            #     return self.__getitem__(index_-1)
            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0,
                'gm0': gm0,
                'pose0': pose0,
                'pc1': pc1,
                'gm1': gm1,
                'pose1': pose1,
            }

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:])
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:])
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

            if self.eval_index:
                eval_mask = torch.tensor(f[key]['eval_mask'][:])
                res_dict['eval_mask'] = eval_mask

        return res_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader test")
    parser.add_argument('--data_mode', '-m', type=str, default='val', metavar='N', help='Dataset mode.')
    parser.add_argument('--data_path', '-d', type=str, default='/home/kin/data/av2/preprocess/sensor', metavar='N', help='preprocess data path.')
    options = parser.parse_args()

    # testing eval mode
    dataset = HDF5Dataset(options.data_path+"/"+options.data_mode, eval=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn_pad)
    for data in tqdm(dataloader, ncols=80, desc="eval mode"):
        res_dict = data