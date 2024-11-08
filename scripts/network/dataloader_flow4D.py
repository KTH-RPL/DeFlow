"""
# Created: 2023-11-04 15:52
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Torch dataloader for the dataset we preprocessed.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py, os, pickle, argparse, sys
from tqdm import tqdm
import yaml
import numpy as np
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

def collate_fn_pad(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground= [], []
    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['pc0'][~batch[i]['gm0']])
        pc1_after_mask_ground.append(batch[i]['pc1'][~batch[i]['gm1']])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict


def map_label(label, mapdict):
# put label from original values to xentropy
# or vice-versa, depending on dictionary values
# make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]


class HDF5Dataset(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('./conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if not os.path.exists(eval_index_file):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

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
            return len(self.eval_data_index)-1
        return len(self.data_index)-1
    
    def __getitem__(self, index_):
        if self.eval_index:
            if(index_ == 156):
                print(1)
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index

            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                index_ = index_ - 1
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            pc0 = torch.tensor(f[key]['lidar'][:])
            gm0 = torch.tensor(f[key]['ground_mask'][:])
            pose0 = torch.tensor(f[key]['pose'][:])

            if self.scene_id_bounds[scene_id]["max_index"] == index_:
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            #next_timestamp = str(self.data_index[index_+1][1])
            #print("index:", index_)
            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

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