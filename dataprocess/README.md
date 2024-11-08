Dataset
---

README for downloading and preprocessing the dataset. We includes waymo, argoverse 2.0 and nuscenes dataset in our project.

- [Download](#download): includes how to download the dataset.
- [Process](#process): run script to preprocess the dataset.

We've updated the process dataset for:

- [x] Argoverse 2.0: check [here](#argoverse-20). The process script Involved from [DeFlow](https://github.com/KTH-RPL/DeFlow).
- [x] Waymo: check [here](#waymo-dataset). The process script was involved from [SeFlow](https://github.com/KTH-RPL/SeFlow).
- [ ] nuScenes: done coding, public after review. Will be involved later by another paper.

If you want to use all datasets above, there is a specific process environment in [envprocess.yml](../envprocess.yml) to install all the necessary packages. As Waymo package have different configuration and conflict with the main environment. Setup through the following command:

```bash
conda env create -f envprocess.yml
conda activate dataprocess
```

## Download

### Argoverse 2.0

Install their download tool:
```bash
mamba install s5cmd -c conda-forge
```

Download the dataset:
```bash
# train is really big (750): totally 966 GB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*" sensor/train

# val (150) and test (150): totally 168GB + 168GB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*" sensor/val
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/test/*" sensor/test

# for local and online eval mask from official repo
s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .
```

Then to quickly pre-process the data, we can [read these commands](#process) on how to generate the pre-processed data for training and evaluation. This will take around 0.5-2 hour for the whole dataset (train & val) based on how powerful your CPU is.

More [self-supervised data in AV2 LiDAR only](https://www.argoverse.org/av2.html#lidar-link), note: It **does not** include **imagery or 3D annotations**. The dataset is designed to support research into self-supervised learning in the lidar domain, as well as point cloud forecasting.
```bash
# train is really big (16000): totally 4 TB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/lidar/train/*" lidar/train

# val (2000): totally 0.5 TB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/lidar/val/*" lidar/val

# test (2000): totally 0.5 TB
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/lidar/test/*" lidar/test
``` 

#### Dataset frames

<!-- Note that some frames in LiDAR don't have any point cloud.... we didn't remove them in the total num. -->

| Dataset      | # Total Scene | # Total Frames |
| ------------ | ------------- | -------------- |
| Sensor/train | 750           | 110071         |
| Sensor/val   | 150           | 23547          |
| Sensor/test  | 150           | 23574          |
| LiDAR/train  | 16000         | -              |
| LiDAR/val    | 2000          | 597590         |
| LiDAR/test   | 2000          | 597575         |

### nuScenes

You need sign up an account at [nuScenes](https://www.nuscenes.org/) to download the dataset from [https://www.nuscenes.org/nuscenes#download](https://www.nuscenes.org/nuscenes#download) Full dataset (v1.0), you can choose to download lidar only. Click donwload mini split and unzip the file to the `nuscenes` folder if you want to test.

![](../assets/docs/nuscenes.png)


### Waymo Dataset

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details. 

For cluster without root user, check [here sdk tar gz](https://cloud.google.com/sdk/docs/downloads-versioned-archives).

Website to check their file: [https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow](https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow)

The thing we need is all things about `lidar`, to download the data, you can use the following command:

```bash
gsutil -m cp -r "gs://waymo_open_dataset_scene_flow/valid" .
gsutil -m cp -r "gs://waymo_open_dataset_scene_flow/train" .
```

And flowlabel data can be downloaded here with ground segmentation by HDMap follow the same style of [ZeroFlow](https://github.com/kylevedder/zeroflow/blob/master/data_prep_scripts/waymo/extract_flow_and_remove_ground.py).

You can download the processed map folder here to free yourself downloaded another type of data again:

```bash
wget https://zenodo.org/records/13744999/files/waymo_map.tar.gz
tar -xvf waymo_map.tar.gz -C /home/kin/data/waymo/flowlabel
# you will see there is a `map` folder in the `flowlabel` folder now.
```

#### Dataset frames

| Dataset | # Total Scene | # Total Frames |
| ------- | ------------- | -------------- |
| train   | 799           | 155687         |
| val     | 203           | 39381          |

## Process
This directory contains the scripts to preprocess the datasets. 

- `extract_av2.py`: Process the datasets in Argoverse 2.0.
- `extract_nus.py`: Process the datasets in nuScenes.
- `extract_waymo.py`: Process the datasets in Waymo.

Example Running command:`
```bash
# av2:
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir /home/kin/data/av2 --output_dir /home/kin/data/av2/preprocess

# waymo:
python dataprocess/extract_waymo.py --mode train --flow_data_dir /home/kin/data/waymo/flowlabel --map_dir /home/kin/data/waymo/flowlabel/map --output_dir /home/kin/data/waymo/preprocess  --nproc 48
```

All these preprocess scripts will generate the same format `.h5` file. The file contains the following in codes:

File: `[*:logid].h5` file named in logid. Every timestamp is the key of group (f[key]).

```python
def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None) :
    def create_group_data(group, pc, gm, pose, flow_0to1=None, flow_valid=None, flow_category=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('ground_mask', data=gm.astype(bool))
        group.create_dataset('pose', data=pose.astype(np.float32))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
```

After preprocessing, all data can use the same dataloader to load the data. As already in our DeFlow code.

Or you can run testing file to visualize the data. 

```bash
# view gt flow
python tools/visualization.py --data_dir /home/kin/data/av2/preprocess/sensor/mini --res_name flow

python tools/visualization.py --data_dir /home/kin/data/waymo/preprocess/val --res_name flow
```