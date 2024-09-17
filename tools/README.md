Tools
---

Here we introduce some tools to help you:
- visualize the data and results.
- convert the pretrained model from others.
- ... More to come.

## Visualization

run `tools/visualization.py` to view the scene flow dataset with ground truth flow. Note the color wheel in under world coordinate.

```bash
# view gt flow
python3 tools/visualization.py --data_dir /home/kin/data/av2/preprocess/sensor/mini --res_name flow

# view est flow
python3 tools/visualization.py --data_dir /home/kin/data/av2/preprocess/sensor/mini --res_name deflow_best
python3 tools/visualization.py --data_dir /home/kin/data/av2/preprocess/sensor/mini --res_name seflow_best
```

Demo Effect (press `SPACE` to stop and start in the visualization window):

https://github.com/user-attachments/assets/f031d1a2-2d2f-4947-a01f-834ed1c146e6

## Conversion

run `tools/zero2ours.py` to convert the ZeroFlow pretrained model to our codebase. 

```bash
python tools/zero2ours.py --model_path /home/kin/nsfp_distilatation_3x_49_epochs.ckpt --reference_path /home/kin/fastflow3d.ckpt --output_path /home/kin/zeroflow3x.ckpt
```

- model_path, you can download from: [kylevedder/zeroflow_weights](https://github.com/kylevedder/zeroflow_weights/tree/master/argo)
- reference_path,  you can download fastflow3d model from: [zendo](https://zenodo.org/records/12632962)
- output_path, the converted model path. You can then run any evaluation script and visualization script with the converted model.