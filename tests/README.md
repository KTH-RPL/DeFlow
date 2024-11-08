Testing
---

Mainly for testing if the code works as expected. Like check the dataloader, ground truth flow and also visualize our predicted flow.

run `tests/scene_flow.py` to view the scene flow dataset with ground truth flow. Note the color wheel in under world coordinate.

```bash
# view gt flow
python3 tests/scene_flow.py --data_dir /home/kin/data/av2/preprocess/sensor/mini --flow_mode flow

# view est flow
python3 tests/scene_flow.py --data_dir /home/kin/data/av2/preprocess/sensor/mini --flow_mode flow_est
```

![](../assets/docs/vis_res.png)