'''
# Usage for old model file to new version code
'''
import torch
import fire, time

def main(
    # download from: https://zenodo.org/records/12632962
    model_path: str = "/home/kin/model_zoo/v2/seflow_best.ckpt", 
    # new output weight file
    output_path: str = "/home/kin/model_zoo/v3/seflow_best.ckpt",
):
    model = torch.load(model_path)
    model_name = model['hyper_parameters']['cfg']['model']['name']
    old_path = model['hyper_parameters']['cfg']['model']['target']['_target_']
    new_path = old_path.replace(f"scripts.network.models.{model_name}", "src.models")
    model['hyper_parameters']['cfg']['model']['target']['_target_'] = new_path
    torch.save(model, output_path)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")