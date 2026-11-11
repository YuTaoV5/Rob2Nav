"""Export the combined model and write its rl_deploy motor contract."""
import json
from pathlib import Path
import sys

import onnx
import torch
import yaml

from r177_onnx_model import OBS_SIZE, STATE_SIZE, R177Step
from search_lateral_omni_teacher import course
from export_r152_onnx import write_deploy_yaml


def main():
    assets,weights,model_path,config_path=map(Path,sys.argv[1:5])
    torch.set_num_threads(1)
    contract,_,geometry=course(assets)
    graph=R177Step(weights,contract,geometry).eval()
    with torch.inference_mode():
        torch.onnx.export(
            graph,(torch.zeros(1,OBS_SIZE),torch.zeros(1,STATE_SIZE)),str(model_path),
            input_names=["obs","state"],output_names=["actions","next_state"],
            opset_version=17,dynamo=False,
        )
    model=onnx.load(model_path)
    model.ir_version=8
    onnx.checker.check_model(model)
    onnx.save(model,model_path)
    write_deploy_yaml(assets,config_path,model_path)
    config=yaml.safe_load(config_path.read_text())
    config["model_version"]="r177-flat24576-stairs81920"
    config["state_size"]=STATE_SIZE
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    print(json.dumps({"observation_size":OBS_SIZE,"state_size":STATE_SIZE,"actions":16}),flush=True)


if __name__=="__main__":
    main()
