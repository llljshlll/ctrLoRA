import os
import torch
import time

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict



def create_model(config_path):
    start = time.time()
    config = OmegaConf.load(config_path)
    config_load_time = time.time() - start
    #print(f"[create_model] Config 파일 로드 시간: {config_load_time:.4f} sec")
    
    start = time.time()
    model = instantiate_from_config(config.model).cpu()
    instantiate_time = time.time() - start
    #print(f"[create_model] instantiate_from_config 실행 시간: {instantiate_time:.4f} sec")
    
    print(f'Loaded model config from [{config_path}]')
    return model
