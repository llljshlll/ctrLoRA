#!/usr/bin/env python
# test_decode_first_stage_override.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
# from api import create_model  # 사용 안함, 직접 로드 방식으로 대체

if __name__ == "__main__":
    # 1) config + ckpt 경로 지정
    config_path = "configs/inference/ctrlora_sd15_rank128_1lora.yaml"
    ckpt_path = "ckpts/sd15/v1-5-pruned-noVAE.ckpt"

    # 2) config 로드 후 모델 생성
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu().eval()

    # 3) ckpt 로드
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[load_state_dict] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    encoder = model.first_stage_model

    # 4) 테스트 이미지 준비
    img = Image.open("images/_aTpHvBraCMhi_wxJ3kZSw.png").convert("RGB").resize((256,256))
    img_np = np.array(img)
    tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        # 5) Stable Diffusion용: 입력 [-1, 1] 정규화
        tensor_norm = tensor * 2 - 1 

        # 6) encode: 분포 → 샘플링
        z_dist = encoder.encode(tensor_norm)
        z_raw = z_dist.sample()  # latent tensor

        # 7) decode
        recon_manual = encoder.decode(z_raw)
        recon_default = model.decode_first_stage(z_raw)

        # 8) [-1,1] → [0,1] for display
        recon_manual = (recon_manual + 1) / 2
        recon_default = (recon_default + 1) / 2

    # 9) 시각화
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img_np); plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Manual decode")
    plt.imshow(recon_manual.squeeze().permute(1,2,0).cpu().numpy())
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Default decode_first_stage")
    plt.imshow(recon_default.squeeze().permute(1,2,0).cpu().numpy())
    plt.axis("off")

    plt.tight_layout()
    plt.show()
