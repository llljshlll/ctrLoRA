#!/usr/bin/env python
# test_decode_first_stage.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 아래 import 경로는 프로젝트 구조에 맞게 조정하세요.
# create_model 함수는 config.model.target 에 지정된 ControlInferenceLDM 인스턴스를 반환합니다.
from api import create_model  

if __name__ == "__main__":
    # 1) 파이프라인 모델 로드 (CPU 모드)
    config_path = "configs/inference/ctrlora_sd15_rank128_1lora.yaml"   # 실제 사용하는 config 파일 경로
    model = create_model(config_path).cpu().eval()

    # (필요 시) TAESD 가중치 로드가 이미 TAESD.__init__에서 됐는지 확인하시고,
    # 아니라면 여기서 직접 load_state_dict를 호출하세요.

    # 2) 테스트용 이미지 준비 (256×256 RGB)
    img = Image.open("images/_aTpHvBraCMhi_wxJ3kZSw.png").convert("RGB").resize((256,256))
    img_np = np.array(img)
    # HWC → CHW → add batch dim
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # (GPU 사용 시) tensor = tensor.cuda(); model = model.cuda()

    with torch.no_grad():
        # 3a) raw latent 얻기
        #z_raw = model.first_stage_model.encode(tensor)         # [1, C_latent, 32, 32]

                # 수정 후 (sampling 결과를 직접 가져옴)
        z_dist = model.first_stage_model.encode(tensor)    # DiagonalGaussianDistribution 객체
        z_raw = z_dist.sample()                            # Tensor: [B, 4, H/8, W/8]

        recon_direct = model.first_stage_model.decode(z_raw)

    z_flat = z_raw.view(-1).cpu().numpy()  # flatten 후 numpy로 변환

    print("Latent z 통계값:")
    print(f"  min  : {z_flat.min():.4f}")
    print(f"  max  : {z_flat.max():.4f}")
    print(f"  mean : {z_flat.mean():.4f}")
    print(f"  std  : {z_flat.std():.4f}")

    # 히스토그램 시각화
    plt.figure(figsize=(6, 4))
    plt.hist(z_flat, bins=100, color='steelblue', alpha=0.8)
    plt.title("Histogram of Latent z values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 4) 시각화
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img_np); plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Direct decode(raw→decode)")
    plt.imshow(recon_direct.squeeze().permute(1,2,0).cpu().numpy())
    plt.axis("off")

    plt.tight_layout()
    plt.show()