import torch
import os

def strip_first_stage(ckpt_in, ckpt_out):
    # 1) 원본 체크포인트 로드
    ckpt = torch.load(ckpt_in, map_location="cpu")
    # 2) state_dict 추출
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    # 3) first_stage_model 키 제거
    new_sd = {k: v for k, v in sd.items() if not k.startswith("first_stage_model.")}
    # 4) 원본 포맷 그대로 재조립
    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_sd
    else:
        ckpt = new_sd
    # 5) 새 파일로 저장
    torch.save(ckpt, ckpt_out)
    print(f"Stripped first_stage_model keys: {len(sd)-len(new_sd)} removed. Saved to {ckpt_out}")

# 사용 예
input_ckpt  = "../ckpts/sd15/v1-5-pruned.ckpt"
output_ckpt = "../ckpts/sd15/v1-5-pruned-noVAE.ckpt"
strip_first_stage(input_ckpt, output_ckpt)
