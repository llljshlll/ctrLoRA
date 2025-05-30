import torch

def remap_checkpoint_keys(src_path, dst_path, rename_fn):
    # 1) 기존 체크포인트 로드
    ckpt = torch.load(src_path, map_location="cpu")
    # 2) state_dict만 추출(감싸여 있다면)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # 3) 새 딕셔너리에 키 리매핑하여 복사
    new_sd = {}
    for k, v in sd.items():
        new_key = rename_fn(k)
        new_sd[new_key] = v

    # 4) (원래 포맷 유지해서) 다시 저장
    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_sd
        torch.save(ckpt, dst_path)
    else:
        torch.save(new_sd, dst_path)

# 예시: “0.weight” → “first_stage_model.encoder.0.weight” 같은 규칙
def rename_fn(old_key):
    return f"first_stage_model.encoder.{old_key}"

# 사용 예
remap_checkpoint_keys(
    src_path="../configs/inference/road_decoder.pth",
    dst_path="../configs/inference/road_decoder_remapped.pth",
    rename_fn=rename_fn
)
