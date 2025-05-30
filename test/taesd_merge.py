import torch

# 1) 원본 no-VAE 체크포인트 로드
orig = torch.load("ckpts/sd15/v1-5-pruned-noVAE.ckpt", map_location="cpu")
state = orig.get("state_dict", orig).copy()

# 2) TAESD encoder & decoder 로드
enc = torch.load("configs/inference/model/taesd_encoder.pth", map_location="cpu")
dec = torch.load("configs/inference/model/taesd_decoder.pth", map_location="cpu")

# 3) 키 앞에 prefix를 붙여서 state_dict에 합치기
#    (실제 키 네이밍에 맞게 "first_stage_model.encoder." 등을 조정하세요)
for k, v in enc.items():
    new_key = f"first_stage_model.encoder.{k}"
    state[new_key] = v

for k, v in dec.items():
    new_key = f"first_stage_model.decoder.{k}"
    state[new_key] = v

# 4) merged ckpt 구성 및 저장
merged = orig.copy()
merged["state_dict"] = state
torch.save(merged, "ckpts/v1-5-pruned-with-TAESD.ckpt")
print("✅ Saved merged checkpoint to ckpts/v1-5-pruned-with-TAESD.ckpt")
