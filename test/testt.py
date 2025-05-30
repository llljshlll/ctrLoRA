import torch

# 1) 학습된 TAESD 모델의 전체 state_dict 불러오기
ckpt = torch.load("D:\stable_diffusion\ctrlora\configs\inference\model\model.pth", map_location="cpu")

# 2) encoder 파라미터만 모아서 저장
encoder_sd = {
    k.replace("encoder.", ""): v
    for k, v in ckpt.items()
    if k.startswith("encoder.")
}
torch.save(encoder_sd, "D:/stable_diffusion/ctrlora/configs/inference/model/regul_encoder.pth")

# 3) decoder 파라미터만 모아서 저장
decoder_sd = {
    k.replace("decoder.", ""): v
    for k, v in ckpt.items()
    if k.startswith("decoder.")
}
torch.save(decoder_sd, "D:/stable_diffusion/ctrlora/configs/inference/model/regul_decoder.pth")