from api import CtrLoRA

ctrlora = CtrLoRA(num_loras=1)
ctrlora.create_model(
    sd_file='ckpts/sd15/v1-5-pruned-noVAE.ckpt',
    #sd_file='ckpts/sd15/v1-5-pruned.ckpt',
    basecn_file='ckpts/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt',
    lora_files=('ckpts/ctrlora-loras/ctrlora_sd15_basecn700k_seg_rank128.ckpt'),
                #'ckpts/ctrlora-loras/ctrlora_sd15_basecn700k_canny_rank128.ckpt'),
                #D:\ctrlora\ckpts\ctrlora-loras\ctrlora_sd15_basecn700k_seg_rank128.ckpt
                #runs/myData/lightning_logs/version_6/checkpoints/epoch=101-step=26621.ckpt
)
#ctrlora.load_conventional_lora('D:ckpts/lora/last.safetensors')
##'D:ckpts/lora/JZCG005-Realistic city photography 1.0.safetensors'

samples = ctrlora.sample(
    cond_image_paths=('images/_aTpHvBraCMhi_wxJ3kZSw.jpg'),            
                   #'images/_aTpHvBraCMhi_wxJ3kZSw_canny.jpg'),
                     # D:\ctrlora\images\_1Gn_xkw7sa_i9GU4mkxxQ_canny.jpg    - 도시
                     # D:\ctrlora\images\_1Gn_xkw7sa_i9GU4mkxxQ.png
                    #('D:\ctrlora\images\_aTpHvBraCMhi_wxJ3kZSw.png',             - 숲
                    #'D:\ctrlora\images\_aTpHvBraCMhi_wxJ3kZSw_canny.jpg'),
    #prompt='photo of load, bike, people, bulding',
    #prompt='photo of a car driving down a road next to a forest',
    prompt='',
    n_prompt='worst quality',
    num_samples=1,
)
for img in samples:
    img.show()