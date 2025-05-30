import torch
import torch.nn as nn
from safetensors.torch import load_file
from cldm.model import load_state_dict

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )

def Decoder(latent_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth", latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        if latent_channels is None:
            latent_channels = self.guess_latent_channels(str(encoder_path))
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        
        
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cuda", weights_only=True))
            #self.encoder.load_state_dict(load_file(encoder_path))  
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cuda", weights_only=True))  # pth
            #self.decoder.load_state_dict(load_file(decoder_path))   # safetensors

 
    def guess_latent_channels(self, encoder_path):
        """guess latent channel count based on encoder filename"""
        if "taef1" in encoder_path:
            return 16
        if "taesd3" in encoder_path:
            return 16
        return 4
    
    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)
    
    def encode(self, x):
        # 1) 픽셀 스케일 [-1,1] → [0,1]
        x = (x + 1) * 0.5
        # 2) raw latent
        z = self.encoder(x)
        z = z*10
        return z

    def decode(self, z):
        # decoder → [0,1]
        z = self.decoder(z)
        # [0,1] → [-1,1]
        #return z 
        return z * 2 - 1
