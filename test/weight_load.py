#!/usr/bin/env python3
import torch
import os
import argparse

def inspect_checkpoint(path, device='cpu', out_path="weight.readme"):
    """
    Inspect a model checkpoint and save its structure to a file.

    Parameters:
    - path (str): 체크포인트 파일 경로 (.pth, .pt, .ckpt, .safetensors).
    - device (str): 텐서를 로드할 디바이스 ('cpu' 또는 'cuda').
    - out_path (str): 결과를 저장할 파일 경로.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
        state_dict = load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)

    lines = []
    header = f"{'Key':<100} {'Shape':<40} {'#Params'}"
    sep = "-" * 100
    lines.append(header)
    lines.append(sep)

    total_params = 0
    for key, tensor in state_dict.items():
        shape = tuple(tensor.shape)
        n = tensor.numel()
        total_params += n
        lines.append(f"{key:<100} {str(shape):<40} {n}")

    lines.append(sep)
    lines.append(f"Total parameters: {total_params}")

    # 파일에 저장
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Checkpoint 구조를 '{out_path}'에 저장했습니다.")

def main():
    parser = argparse.ArgumentParser(description="Inspect the structure of a model checkpoint and save to file.")
    parser.add_argument("ckpt_path", help="Path to the checkpoint file.")
    parser.add_argument("--device", default="cpu", help="Device for loading tensors (default: cpu).")
    parser.add_argument("--out", default="weight.readme", help="Output file path (default: weight.readme).")
    args = parser.parse_args()
    inspect_checkpoint(args.ckpt_path, args.device, args.out)

if __name__ == '__main__':
    main()
