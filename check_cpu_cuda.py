import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"""This device supports {DEVICE} as processing unit.""")