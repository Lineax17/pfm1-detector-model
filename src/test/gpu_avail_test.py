import torch

print("PyTorch:", torch.__version__)
print("CUDA verfügbar:", torch.cuda.is_available())
print("Anzahl GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))