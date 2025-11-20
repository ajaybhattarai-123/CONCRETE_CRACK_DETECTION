import sys
print("Python executable:", sys.executable)
print("Attempting to import torch...")
try:
    import torch
    print("torch imported successfully.")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Error importing torch:", e)

print("Attempting to import torchvision...")
try:
    import torchvision
    print("torchvision imported successfully.")
except Exception as e:
    print("Error importing torchvision:", e)
