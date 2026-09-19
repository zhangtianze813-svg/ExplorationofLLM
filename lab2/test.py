import torch

# 检查 GPU 是否可用
if torch.cuda.is_available():
    # 获取当前 GPU 的名称
    gpu_name = torch.cuda.get_device_name(0)
    print(f"检测到 GPU 型号: {gpu_name}")
else:
    print("未检测到可用的 NVIDIA GPU，当前运行在 CPU 上。")