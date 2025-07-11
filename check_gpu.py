import torch

def check_pytorch_gpu():
    """
    检查 PyTorch 是否能成功识别和使用 CUDA GPU。
    """
    print(f"PyTorch 版本: {torch.__version__}")
    
    is_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {is_available}")

    if is_available:
        print("-" * 30)
        # 获取 GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"找到 {gpu_count} 个可用的 GPU。")

        # 获取当前 GPU 的索引和名称
        current_device_id = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device_id)
        print(f"当前使用的 GPU ID: {current_device_id}")
        print(f"当前 GPU 名称: {current_device_name}")
        
        # 尝试创建一个张量并将其移动到 GPU
        try:
            tensor = torch.tensor([1.0, 2.0]).to("cuda")
            print("\n成功在 GPU 上创建了一个张量:")
            print(tensor)
            print("您的环境已为 GPU 加速准备就绪！")
        except Exception as e:
            print(f"\n在尝试使用 GPU 时发生错误: {e}")
            print("请检查您的 PyTorch 和 CUDA 版本是否兼容。")
        print("-" * 30)
    else:
        print("\nPyTorch 未能找到可用的 CUDA GPU。")
        print("请检查以下几点：")
        print("1. 您的 NVIDIA 驱动程序是否已正确安装并为最新版本。")
        print("2. 您安装的 PyTorch 版本是否是为您的 CUDA 版本编译的。")
        print("   (例如，从 PyTorch 官网获取正确的 pip 或 conda 安装命令)。")

if __name__ == "__main__":
    check_pytorch_gpu()