#!/usr/bin/env python3
"""
GPU训练环境测试脚本
用于验证YOLOv9训练环境是否正确配置
"""

import torch
import sys
from pathlib import Path

def test_gpu_environment():
    """测试GPU训练环境"""
    print("=" * 50)
    print("YOLOv9 GPU训练环境测试")
    print("=" * 50)
    
    # 检查PyTorch
    print(f"✓ PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        
        # 检查GPU信息
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"✓ GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1024**3:.1f} GB)")
    else:
        print("✗ CUDA不可用，将使用CPU训练")
        return False
    
    # 测试设备选择
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"✓ 选择设备: {device}")
        
        # 创建测试张量
        test_tensor = torch.randn(1, 3, 640, 640).to(device)
        print(f"✓ 测试张量创建成功: {test_tensor.shape} on {test_tensor.device}")
        
    except Exception as e:
        print(f"✗ 设备测试失败: {e}")
        return False
    
    # 检查必要的文件
    current_dir = Path(__file__).parent
    required_files = [
        "train_dual.py",
        "models/detect/yolov9-e.yaml",
        "data/hyps/hyp.scratch-high.yaml"
    ]
    
    print("\n检查训练所需文件:")
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
    
    print("\n" + "=" * 50)
    print("环境测试完成！")
    print("可以开始GPU训练。")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_gpu_environment()
    if not success:
        sys.exit(1)
