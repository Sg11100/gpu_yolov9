#!/usr/bin/env python3
"""
YOLOv9 GPU训练完整性测试脚本
验证所有修改是否正确并且可以开始训练
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """测试所有关键模块的导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    test_modules = [
        'models.common',
        'models.experimental', 
        'utils.general',
        'utils.torch_utils',
        'utils.tal.assigner',
        'utils.panoptic.metrics',
        'val_dual'
    ]
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试模型创建...")
    print("=" * 60)
    
    try:
        from models.yolo import Model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 测试YOLOv9-E模型配置
        cfg_path = 'models/detect/yolov9-e.yaml'
        if Path(cfg_path).exists():
            model = Model(cfg_path, ch=3, nc=80, anchors=None)
            model = model.to(device)
            print(f"✓ YOLOv9-E模型创建成功")
            print(f"✓ 模型设备: {next(model.parameters()).device}")
            
            # 测试前向传播
            x = torch.randn(1, 3, 640, 640).to(device)
            with torch.no_grad():
                y = model(x)
            # 检查输出是否为列表或元组
            if isinstance(y, (list, tuple)):
                print(f"✓ 前向传播测试成功，输出数量: {len(y)}")
                print(f"✓ 输出类型: {type(y)}")
            else:
                print(f"✓ 前向传播测试成功，输出形状: {y.shape}")
            return True
        else:
            print(f"✗ 配置文件不存在: {cfg_path}")
            return False
            
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_training_setup():
    """测试训练设置"""
    print("\n" + "=" * 60)
    print("测试训练设置...")
    print("=" * 60)
    
    try:
        from utils.torch_utils import select_device, smart_optimizer
        from utils.loss_tal_dual import ComputeLoss
        from models.yolo import Model
        import yaml
        
        # 设备选择
        device = select_device('0')
        print(f"✓ 设备选择成功: {device}")
        
        # 模型创建
        cfg_path = 'models/detect/yolov9-e.yaml'
        hyp_path = 'data/hyps/hyp.scratch-high.yaml'
        
        if Path(cfg_path).exists() and Path(hyp_path).exists():
            # 加载超参数
            with open(hyp_path, 'r') as f:
                hyp = yaml.safe_load(f)
            
            model = Model(cfg_path, ch=3, nc=80)
            # 添加hyp属性到模型
            model.hyp = hyp
            model = model.to(device)
            print("✓ 训练模型创建成功")
            
            # 优化器创建
            optimizer = smart_optimizer(model, 'SGD', 0.01, 0.937, 5e-4)
            print(f"✓ 优化器创建成功: {type(optimizer).__name__}")
            
            # 损失函数
            compute_loss = ComputeLoss(model)
            print("✓ 损失函数创建成功")
            
            # GradScaler
            scaler = torch.cuda.amp.GradScaler()
            print("✓ GradScaler创建成功")
            
            return True
        else:
            missing = []
            if not Path(cfg_path).exists():
                missing.append(cfg_path)
            if not Path(hyp_path).exists():
                missing.append(hyp_path)
            print(f"✗ 配置文件不存在: {missing}")
            return False
            
    except Exception as e:
        print(f"✗ 训练设置失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试数据配置...")
    print("=" * 60)
    
    try:
        from utils.general import check_dataset
        
        # 检查数据配置文件
        data_path = '/root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/pcbdata1/data/data.yaml'
        if Path(data_path).exists():
            print(f"✓ 数据配置文件存在: {data_path}")
            
            # 尝试加载数据配置
            import yaml
            with open(data_path, 'r') as f:
                data_config = yaml.safe_load(f)
            print(f"✓ 数据配置加载成功")
            print(f"  - 训练路径: {data_config.get('train', 'N/A')}")
            print(f"  - 验证路径: {data_config.get('val', 'N/A')}")
            print(f"  - 类别数量: {data_config.get('nc', 'N/A')}")
            
            return True
        else:
            print(f"✗ 数据配置文件不存在: {data_path}")
            return False
            
    except Exception as e:
        print(f"✗ 数据配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("YOLOv9 GPU训练完整性测试")
    print("测试所有组件是否正确配置...")
    
    tests = [
        test_imports,
        test_model_creation,
        test_training_setup,
        test_data_loading
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    if all(results):
        print("🎉 所有测试通过！YOLOv9 GPU训练环境配置正确。")
        print("可以开始训练：")
        print("  ./train_yolov9_e_gpu.sh")
        print("  或者")
        print("  python train_dual.py [参数...]")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息并修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
