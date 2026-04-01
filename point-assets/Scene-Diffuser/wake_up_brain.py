import torch
import os

print("🔌 [1/3] 正在检测系统硬件...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 当前使用设备: {device}")

# 我们要测试的两个大脑记忆文件
brain_files = ["model.pth", "modelplan.pth"]

print("\n🧠 [2/3] 开始依次唤醒大脑记忆库...\n")

for file_name in brain_files:
    ckpt_path = os.path.join("checkpoints", file_name)
    print(f"▶️ 正在测试: {file_name}")
    
    # 1. 检查文件是否存在
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到文件 {ckpt_path}，请确认文件就在 checkpoints 文件夹里！\n")
        continue
        
    # 2. 尝试将记忆注入显卡
    try:
        memory = torch.load(ckpt_path, map_location=device)
        print(f"  ✨ 点亮成功！{file_name} 已顺利苏醒！")
        
        # 偷窥内部结构 (找出作者把核心权重藏在哪个字典键里了)
        if isinstance(memory, dict):
            keys = list(memory.keys())
            print(f"  📊 记忆区块包含: {keys}\n")
        else:
            print("  📊 这是一个纯张量记忆库。\n")
            
    except Exception as e:
        print(f"  ❌ 唤醒失败，出现排斥反应: {e}\n")

print("🏁 测试结束！")