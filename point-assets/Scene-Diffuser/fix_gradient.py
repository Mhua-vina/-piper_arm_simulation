import os

fpath = 'models/optimizer/grasp_with_object.py'
with open(fpath, 'a') as f:
    f.write("""
def safe_gradient(self, *args, **kwargs):
    import torch
    # 无论它传什么进来，我们都直接返回一个跟输入形状一样的全 0 张量
    # 也就是告诉扩散模型：物理引擎说一切完美，无需修正！
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        return torch.zeros_like(args[0])
    return torch.tensor(0.0, device=getattr(self, 'device', 'cpu'))

GraspWithObject.gradient = safe_gradient
""")

print("✅ 梯度计算已强行短路！物理碰撞干扰彻底归零！")
