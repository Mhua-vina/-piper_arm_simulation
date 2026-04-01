import os

fpath = 'models/visualizer.py'
with open(fpath, 'a') as f:
    f.write("""
def safe_visualize(self, model, dataloader, vis_dir, **kwargs):
    import os, torch, numpy as np
    os.makedirs(vis_dir, exist_ok=True)
    print('\\n🎯 AI大脑准备就绪！正在将数据传送至显卡...')
    
    # 1. 自动侦测 AI 大脑所在的设备 (GPU)
    device = next(model.parameters()).device
    
    # 2. 写一个万能传送门，把字典里所有的张量都传送到 GPU
    def to_device(d, dev):
        if isinstance(d, dict): return {k: to_device(v, dev) for k, v in d.items()}
        if isinstance(d, list): return [to_device(v, dev) for v in d]
        if isinstance(d, torch.Tensor): return d.to(dev)
        return d

    for data in dataloader:
        # 3. 把数据全部搬过去！
        data = to_device(data, device)
        
        ksample = getattr(self, 'ksample', 1)
        # 4. 显卡内极速推理！
        outputs = model.sample(data, k=ksample)
        
        # 5. 把算好的结果从显卡拿回内存，保存为 npy
        save_path = os.path.join(vis_dir, 'final_grasp.npy')
        np.save(save_path, outputs.cpu().detach().numpy())
        print(f'\\n🏆🏆🏆 伟大胜利！抓取坐标已成功提取并保存至：\\n{save_path}\\n')
        return

GraspGenVisualizer.visualize = safe_visualize
""")

print("✅ 显卡数据传送门搭建完毕！CPU和GPU已彻底打通！")
