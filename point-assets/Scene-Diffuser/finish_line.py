import os

fpath = 'models/visualizer.py'
with open(fpath, 'a') as f:
    f.write("""
def safe_visualize(self, model, dataloader, vis_dir, **kwargs):
    import os, torch, numpy as np
    os.makedirs(vis_dir, exist_ok=True)
    print('\\n🎯 AI大脑推理已完成！正在拦截抓取数据...')
    for data in dataloader:
        ksample = getattr(self, 'ksample', 1)
        outputs = model.sample(data, k=ksample)
        save_path = os.path.join(vis_dir, 'final_grasp.npy')
        np.save(save_path, outputs.cpu().numpy())
        print(f'\\n🏆🏆🏆 伟大胜利！抓取坐标已成功提取并保存至：\\n{save_path}\\n')
        return # 拿到数据立刻跑路，绝不画图！

GraspGenVisualizer.visualize = safe_visualize
""")

print("✅ 终极拦截网部署完毕！画图功能已阉割，准备直取核心坐标！")
