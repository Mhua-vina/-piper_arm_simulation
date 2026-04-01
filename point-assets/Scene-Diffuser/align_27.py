import os

# 1. 把假数据里的 33 维斩断成 27 维
fpath1 = 'datasets/multidex_shadowhand_ur.py'
if os.path.exists(fpath1):
    with open(fpath1, 'r') as f: code1 = f.read()
    code1 = code1.replace('np.zeros(33', 'np.zeros(27')
    with open(fpath1, 'w') as f: f.write(code1)

# 2. 强行修改大脑初始化配置，只生成 27 根神经
fpath2 = 'sample.py'
if os.path.exists(fpath2):
    with open(fpath2, 'r') as f: code2 = f.read()
    if 'cfg.model.d_x = 27' not in code2:
        # 在创建模型的前一刻，强行把维度锁死在 27
        code2 = code2.replace('model = create_model(', 'cfg.model.d_x = 27\n    model = create_model(')
        with open(fpath2, 'w') as f: f.write(code2)

print("✅ 全局 27 维度对齐完毕！大脑和身体彻底匹配！")
