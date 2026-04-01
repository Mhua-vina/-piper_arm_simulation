fpath = 'datasets/multidex_shadowhand_ur.py'
with open(fpath, 'r') as f:
    code = f.read()

# 把所有的 'blue_cup' 替换成带加号的 'mock+blue_cup'
code = code.replace("'blue_cup'", "'mock+blue_cup'")

with open(fpath, 'w') as f:
    f.write(code)

print("✅ 命名格式已完美对齐！保存程序的最后一道坎已被踏平！")
