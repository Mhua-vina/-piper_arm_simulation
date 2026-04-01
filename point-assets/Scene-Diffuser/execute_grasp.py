import time
import math
import sys

# ==========================================
# 0. 挂载本地 SDK 源码 (请确认路径是否正确)
# ==========================================
local_sdk_repo = "/home/qinminghua/piper_arm_project/piper_sdk_repo"
if local_sdk_repo not in sys.path:
    sys.path.insert(0, local_sdk_repo)

try:
    from piper_sdk import C_PiperInterface
    print("✅ 成功导入 PiPER SDK!")
except ImportError as e:
    print(f"❌ 导入失败。请检查路径: {local_sdk_repo}")
    exit()

# ==========================================
# 1. 填入 AI 算出的目标数据
# ==========================================
# 暴力映射：我们直接把 XYZ 和 RxRyRz 当成 J1-J6 的弧度值
ai_wrist_pose = [-0.02949, -0.28768, -0.20678, 0.11560, -0.46687, -0.95777]
gripper_val = 0.8 # 夹爪闭合

# ==========================================
# 2. 初始化实体机械臂并发送指令
# ==========================================
print("⏳ 正在连接实体机械臂...")
piper = C_PiperInterface("can0")
piper.ConnectPort()
piper.EnableArm(7) # 激活使能
time.sleep(1) 

try:
    # 切换到位置控制模式 (0x01)
    piper.MotionCtrl_2(0x01, 0x01, 50) # 速度设低一点 (50)，安全第一
    print("✅ 实体机械臂已进入 CAN 控制模式！")
    
    # 换算逻辑：将 AI 的假想弧度，乘以 180/pi 转为角度，再乘以 1000 符合电机通讯协议
    scale = 1000
    j = [int(float(val) * (180 / math.pi) * scale) for val in ai_wrist_pose]
    
    print(f"📡 正在发送底层电机指令: J1:{j[0]} J2:{j[1]} J3:{j[2]} J4:{j[3]} J5:{j[4]} J6:{j[5]}")
    
    # 执行 5 秒的持续发送，确保电机走到位
    start_time = time.time()
    while time.time() - start_time < 5.0:
        piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
        
        # 夹爪控制
        g_val = int(abs(gripper_val) * 1000)
        piper.GripperCtrl(min(g_val, 1000), 50)
        time.sleep(0.02)
        
    print("🎉 动作执行完毕！链路完全畅通！")

except Exception as e:
    print(f"❌ 发送指令报错: {e}")
finally:
    print("🛑 正在断开连接并下电...")
    try:
        piper.DisableArm(7)
    except:
        pass