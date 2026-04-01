import math
import ikpy.chain
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 加载 Piper 机械臂的物理模型 (URDF)
# ==========================================
# ⚠️ 注意：确保这个路径是你电脑里真实的 URDF 路径
urdf_path = "/home/qinminghua/piper_arm_project/Piper_ros/src/piper_description/urdf/piper_description.urdf"

print("⏳ 正在解析 URDF 构建运动学链...")
# 自动提取从基座到末端执行器的核心骨架
my_chain = ikpy.chain.Chain.from_urdf_file(urdf_path)

# ==========================================
# 2. 导入 AI 算出的 6D 目标位姿
# ==========================================
# XYZ 坐标 (注意：Z轴加了 0.3 的高度补偿，防止机械臂往地底下钻导致无解)
target_pos = [-0.0295, -0.2877, -0.2068 + 0.3] 
# Rx, Ry, Rz 旋转角 (弧度)
target_rpy = [0.1156, -0.4668, -0.9577]       

# 将欧拉角转换为 3x3 的旋转矩阵
rot_matrix = R.from_euler('xyz', target_rpy).as_matrix()

# ==========================================
# 3. 召唤 IK 数学引擎求解！
# ==========================================
print(f"🎯 目标空间位置: {target_pos}")
print("🧠 正在进行逆运动学反解 (IK)...")

# 求解目标角度
ik_angles = my_chain.inverse_kinematics(
    target_position=target_pos, 
    target_orientation=rot_matrix, 
    orientation_mode="all"
)

# ==========================================
# 4. 提取真实电机的关节角度
# ==========================================
print("\n✅ 求解成功！各个电机的目标角度 (弧度制):")
joint_angles = []

# ikpy 算出的数组包含了基座和固定关节，我们需要过滤出真正的活动关节
for i, link in enumerate(my_chain.links):
    # 如果关节名字里有 joint，且不是固定的 base，就把角度提取出来
    if "joint" in link.name.lower() and "fixed" not in link.name.lower():
        angle = ik_angles[i]
        joint_angles.append(angle)
        print(f" ⚙️ {link.name}: {angle:>8.4f} rad ({math.degrees(angle):>6.2f} 度)")

# 提取前 6 个核心自由度，准备发送给仿真器
final_6_dof = [float(f"{a:.4f}") for a in joint_angles[:6]]
print("\n=========================================")
print("🚀 请将以下数组复制到你的 sim_grasp.py 脚本中：")
print(f"target_joint_angles = {final_6_dof}")
print("=========================================\n")