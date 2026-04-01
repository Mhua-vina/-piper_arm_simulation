import os
import math
import time
import sys
from isaacgym import gymapi

# ==========================================
# 0. 强行挂载本地 SDK 源码
# ==========================================
local_sdk_repo = "/home/qinminghua/piper_arm_project/piper_sdk_repo"
if local_sdk_repo not in sys.path:
    sys.path.insert(0, local_sdk_repo)

try:
    from piper_sdk import C_PiperInterface
    print("✅ 物理指路成功！成功导入 PiPER SDK!")
except ImportError as e:
    print(f"❌ 导入失败。请检查路径: {local_sdk_repo}")
    exit()

# ==========================================
# 1. 初始化实体机械臂 (CAN 通讯)
# ==========================================
print("⏳ 正在连接实体机械臂...")
piper = C_PiperInterface("can0")
piper.ConnectPort()
piper.EnableArm(7) # 激活使能
time.sleep(1) 
try:
    piper.MotionCtrl_2(0x01, 0x01, 100) 
    print("✅ 实体机械臂已进入 CAN 控制模式！")
except Exception as e:
    print(f"⚠️ 模式切换提示: {e}")

# ==========================================
# 2. 初始化 Isaac Gym 仿真环境
# ==========================================
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 2
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
gym.add_ground(sim, plane_params)

# ==========================================
# 3. 加载虚拟模型与姿态矫正 (完全同步你正常版的逻辑)
# ==========================================
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.override_com = True
asset_options.override_inertia = True
asset_options.armature = 0.01
asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

asset_root = "/home/qinminghua/piper_arm_project/Piper_ros/src/piper_description"
robot_urdf_file = "urdf/piper_description.urdf"
robot_asset = gym.load_asset(sim, asset_root, robot_urdf_file, asset_options)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

pose = gymapi.Transform()
# 使用你测试正常的参数：高度 0.05，绕 Z 轴旋转 -90 度
pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
axis = gymapi.Vec3(0.0, 0.0, 1.0) 
angle = -math.pi / 2               
pose.r = gymapi.Quat.from_axis_angle(axis, angle)

robot_handle = gym.create_actor(env, robot_asset, pose, "PiPER", 0, 1)

# --- 检查点：确保这里的函数名完整 ---
props = gym.get_actor_dof_properties(env, robot_handle)
props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
props["stiffness"].fill(2500.0) 
props["damping"].fill(200.0)     
# 下面这行一定要写完整，不能只写 gym.set
gym.set_actor_dof_properties(env, robot_handle, props)

num_dofs = gym.get_asset_dof_count(robot_asset)
default_dof_pos = [0.0] * num_dofs 
gym.set_actor_dof_position_targets(env, robot_handle, default_dof_pos)

# ==========================================
# 4. 视角与主循环
# ==========================================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# 设置光照，防止黑屏
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, 1, 1))

cam_pos = gymapi.Vec3(1.2, -1.2, 0.8)
cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("🚀 虚实同步已启动！缓慢拖动滑块，实体机械臂将同步动作！")

send_rate = 0.02 
last_send_time = time.time()
last_gripper_pulse = -1


try:
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        current_time = time.time()
        if current_time - last_send_time >= send_rate:
            dof_targets = gym.get_actor_dof_position_targets(env, robot_handle)
            
            try:
                # 1. 挂挡心跳：每帧都告诉机械臂“进入控制模式，全速运行”
                # 参数：位置模式(0x01), 指令模式(0x01), 速度100(整数)
                piper.MotionCtrl_2(0x01, 0x01, 100)
                
                # 2. 计算指令（角度制 * 1000）
                scale = 1000 
                # 注意这里统一使用变量名 j
                j = [int(float(dof_targets[i]) * (180 / math.pi) * scale) for i in range(6)]
                
                # 3. 发送给实体：必须使用 j，不要写 j_vals
                if len(j) >= 6:
                    # 调试打印：如果你拖动滑块，看这里的数字变没变
                    # print(f"📡 正在发送: J1:{j[0]} J2:{j[1]}", end='\r')
                    piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
                
               # 4. 夹爪控制 (
                if num_dofs > 6:
                    # A. 读取滑块位移
                    finger_val = abs(float(dof_targets[6]))
                    
                    # B. 参数映射 (0.04 米映射到 80000 脉冲，即 80mm)
                    max_sim_open = 0.04 
                    max_real_pulse = 80000 
                    
                    open_ratio = min(finger_val / max_sim_open, 1.0)
                    target_pulse = int(open_ratio * max_real_pulse)
                    
                    # C. 防抖过滤
                    if abs(target_pulse - last_gripper_pulse) > 1000:
                        
                        # 【👇 救命神级改动 👇】
                        # 参数1：target_pulse (目标位置)
                        # 参数2：5000 (最大扭矩，给足力气！)
                        # 参数3：0x01 (强制使能夹爪电源！)
                        # 参数4：0x00 (不设置零点)
                        piper.GripperCtrl(target_pulse, 5000, 0x01, 0x00)
                        
                        last_gripper_pulse = target_pulse # 更新记忆
                        print(f"📡 夹爪指令下发 -> 比例: {open_ratio*100:05.1f}% | 脉冲: {target_pulse}      ")
                    
            except Exception as e:
                print(f"❌ 发送指令报错: {e}")
            last_send_time = current_time

except KeyboardInterrupt:
    print("\n程序中断")
finally:
    print("🛑 正在断开连接...")
    try:
        piper.DisableArm(7)
    except:
        pass
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    arm_status = piper.GetArmJointMsgs()
print(f"📊 当前机械臂真实位置: {arm_status}")