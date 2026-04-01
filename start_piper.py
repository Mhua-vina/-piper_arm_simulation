import os
import sys
import math
import time
import re
import numpy as np
from isaacgym import gymapi

# ==========================================
# 0. 真机初始化 (CAN 通讯心跳防掉线)
# ==========================================
local_sdk_repo = "/home/qinminghua/piper_arm_project/piper_sdk_repo"
if local_sdk_repo not in sys.path:
    sys.path.insert(0, local_sdk_repo)

try:
    from piper_sdk import C_PiperInterface
    print("✅ 成功导入 PiPER SDK!")
except ImportError as e:
    print(f"❌ 导入失败。请检查路径: {local_sdk_repo}")
    sys.exit()

print("⏳ 正在连接实体机械臂...")
piper = C_PiperInterface("can0")
piper.ConnectPort()
piper.EnableArm(7) # 激活电机使能
time.sleep(1) 

try:
    piper.MotionCtrl_2(0x01, 0x01, 100) 
    print("✅ 实体机械臂已进入 CAN 控制模式！")
except Exception as e:
    print(f"⚠️ 模式切换提示: {e}")

# ==========================================
# 1. 官方原生 URDF 路径与兼容处理
# ==========================================
asset_root = "/home/qinminghua/piper_arm_project/piper_new_assets/asset"
original_urdf = os.path.join(asset_root, "urdf/piper_x_description_isaacgym.urdf")
fixed_urdf = os.path.join(asset_root, "urdf/piper_ready.urdf")

if not os.path.exists(original_urdf):
    print(f"❌ 找不到官方原版文件：{original_urdf}")
    sys.exit()

with open(original_urdf, 'r') as f:
    content = f.read()
content = re.sub(r'filename="package://[^/]+/meshes/', 'filename="meshes/', content)
with open(fixed_urdf, 'w') as f:
    f.write(content)
print("🔧 官方模型加载准备就绪...")

# ==========================================
# 2. 核心系统初始化 (物理引擎精调)
# ==========================================
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8 
sim_params.physx.num_velocity_iterations = 2
sim_params.physx.contact_offset = 0.01        
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.5
sim_params.physx.max_depenetration_velocity = 1.0 

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 0.5
plane_params.dynamic_friction = 0.5
gym.add_ground(sim, plane_params)

# ==========================================
# 3. 资产加载 (翻转视觉 + 加载真实材质)
# ==========================================
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True 
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False
asset_options.thickness = 0.01
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.use_mesh_materials = True
asset_options.use_physx_armature = True
asset_options.vhacd_enabled = True

print("⌛ 正在加载原生高精度模型...")
robot_asset = gym.load_asset(sim, asset_root, "urdf/piper_ready.urdf", asset_options)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# ==========================================
# 4. 姿态与动力学设置 (柔和操控手感)
# ==========================================
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0) 
axis = gymapi.Vec3(0.0, 0.0, 1.0) 
angle = -math.pi / 2               
pose.r = gymapi.Quat.from_axis_angle(axis, angle)

robot_handle = gym.create_actor(env, robot_asset, pose, "piper_robot", 0, 1)

dof_props = gym.get_actor_dof_properties(env, robot_handle)
for i in range(len(dof_props)):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 400.0  # 柔和支撑，防硬碰硬
    dof_props['damping'][i] = 40.0     # 抹除震荡

gym.set_actor_dof_properties(env, robot_handle, dof_props)

num_dofs = gym.get_asset_dof_count(robot_asset)
initial_pos = np.zeros(num_dofs, dtype=np.float32)
gym.set_actor_dof_position_targets(env, robot_handle, initial_pos)

# ==========================================
# 5. 相机与主循环 (虚实同步通信)
# ==========================================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
cam_pos = gymapi.Vec3(1.2, -1.2, 0.8)  
cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("🚀 虚实同步完全体启动！请缓慢拖动左侧面板滑块，实体机械臂将完美跟随！")

send_rate = 0.02 # 50Hz 刷新率
last_send_time = time.time()
last_gripper_pulse = -1

try:
    while not gym.query_viewer_has_closed(viewer):
        # 1. 物理计算与渲染
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        # 2. 读取面板滑块数据并执行 (安全更新仿真环境)
        targets = gym.get_actor_dof_position_targets(env, robot_handle)
        safe_targets = np.array(targets, dtype=np.float32)
        gym.set_actor_dof_position_targets(env, robot_handle, safe_targets)
        
        # 3. 硬件 CAN 总线同步
        current_time = time.time()
        if current_time - last_send_time >= send_rate:
            try:
                # A. 挂挡心跳 (防掉线)
                piper.MotionCtrl_2(0x01, 0x01, 100)
                
                # B. 前 6 轴同步
                if num_dofs >= 6:
                    scale = 1000 
                    j = [int(float(safe_targets[i]) * (180.0 / math.pi) * scale) for i in range(6)]
                    piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
                
                # C. 第 7 轴 (夹爪) 满血同步
                if num_dofs >= 7:
                    finger_val = abs(float(safe_targets[6]))
                    max_sim_open = 0.04 
                    max_real_pulse = 80000 
                    
                    open_ratio = min(finger_val / max_sim_open, 1.0)
                    target_pulse = int(open_ratio * max_real_pulse)
                    
                    # 差值防抖，拒绝堵塞 CAN 端口
                    if abs(target_pulse - last_gripper_pulse) > 1000:
                        piper.GripperCtrl(target_pulse, 5000, 0x01, 0x00)
                        last_gripper_pulse = target_pulse 
                        print(f"📡 夹爪动作 -> 比例: {open_ratio*100:05.1f}% | 指令脉冲: {target_pulse}      ", end='\r')
                        
            except Exception as e:
                pass # 忽略微小报错，保持系统流畅运行
                
            last_send_time = current_time

except KeyboardInterrupt:
    print("\n👋 收到中断信号，正在退出...")
finally:
    print("🛑 正在断开连接并释放实体机械臂...")
    try:
        piper.DisableArm(7)
    except:
        pass
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("✅ 进程已完全退出。")