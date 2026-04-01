import math
import numpy as np
from isaacgym import gymapi
# 引入 scipy 处理旋转，这是机器人学中的“瑞士军刀”
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ==========================================
# 1. 初始化仿真环境
# ==========================================
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 添加地面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# ==========================================
# 2. 创建资产 (Asset)
# ==========================================
asset_options = gymapi.AssetOptions()
asset_options.disable_gravity = True # 让它们浮在空中，方便我们观察

# 目标物体：红球 (代表点云算出来的质心)
target_asset = gym.create_sphere(sim, 0.03, asset_options)

# 虚拟夹爪：绿色的长方体
gripper_asset = gym.create_box(sim, 0.12, 0.03, 0.06, asset_options)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# ==========================================
# 3. 设定“大脑” (未来接入 Scene-Diffuser) 的指令
# ==========================================
# 目标位置 (X, Y, Z)
goal_pos = gymapi.Vec3(0.5, 0.2, 0.3)
# 目标姿态：绕 Y 轴旋转 90 度 (横着抓取)
# 我们先定义一个 Scipy 旋转，再转成 Isaac Gym 能懂的四元数
target_rotation = R.from_euler('y', 90, degrees=True)
q = target_rotation.as_quat() # 得到 [x, y, z, w]
goal_rot = gymapi.Quat(q[0], q[1], q[2], q[3])

# 在环境中放置红球
target_handle = gym.create_actor(env, target_asset, gymapi.Transform(p=goal_pos), "Target", 0, 0)
gym.set_rigid_body_color(env, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))

# 放置初始位置的绿夹爪 (斜着放，方便观察旋转过程)
initial_rot = R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
start_pose = gymapi.Transform(p=gymapi.Vec3(0, 0, 0.6), r=gymapi.Quat(*initial_rot))
gripper_handle = gym.create_actor(env, gripper_asset, start_pose, "Gripper", 0, 0)
gym.set_rigid_body_color(env, gripper_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0, 1, 0))

# ==========================================
# 4. 实时控制主循环
# ==========================================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.2, 1.2, 1.0), goal_pos)

print("🚀 6-DoF 追踪启动！")
print("绿夹爪将同时调整【位置】和【角度】来对准红球...")

pos_speed = 0.005 # 位置移动步长
rot_step = 0.02   # 旋转插值系数 (0-1之间，越大转得越快)

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 1. 获取夹爪当前状态
    states = gym.get_actor_rigid_body_states(env, gripper_handle, gymapi.STATE_ALL)
    curr_p = states['pose']['p'][0] 
    curr_q = states['pose']['r'][0] # [x, y, z, w]

    # 2. 位置追踪逻辑 (Translation)
    diff_p = np.array([goal_pos.x - curr_p[0], goal_pos.y - curr_p[1], goal_pos.z - curr_p[2]])
    dist = np.linalg.norm(diff_p)
    if dist > 0.005:
        step_p = diff_p * (pos_speed / dist)
        new_p = (curr_p[0] + step_p[0], curr_p[1] + step_p[1], curr_p[2] + step_p[2])
    else:
        new_p = (goal_pos.x, goal_pos.y, goal_pos.z)

    # 3. 旋转对齐逻辑 (Orientation using Scipy SLERP)
    # 理由：为了防止旋转时出现剧烈抖动，我们需要平滑插值
    curr_r_obj = R.from_quat([curr_q[0], curr_q[1], curr_q[2], curr_q[3]])
    goal_r_obj = R.from_quat([goal_rot.x, goal_rot.y, goal_rot.z, goal_rot.w])
    
    # 定义插值器：时间 0 是当前角度，时间 1 是目标角度
    slerp_func = Slerp([0, 1], R.from_quat([curr_r_obj.as_quat(), goal_r_obj.as_quat()]))
    # 计算向目标迈进一小步后的新角度
    new_r_obj = slerp_func([rot_step])[0]
    new_q = new_r_obj.as_quat()

    # 4. 把计算好的新位姿塞回仿真环境
    states['pose']['p'][0] = new_p
    states['pose']['r'][0] = (new_q[0], new_q[1], new_q[2], new_q[3])
    gym.set_actor_rigid_body_states(env, gripper_handle, states, gymapi.STATE_POS)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)