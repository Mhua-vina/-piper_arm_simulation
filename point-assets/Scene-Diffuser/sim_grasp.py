import math
import time
from isaacgym import gymapi

# ==========================================
# 1. 初始化 Isaac Gym 纯虚拟环境
# ==========================================
print("⏳ 正在启动 Isaac Gym 虚拟测试场...")
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
# 2. 加载虚拟机械臂 (Piper)
# ==========================================
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

# ⚠️ 注意：请确保这个路径是你电脑里真实的 Piper URDF 路径
asset_root = "/home/qinminghua/piper_arm_project/Piper_ros/src/piper_description"
robot_urdf_file = "urdf/piper_description.urdf"
robot_asset = gym.load_asset(sim, asset_root, robot_urdf_file, asset_options)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# 初始化机械臂姿态
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
robot_handle = gym.create_actor(env, robot_asset, pose, "PiPER", 0, 1)

# 设置虚拟电机的刚度和阻尼
props = gym.get_actor_dof_properties(env, robot_handle)
props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
props["stiffness"].fill(2500.0) 
props["damping"].fill(200.0)     
gym.set_actor_dof_properties(env, robot_handle, props)

# ==========================================
# 3. 🎯 放置 AI 的“抓取目标点”(可视化)
# ==========================================
# AI 算出的目标 XYZ: [-0.0295, -0.2877, -0.2068]
# 因为 AI 的坐标系可能和世界坐标系有偏移，我们先加一个 Z 轴的基础高度 0.3
target_x, target_y, target_z = -0.0295, -0.2877, -0.2068 + 0.3 

box_opts = gymapi.AssetOptions()
box_asset = gym.create_box(sim, 0.02, 0.02, 0.02, box_opts)
box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(target_x, target_y, target_z)
# 放一个红色的小方块代表 AI 算出来的目标位置
target_handle = gym.create_actor(env, box_asset, box_pose, "AI_Target", 0, 1)
gym.set_rigid_body_color(env, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))

# ==========================================
# 4. 视角与主循环
# ==========================================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, 1, 1))

cam_pos = gymapi.Vec3(1.0, -1.0, 0.8)
cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("🚀 纯虚拟测试场已启动！你可以用鼠标拖动视角，看到机械臂和红色的 AI 目标点！")

try:
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
except KeyboardInterrupt:
    print("\n程序中断")
finally:
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)