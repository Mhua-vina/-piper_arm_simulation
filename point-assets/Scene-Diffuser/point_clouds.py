import open3d as o3d
import numpy as np

# ==========================================
# 1. 生成目标物体的点云（蓝色的杯子）
# ==========================================
mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.15)
pcd = mesh.sample_points_uniformly(number_of_points=2000)
pcd.paint_uniform_color([0, 0.6, 0.8]) 

# ==========================================
# 2. 生成红球：目标点 (Target)
# ==========================================
# 核心逻辑：自动计算点云的几何中心
center_coordinate = pcd.get_center() 

# 创建一个半径 1 厘米的球体，涂成红色
target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
target_sphere.paint_uniform_color([1, 0, 0])
# 将红球移动到计算出的点云中心
target_sphere.translate(center_coordinate)

# ==========================================
# 3. 生成绿球：机械臂末端夹爪 (TCP)
# ==========================================
# 假设机械臂现在悬停在杯子正上方 10 厘米处 (Z轴 + 0.1)
tcp_start_pos = center_coordinate + np.array([0.0, 0.0, 0.1])

tcp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
tcp_sphere.paint_uniform_color([0, 1, 0])
tcp_sphere.translate(tcp_start_pos)

# ==========================================
# 4. 可视化：把它们放在一起看
# ==========================================
print(f"📊 目标红球坐标: {center_coordinate}")
print(f"📊 夹爪绿球坐标: {tcp_start_pos}")
print("🚀 请用鼠标旋转查看：我们的目标就是让绿球去重合红球！")

o3d.visualization.draw_geometries([pcd, target_sphere, tcp_sphere])