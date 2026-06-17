#!/usr/bin/env python3

import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
from box_grasp_detection.msg import BoxGraspArray
import sensor_msgs.point_cloud2 as pc2

def pointcloud2_to_o3d(cloud_msg):
    """转换ROS PointCloud2到Open3D点云"""
    print(f"Converting PointCloud2 with {cloud_msg.width * cloud_msg.height} points")
    
    # 从PointCloud2提取点
    points = []
    colors = []
    
    for point in pc2.read_points(cloud_msg, skip_nans=True):
        if len(point) >= 3:  # 至少包含x,y,z
            points.append([point[0], point[1], point[2]])
            
            # 尝试提取颜色信息
            if len(point) >= 6:  # 包含RGB
                colors.append([point[3]/255.0, point[4]/255.0, point[5]/255.0])
            else:
                colors.append([0.5, 0.5, 0.5])  # 灰色
    
    print(f"Extracted {len(points)} valid points")
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    if points:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def simple_visualizer():
    """简化的可视化器用于调试"""
    rospy.init_node('debug_visualizer')
    
    print("Waiting for point cloud message...")
    cloud_msg = rospy.wait_for_message('/camera/depth/color/points', PointCloud2, timeout=10)
    print("Got point cloud message!")
    
    print("Converting to Open3D...")
    pcd = pointcloud2_to_o3d(cloud_msg)
    print(f"Conversion complete: {len(pcd.points)} points")
    
    print("Creating visualizer...")
    vis = o3d.visualization.Visualizer()
    
    print("Creating window...")
    result = vis.create_window(window_name='Debug Test', width=800, height=600)
    print(f"Window creation result: {result}")
    
    if result:
        print("Adding point cloud...")
        vis.add_geometry(pcd)
        print("Point cloud added")
        
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coord_frame)
        print("Coordinate frame added")
        
        print("Starting visualization loop...")
        count = 0
        while not rospy.is_shutdown() and count < 100:  # 最多100帧后自动退出
            if not vis.poll_events():
                print("Window closed by user")
                break
            vis.update_renderer()
            count += 1
            rospy.sleep(0.05)
        
        print("Destroying window...")
        vis.destroy_window()
        print("Visualization complete")
    else:
        print("Failed to create window!")

if __name__ == '__main__':
    try:
        simple_visualizer()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()