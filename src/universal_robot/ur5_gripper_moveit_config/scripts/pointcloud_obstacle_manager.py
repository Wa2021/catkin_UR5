#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云障碍物管理器 - 直接使用过滤后的点云生成MoveIt障碍物
替代原有的基于YOLO坐标的障碍物生成方式
"""

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, Point
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

class PointCloudObstacleManager:
    def __init__(self):
        rospy.init_node('pointcloud_obstacle_manager')
        
        # 参数配置
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.voxel_size = rospy.get_param('~voxel_size', 0.01)  # 体素大小（米）
        self.min_cluster_size = rospy.get_param('~min_cluster_size', 100)  # 最小簇大小
        self.obstacle_height = rospy.get_param('~obstacle_height', 0.05)  # 障碍物默认高度
        self.update_rate = rospy.get_param('~update_rate', 2.0)  # 更新频率（Hz）
        
        # TF缓冲区
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 数据存储
        self.latest_pointcloud = None
        self.obstacle_id_counter = 0
        
        # ROS订阅者
        self.pointcloud_sub = rospy.Subscriber(
            '/camera/depth/color/points_filtered', 
            PointCloud2, 
            self.pointcloud_callback
        )
        
        # ROS发布者
        self.planning_scene_pub = rospy.Publisher(
            '/planning_scene', 
            PlanningScene, 
            queue_size=1
        )
        
        # 定时器，定期更新障碍物
        if self.update_rate > 0:
            self.update_timer = rospy.Timer(
                rospy.Duration(1.0 / self.update_rate), 
                self.update_obstacles_timer
            )
        
        rospy.loginfo("点云障碍物管理器已启动")
        rospy.loginfo(f"机器人坐标系: {self.robot_frame}")
        rospy.loginfo(f"相机坐标系: {self.camera_frame}")
        rospy.loginfo(f"体素大小: {self.voxel_size}m")
        rospy.loginfo(f"更新频率: {self.update_rate}Hz")
        
        # 等待TF变换可用
        rospy.loginfo("等待TF变换...")
        try:
            self.tf_buffer.lookup_transform(
                self.robot_frame, 
                self.camera_frame, 
                rospy.Time(), 
                rospy.Duration(10.0)
            )
            rospy.loginfo("TF变换就绪")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF变换等待超时: {e}")
    
    def pointcloud_callback(self, msg):
        """点云数据回调"""
        self.latest_pointcloud = msg
        rospy.logdebug(f"接收到过滤后的点云: {msg.width}x{msg.height}")
    
    def update_obstacles_timer(self, event):
        """定时器回调，更新障碍物"""
        if self.latest_pointcloud is not None:
            self.process_pointcloud_to_obstacles(self.latest_pointcloud)
    
    def process_pointcloud_to_obstacles(self, pointcloud_msg):
        """将点云处理为障碍物"""
        try:
            # 提取点云数据
            points = list(pc2.read_points(
                pointcloud_msg, 
                field_names=("x", "y", "z"), 
                skip_nans=True
            ))
            
            if not points:
                rospy.logdebug("点云中没有有效点")
                return
            
            # 转换到机器人坐标系
            transformed_points = self.transform_points_to_robot_frame(
                points, pointcloud_msg.header
            )
            
            if not transformed_points:
                rospy.logwarn("点云坐标变换失败")
                return
            
            # 体素化和聚类
            obstacle_clusters = self.cluster_points(transformed_points)
            
            # 生成障碍物
            collision_objects = self.create_collision_objects(obstacle_clusters)
            
            # 发布到MoveIt
            self.publish_planning_scene(collision_objects)
            
            rospy.loginfo(f"生成了 {len(collision_objects)} 个障碍物")
            
        except Exception as e:
            rospy.logerr(f"处理点云时出错: {e}")
    
    def transform_points_to_robot_frame(self, points, header):
        """将点从相机坐标系转换到机器人坐标系"""
        try:
            # 获取变换
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame, 
                header.frame_id, 
                header.stamp, 
                rospy.Duration(1.0)
            )
            
            transformed_points = []
            
            for point in points:
                # 创建点消息
                point_msg = Point()
                point_msg.x = point[0]
                point_msg.y = point[1] 
                point_msg.z = point[2]
                
                # 应用变换
                transformed_point = tf2_geometry_msgs.do_transform_point(
                    point_msg, transform
                )
                
                transformed_points.append([
                    transformed_point.x, 
                    transformed_point.y, 
                    transformed_point.z
                ])
            
            return np.array(transformed_points)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"坐标变换失败: {e}")
            return None
    
    def cluster_points(self, points):
        """对点进行体素化和简单聚类"""
        if len(points) == 0:
            return []
        
        # 体素化 - 将点分配到网格中
        voxel_coords = np.floor(points / self.voxel_size).astype(int)
        
        # 获取唯一的体素
        unique_voxels = np.unique(voxel_coords, axis=0)
        
        # 简单聚类：连通的体素组成一个障碍物
        clusters = []
        visited = set()
        
        for voxel in unique_voxels:
            voxel_tuple = tuple(voxel)
            if voxel_tuple not in visited:
                cluster = self.get_connected_voxels(voxel, unique_voxels, visited)
                if len(cluster) >= self.min_cluster_size:
                    # 将体素坐标转换回实际坐标
                    cluster_points = np.array(cluster) * self.voxel_size
                    clusters.append(cluster_points)
        
        return clusters
    
    def get_connected_voxels(self, start_voxel, all_voxels, visited):
        """获取连通的体素（简单的邻接搜索）"""
        cluster = []
        stack = [start_voxel]
        
        while stack and len(cluster) < 1000:  # 限制搜索规模
            current = stack.pop()
            current_tuple = tuple(current)
            
            if current_tuple in visited:
                continue
                
            visited.add(current_tuple)
            cluster.append(current)
            
            # 检查6连通邻居
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                neighbor = current + np.array([dx, dy, dz])
                neighbor_tuple = tuple(neighbor)
                
                if (neighbor_tuple not in visited and 
                    any(np.array_equal(neighbor, v) for v in all_voxels)):
                    stack.append(neighbor)
        
        return cluster
    
    def create_collision_objects(self, clusters):
        """为每个点簇创建碰撞对象"""
        collision_objects = []
        
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            
            # 计算簇的边界框
            min_coords = np.min(cluster, axis=0)
            max_coords = np.max(cluster, axis=0)
            
            # 计算中心和尺寸
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            
            # 确保最小尺寸
            size = np.maximum(size, [self.voxel_size * 2] * 3)
            
            # 创建碰撞对象
            collision_object = CollisionObject()
            collision_object.id = f"pointcloud_obstacle_{self.obstacle_id_counter}_{i}"
            collision_object.header.frame_id = self.robot_frame
            collision_object.header.stamp = rospy.Time.now()
            
            # 设置操作为ADD
            collision_object.operation = CollisionObject.ADD
            
            # 创建立方体形状
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [size[0], size[1], size[2]]
            
            collision_object.primitives.append(primitive)
            
            # 设置位置
            pose = geometry_msgs.msg.Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]
            pose.orientation.w = 1.0
            
            collision_object.primitive_poses.append(pose)
            
            collision_objects.append(collision_object)
        
        self.obstacle_id_counter += 1
        return collision_objects
    
    def publish_planning_scene(self, collision_objects):
        """发布规划场景"""
        # 首先清除旧的障碍物
        self.clear_old_obstacles()
        
        # 添加新的障碍物
        planning_scene = PlanningScene()
        planning_scene.world.collision_objects = collision_objects
        planning_scene.is_diff = True
        
        self.planning_scene_pub.publish(planning_scene)
        
        rospy.logdebug(f"发布了 {len(collision_objects)} 个障碍物到规划场景")
    
    def clear_old_obstacles(self):
        """清除旧的点云障碍物"""
        # 创建清除消息
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        
        # 清除所有以"pointcloud_obstacle_"开头的障碍物
        clear_object = CollisionObject()
        clear_object.id = "pointcloud_obstacle_"  # MoveIt会清除所有包含此前缀的对象
        clear_object.operation = CollisionObject.REMOVE
        
        planning_scene.world.collision_objects.append(clear_object)
        self.planning_scene_pub.publish(planning_scene)
        
        rospy.sleep(0.1)  # 给MoveIt一点时间处理清除请求

if __name__ == '__main__':
    try:
        # 导入geometry_msgs（这里修正了之前的导入错误）
        import geometry_msgs.msg
        
        obstacle_manager = PointCloudObstacleManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass