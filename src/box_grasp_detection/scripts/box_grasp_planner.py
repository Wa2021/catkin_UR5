#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import PointCloud2
from box_grasp_detection.msg import BoxGrasp, BoxGraspArray
from tf.transformations import quaternion_from_matrix, quaternion_multiply, quaternion_matrix
import tf2_ros
import tf2_geometry_msgs


class BoxGraspPlanner:
    """
    生成和评分多个抓取策略:
    1. 顶部抓取 (Top grasp)
    2. 侧面长边抓取 (Side grasp - long edge)
    3. 侧面短边抓取 (Side grasp - short edge)
    """
    
    def __init__(self):
        rospy.init_node('box_grasp_planner', anonymous=True)
        
        # Parameters
        self.gripper_width = rospy.get_param('~gripper_width', 0.085)  # 85mm max width
        self.gripper_depth = rospy.get_param('~gripper_depth', 0.05)   # 50mm finger depth
        self.approach_distance = rospy.get_param('~approach_distance', 0.15)  # 15cm approach
        self.grasp_offset = rospy.get_param('~grasp_offset', 0.0)  # Offset from surface
        
        # Weight factors for scoring
        self.weight_stability = rospy.get_param('~weight_stability', 0.4)
        self.weight_reachability = rospy.get_param('~weight_reachability', 0.3)
        self.weight_collision = rospy.get_param('~weight_collision', 0.3)

        # 用于给 OBB 的“无方向长边轴”定一个稳定符号，避免 0/180 度来回跳。
        self.axis_sign_references = [
            np.array([1.0, 0.3, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([1.0, 1.0, 0.0], dtype=float),
            np.array([1.0, -1.0, 0.0], dtype=float),
        ]
        
        rospy.loginfo("Box Grasp Planner initialized")
        rospy.loginfo(f"Gripper: width={self.gripper_width}m, depth={self.gripper_depth}m")

    def _normalize_vector(self, vector, fallback):
        """归一化向量，退化时回退到给定方向。"""
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            return vector / norm

        fallback = np.asarray(fallback, dtype=float)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm > 1e-8:
            return fallback / fallback_norm
        return fallback

    def _normalize_quaternion(self, quat):
        quat = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            return quat / norm
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    def _stabilize_in_plane_axis_sign(self, axis, plane_normal):
        """
        把平面内轴当作“无方向轴”处理后，再用一组全局参考给它选定稳定符号。
        这样同一条轴的 0 度和 180 度会落到同一个 canonical 方向。
        """
        axis = self._normalize_vector(axis, [1.0, 0.0, 0.0])
        plane_normal = self._normalize_vector(plane_normal, [0.0, 0.0, 1.0])

        for reference in self.axis_sign_references:
            projected = reference - np.dot(reference, plane_normal) * plane_normal
            projected_norm = np.linalg.norm(projected)
            if projected_norm < 1e-4:
                continue

            projected /= projected_norm
            dot = np.dot(axis, projected)
            if abs(dot) > 1e-3:
                return axis if dot >= 0.0 else -axis

        return axis

    def _canonicalize_box_orientation(self, orientation):
        """
        将输入 OBB 姿态 canonicalize：
        - 盒子法向固定到 +Z 半球
        - 盒子长边只保留“轴”语义，不保留箭头方向
        - 对 0/180 度跳变输出相同的稳定姿态
        """
        rot_matrix = quaternion_matrix(orientation)[:3, :3]

        box_z_axis = self._normalize_vector(rot_matrix[:, 2], [0.0, 0.0, 1.0])
        if np.dot(box_z_axis, np.array([0.0, 0.0, 1.0])) < 0.0:
            box_z_axis = -box_z_axis

        box_x_axis = rot_matrix[:, 0] - np.dot(rot_matrix[:, 0], box_z_axis) * box_z_axis
        if np.linalg.norm(box_x_axis) < 1e-6:
            box_x_axis = rot_matrix[:, 1] - np.dot(rot_matrix[:, 1], box_z_axis) * box_z_axis

        box_x_axis = self._normalize_vector(box_x_axis, [1.0, 0.0, 0.0])
        box_x_axis = self._stabilize_in_plane_axis_sign(box_x_axis, box_z_axis)

        box_y_axis = np.cross(box_z_axis, box_x_axis)
        box_y_axis = self._normalize_vector(box_y_axis, [0.0, 1.0, 0.0])
        box_x_axis = self._normalize_vector(np.cross(box_y_axis, box_z_axis), box_x_axis)

        canonical_rot = np.column_stack([box_x_axis, box_y_axis, box_z_axis])
        if np.linalg.det(canonical_rot) < 0.0:
            canonical_rot[:, 1] *= -1.0

        canonical_quat_matrix = np.eye(4)
        canonical_quat_matrix[:3, :3] = canonical_rot
        canonical_quat = self._normalize_quaternion(quaternion_from_matrix(canonical_quat_matrix))

        return canonical_rot, canonical_quat
    
    def generate_grasps(self, box_center, box_dims, box_orientation, object_cloud, frame_id):
        """
        为检测到的长方体生成多个抓取候选
        
        Args:
            box_center: [x, y, z] 盒子中心
            box_dims: [length, width, height] 盒子尺寸 (已排序: length >= width >= height)
            box_orientation: quaternion [x, y, z, w] 盒子方向
            object_cloud: 物体点云
            frame_id: 坐标系
            
        Returns:
            list of BoxGrasp messages
        """
        start_time = rospy.Time.now()
        grasps = []
        
        length, width, height = box_dims
        canonical_rot, canonical_orientation = self._canonicalize_box_orientation(box_orientation)
        
        # 1. 顶部抓取 (Top Grasp)
        top_grasps = self._generate_top_grasp(
            box_center, box_dims, canonical_rot, canonical_orientation, frame_id)
        grasps.extend(top_grasps)
        
        # 2. 侧面抓取 - 沿长边 (Side Grasp - Long edge)
        if width <= self.gripper_width:
            side_long_grasps = self._generate_side_grasp(
                box_center, box_dims, canonical_rot, canonical_orientation, frame_id,
                grasp_width=width, grasp_type='side_long')
            grasps.extend(side_long_grasps)
        
        # 3. 侧面抓取 - 沿短边 (Side Grasp - Short edge)
        if length <= self.gripper_width:
            side_short_grasps = self._generate_side_grasp(
                box_center, box_dims, canonical_rot, canonical_orientation, frame_id,
                grasp_width=length, grasp_type='side_short')
            grasps.extend(side_short_grasps)
        
        # 评分所有抓取
        for grasp in grasps:
            grasp.score = self._score_grasp(grasp, box_dims)
        
        # 按分数排序
        grasps.sort(key=lambda x: x.score, reverse=True)
        
        # 填充物体点云
        for grasp in grasps:
            grasp.object_cloud = object_cloud
        
        duration_ms = (rospy.Time.now() - start_time).toSec() * 1000.0
        rospy.loginfo(f"[Performance] Grasp Planning Time: {duration_ms:.2f} ms for {len(grasps)} candidates")
        
        return grasps
    
    def _generate_top_grasp(self, center, dims, rot_matrix, canonical_orientation, frame_id):
        """
        生成顶部抓取 (从上方抓取)
        
        简单明了的逻辑：
        - Z轴：固定向下 [0, 0, -1]
        - Y轴：与物体长轴在水平面的投影对齐
        - X轴：由 Y × Z 得到
        """
        grasps = []
        length, width, height = dims
        
        # 检查夹爪是否能抓取
        if width > self.gripper_width and length > self.gripper_width:
            rospy.logwarn(f"Box too large for top grasp: {width}m x {length}m > gripper {self.gripper_width}m")
            return grasps
        
        box_x_axis = rot_matrix[:3, 0]  # 长边方向 (length)
        box_y_axis = rot_matrix[:3, 1]  # 短边方向 (width)
        
        rospy.loginfo(f"📦 物体尺寸: L={length*100:.1f}cm, W={width*100:.1f}cm, H={height*100:.1f}cm")
        
        # ============================================================
        # 固定Z轴向下
        # ============================================================
        gripper_z = np.array([0.0, 0.0, -1.0])
        
        # 生成抓取配置
        grasp_configs = []
        
        # 配置1: 抓取短边 - Y轴对齐物体长轴
        if width <= self.gripper_width:
            # 取物体长轴在水平面的投影作为Y轴
            long_axis_xy = np.array([box_x_axis[0], box_x_axis[1], 0.0])
            if np.linalg.norm(long_axis_xy) > 0.001:
                long_axis_xy = long_axis_xy / np.linalg.norm(long_axis_xy)
            else:
                long_axis_xy = np.array([1.0, 0.0, 0.0])
            
            grasp_configs.append({
                'grasp_dim': width,
                'description': 'top_grasp_short_edge',
                'gripper_y': long_axis_xy,
                'priority': 1.0
            })
        
        # 配置2: 抓取长边 - Y轴对齐物体短轴
        if length <= self.gripper_width:
            short_axis_xy = np.array([box_y_axis[0], box_y_axis[1], 0.0])
            if np.linalg.norm(short_axis_xy) > 0.001:
                short_axis_xy = short_axis_xy / np.linalg.norm(short_axis_xy)
            else:
                short_axis_xy = np.array([0.0, 1.0, 0.0])
            
            grasp_configs.append({
                'grasp_dim': length,
                'description': 'top_grasp_long_edge',
                'gripper_y': short_axis_xy,
                'priority': 0.8
            })
        
        for config in grasp_configs:
            grasp = BoxGrasp()
            grasp.header.frame_id = frame_id
            grasp.header.stamp = rospy.Time.now()
            
            # Box info
            grasp.length, grasp.width, grasp.height = dims
            grasp.box_pose.position = Point(*center)
            grasp.box_pose.orientation = Quaternion(*canonical_orientation)
            grasp.grasp_type = config['description']
            
            # ============================================================
            # 抓取位置：盒子中心上方
            # ============================================================
            grasp_pos = np.array([center[0], center[1], center[2] + height/2 + self.grasp_offset])
            
            # ============================================================
            # 构建旋转矩阵 - 非常简单！
            # Z轴：固定向下 [0, 0, -1]
            # Y轴：物体长轴的水平投影
            # X轴：Y × Z
            # ============================================================
            gripper_y = config['gripper_y']
            
            # X = Y × Z
            gripper_x = np.cross(gripper_y, gripper_z)
            gripper_x = gripper_x / np.linalg.norm(gripper_x)
            
            # 组装旋转矩阵 [X | Y | Z]
            grasp_rot = np.column_stack([gripper_x, gripper_y, gripper_z])
            
            # 转换为quaternion
            grasp_quat_matrix = np.eye(4)
            grasp_quat_matrix[:3, :3] = grasp_rot
            grasp_quat = quaternion_from_matrix(grasp_quat_matrix)
            
            grasp.grasp_pose.header = grasp.header
            grasp.grasp_pose.pose.position = Point(*grasp_pos)
            grasp.grasp_pose.pose.orientation = Quaternion(*grasp_quat)
            
            grasp._priority = config['priority']
            
            rospy.loginfo(f"  ✓ {config['description']}: 夹持={config['grasp_dim']*100:.1f}cm, Y轴=[{gripper_y[0]:.2f}, {gripper_y[1]:.2f}, {gripper_y[2]:.2f}]")
            
            grasps.append(grasp)
        
        return grasps
    
    def _generate_side_grasp(self, center, dims, rot_matrix, canonical_orientation, frame_id, grasp_width, grasp_type):
        """生成侧面抓取"""
        grasps = []
        length, width, height = dims
        
        if grasp_width > self.gripper_width:
            return grasps
        
        # 生成4个侧面抓取 (从4个方向)
        for side_idx in range(4):
            grasp = BoxGrasp()
            grasp.header.frame_id = frame_id
            grasp.header.stamp = rospy.Time.now()
            
            # Box info
            grasp.length, grasp.width, grasp.height = dims
            grasp.box_pose.position = Point(*center)
            grasp.box_pose.orientation = Quaternion(*canonical_orientation)
            grasp.grasp_type = f"{grasp_type}_side{side_idx}"
            
            # 计算抓取位置和方向
            if grasp_type == 'side_long':
                # 沿长边抓取，夹爪夹住width方向
                if side_idx < 2:
                    # 从y方向接近
                    approach_dir = rot_matrix[:3, 1] * (1 if side_idx == 0 else -1)
                    grasp_offset_dist = width/2 + self.grasp_offset
                else:
                    # 从x方向接近
                    approach_dir = rot_matrix[:3, 0] * (1 if side_idx == 2 else -1)
                    grasp_offset_dist = length/2 + self.grasp_offset
            else:  # side_short
                # 沿短边抓取，夹爪夹住length方向
                if side_idx < 2:
                    # 从x方向接近
                    approach_dir = rot_matrix[:3, 0] * (1 if side_idx == 0 else -1)
                    grasp_offset_dist = length/2 + self.grasp_offset
                else:
                    # 从y方向接近
                    approach_dir = rot_matrix[:3, 1] * (1 if side_idx == 2 else -1)
                    grasp_offset_dist = width/2 + self.grasp_offset
            
            # 位置：盒子中心，从侧面偏移
            grasp_pos = np.array(center) + approach_dir * grasp_offset_dist
            
            # 调整到合适的高度（盒子中间）
            grasp_pos += rot_matrix[:3, 2] * 0  # 可以调整高度偏移
            
            # 方向：夹爪朝向盒子中心
            # 构建旋转矩阵：x轴指向接近方向，z轴向上
            x_axis = -approach_dir  # 指向盒子
            z_axis = rot_matrix[:3, 2]  # 保持向上
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)  # 重新正交化
            
            grasp_rot = np.column_stack([x_axis, y_axis, z_axis])
            grasp_quat_matrix = np.eye(4)
            grasp_quat_matrix[:3, :3] = grasp_rot
            grasp_quat = quaternion_from_matrix(grasp_quat_matrix)
            
            grasp.grasp_pose.header = grasp.header
            grasp.grasp_pose.pose.position = Point(*grasp_pos)
            grasp.grasp_pose.pose.orientation = Quaternion(*grasp_quat)
            
            grasps.append(grasp)
        
        return grasps
    
    def _score_grasp(self, grasp, box_dims):
        """
        评分抓取质量
        考虑因素:
        1. 稳定性 (重心，接触面积，夹持方向)
        2. 可达性 (高度，方向)
        3. 碰撞风险
        4. 物体长短边优先级（新增）
        """
        score = 0.0
        length, width, height = box_dims
        
        # 获取优先级（如果有）
        priority_bonus = getattr(grasp, '_priority', 1.0)
        
        # 1. 稳定性评分
        stability_score = 0.0
        if 'top' in grasp.grasp_type:
            # 顶部抓取：更稳定，重心在下方
            stability_score = 0.9
            
            # 关键改进：夹持短边的抓取更稳定
            if 'short_edge' in grasp.grasp_type:
                # 夹持短边：更稳定，力臂更短
                grip_fit = 1.0 - abs(width - self.gripper_width * 0.6) / self.gripper_width
                stability_score *= max(0.7, grip_fit)
                rospy.logdebug(f"短边抓取稳定性: {stability_score:.3f}")
            elif 'long_edge' in grasp.grasp_type:
                # 夹持长边：稍不稳定
                grip_fit = 1.0 - abs(length - self.gripper_width * 0.6) / self.gripper_width
                stability_score *= max(0.5, grip_fit) * 0.85  # 降低基础分
                rospy.logdebug(f"长边抓取稳定性: {stability_score:.3f}")
            else:
                # 兼容旧命名格式
                if 'along_length' in grasp.grasp_type:
                    grip_fit = 1.0 - abs(width - self.gripper_width/2) / self.gripper_width
                else:
                    grip_fit = 1.0 - abs(length - self.gripper_width/2) / self.gripper_width
                stability_score *= max(0.5, grip_fit)
        else:
            # 侧面抓取：相对不稳定
            stability_score = 0.7
            # 高度越矮越稳定
            height_factor = max(0.5, 1.0 - height / 0.3)  # 假设30cm以上不稳定
            stability_score *= height_factor
        
        # 2. 可达性评分
        reachability_score = 0.0
        grasp_height = grasp.grasp_pose.pose.position.z
        
        # 高度评分：0.5-1.0m 最佳
        if 0.5 <= grasp_height <= 1.0:
            reachability_score = 1.0
        elif grasp_height < 0.5:
            reachability_score = max(0.3, grasp_height / 0.5)
        else:
            reachability_score = max(0.3, 1.0 - (grasp_height - 1.0) / 0.5)
        
        # 顶部抓取通常更容易到达
        if 'top' in grasp.grasp_type:
            reachability_score *= 1.1
        
        # 3. 碰撞风险评分
        collision_score = 0.8  # 基础分
        
        # 侧面抓取可能更容易碰撞
        if 'side' in grasp.grasp_type:
            collision_score *= 0.9
        
        # 4. 综合评分（加入优先级奖励）
        base_score = (self.weight_stability * stability_score +
                     self.weight_reachability * reachability_score +
                     self.weight_collision * collision_score)
        
        # 应用优先级（短边抓取优先）
        score = base_score * priority_bonus
        
        return min(1.0, max(0.0, score))


def main():
    planner = BoxGraspPlanner()
    rospy.loginfo("Box Grasp Planner is ready")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
