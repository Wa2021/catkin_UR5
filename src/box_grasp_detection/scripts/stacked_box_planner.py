#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
堆叠场景抓取规划器 - Stacked Box Grasp Planner
专门处理药盒堆叠场景的智能抓取规划
"""

import rospy
import numpy as np
from box_grasp_detection.msg import BoxGraspArray, BoxGrasp
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from threading import Lock
import copy


class StackedBoxGraspPlanner:
    """堆叠场景抓取规划器"""
    
    def __init__(self):
        rospy.init_node('stacked_box_grasp_planner', anonymous=True)
        
        self.lock = Lock()
        self.all_grasps = []
        
        # 参数
        self.z_threshold = rospy.get_param('~z_stacking_threshold', 0.05)  # 5cm认为是堆叠
        self.xy_overlap_threshold = rospy.get_param('~xy_overlap_threshold', 0.7)  # 70%重叠
        self.min_clearance = rospy.get_param('~min_clearance', 0.15)  # 15cm最小间隙
        
        # 订阅原始抓取结果
        self.grasp_sub = rospy.Subscriber(
            '/box_grasps',
            BoxGraspArray,
            self.grasp_callback,
            queue_size=1
        )
        
        # 发布优化后的抓取结果
        self.smart_grasp_pub = rospy.Publisher(
            '/box_grasps_smart',
            BoxGraspArray,
            queue_size=10
        )
        
        rospy.loginfo("=" * 70)
        rospy.loginfo("堆叠场景抓取规划器已启动")
        rospy.loginfo("=" * 70)
        rospy.loginfo("功能: 智能处理药盒堆叠场景")
        rospy.loginfo("  - 检测堆叠关系")
        rospy.loginfo("  - 优先抓取顶层药盒")
        rospy.loginfo("  - 避免碰撞其他药盒")
        rospy.loginfo("  - 重新评分排序")
        rospy.loginfo("=" * 70)
    
    def grasp_callback(self, msg):
        """接收抓取并进行智能规划"""
        if len(msg.grasps) == 0:
            return
        
        rospy.loginfo(f"\n收到 {len(msg.grasps)} 个药盒检测结果")
        
        # 分析堆叠关系
        stacking_info = self.analyze_stacking(msg.grasps)
        
        # 智能重新评分
        smart_grasps = self.smart_scoring(msg.grasps, stacking_info)
        
        # 发布结果
        smart_msg = BoxGraspArray()
        smart_msg.header = msg.header
        smart_msg.grasps = smart_grasps
        
        self.smart_grasp_pub.publish(smart_msg)
        
        # 打印结果
        self.print_stacking_analysis(smart_grasps, stacking_info)
    
    def analyze_stacking(self, grasps):
        """分析药盒堆叠关系"""
        n = len(grasps)
        stacking_info = {
            'layers': {},  # 每个盒子的层级
            'above': {},   # 每个盒子上面有什么
            'below': {},   # 每个盒子下面有什么
            'top_boxes': [],  # 顶层盒子
            'accessible': {}  # 是否可抓取
        }
        
        # 按Z坐标排序（从低到高）
        sorted_grasps = sorted(enumerate(grasps), 
                               key=lambda x: x[1].box_pose.position.z)
        
        # 初始化
        for idx, _ in sorted_grasps:
            stacking_info['above'][idx] = []
            stacking_info['below'][idx] = []
            stacking_info['layers'][idx] = 0
            stacking_info['accessible'][idx] = True
        
        # 检测堆叠关系
        for i in range(n):
            idx_i, grasp_i = sorted_grasps[i]
            z_i = grasp_i.box_pose.position.z
            
            for j in range(i + 1, n):
                idx_j, grasp_j = sorted_grasps[j]
                z_j = grasp_j.box_pose.position.z
                
                # 检查Z轴距离
                z_diff = z_j - z_i
                
                # 如果Z轴差距在合理范围内，检查XY平面重叠
                if z_diff < grasp_i.height + self.z_threshold:
                    if self.check_xy_overlap(grasp_i, grasp_j):
                        # j在i上面
                        stacking_info['above'][idx_i].append(idx_j)
                        stacking_info['below'][idx_j].append(idx_i)
        
        # 计算层级
        for idx in range(n):
            if len(stacking_info['below'][idx]) == 0:
                stacking_info['layers'][idx] = 0  # 底层
            else:
                # 递归计算层级
                max_below_layer = max([stacking_info['layers'][b] 
                                      for b in stacking_info['below'][idx]])
                stacking_info['layers'][idx] = max_below_layer + 1
        
        # 找出顶层盒子（上面没有其他盒子）
        stacking_info['top_boxes'] = [idx for idx in range(n) 
                                      if len(stacking_info['above'][idx]) == 0]
        
        # 判断可抓取性
        for idx in range(n):
            # 如果上面有盒子，不可抓取
            if len(stacking_info['above'][idx]) > 0:
                stacking_info['accessible'][idx] = False
            
            # 检查是否有足够的间隙
            grasp = grasps[idx]
            if not self.check_clearance(grasp, grasps, idx):
                stacking_info['accessible'][idx] = False
        
        return stacking_info
    
    def check_xy_overlap(self, grasp1, grasp2):
        """检查两个盒子在XY平面的重叠"""
        # 获取盒子中心
        x1 = grasp1.box_pose.position.x
        y1 = grasp1.box_pose.position.y
        x2 = grasp2.box_pose.position.x
        y2 = grasp2.box_pose.position.y
        
        # 简化：假设盒子没有旋转或旋转很小
        # 使用AABB包围盒检测
        half_l1 = grasp1.length / 2
        half_w1 = grasp1.width / 2
        half_l2 = grasp2.length / 2
        half_w2 = grasp2.width / 2
        
        # 计算重叠区域
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        overlap_x = max(0, (half_l1 + half_l2) - dx)
        overlap_y = max(0, (half_w1 + half_w2) - dy)
        
        if overlap_x > 0 and overlap_y > 0:
            overlap_area = overlap_x * overlap_y
            area1 = grasp1.length * grasp1.width
            area2 = grasp2.length * grasp2.width
            
            # 重叠面积占较小盒子面积的比例
            overlap_ratio = overlap_area / min(area1, area2)
            
            return overlap_ratio > self.xy_overlap_threshold
        
        return False
    
    def check_clearance(self, target_grasp, all_grasps, target_idx):
        """检查抓取是否有足够的间隙（避免碰撞）"""
        # 抓取位置
        grasp_pos = np.array([
            target_grasp.grasp_pose.pose.position.x,
            target_grasp.grasp_pose.pose.position.y,
            target_grasp.grasp_pose.pose.position.z
        ])
        
        # 检查与其他盒子的距离
        for idx, other_grasp in enumerate(all_grasps):
            if idx == target_idx:
                continue
            
            other_pos = np.array([
                other_grasp.box_pose.position.x,
                other_grasp.box_pose.position.y,
                other_grasp.box_pose.position.z
            ])
            
            distance = np.linalg.norm(grasp_pos - other_pos)
            
            # 如果距离太近，可能碰撞
            if distance < self.min_clearance:
                return False
        
        return True
    
    def smart_scoring(self, grasps, stacking_info):
        """智能重新评分"""
        smart_grasps = []
        
        for idx, grasp in enumerate(grasps):
            new_grasp = copy.deepcopy(grasp)
            original_score = grasp.score
            
            # 基础分数
            score = original_score
            
            # 惩罚：如果不可抓取（被遮挡）
            if not stacking_info['accessible'][idx]:
                score *= 0.1  # 大幅降低分数
                new_grasp.grasp_type += "_blocked"
            
            # 奖励：顶层盒子
            if idx in stacking_info['top_boxes']:
                score *= 1.3
                new_grasp.grasp_type += "_top"
            
            # 根据层级调整分数（越高层越优先）
            layer = stacking_info['layers'][idx]
            score *= (1.0 + 0.1 * layer)
            
            # 惩罚：底层盒子（如果上面有东西）
            if len(stacking_info['above'][idx]) > 0:
                score *= 0.3
            
            # 更新分数
            new_grasp.score = min(score, 1.0)
            smart_grasps.append(new_grasp)
        
        # 按新分数排序
        smart_grasps.sort(key=lambda x: x.score, reverse=True)
        
        return smart_grasps
    
    def print_stacking_analysis(self, grasps, stacking_info):
        """打印堆叠分析结果"""
        print("\n" + "=" * 70)
        print("📦 堆叠场景分析")
        print("=" * 70)
        
        # 按层级分组
        layers = {}
        for idx, layer in stacking_info['layers'].items():
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(idx)
        
        print(f"\n检测到 {max(layers.keys()) + 1} 层堆叠:")
        for layer in sorted(layers.keys(), reverse=True):
            print(f"\n  第 {layer + 1} 层 ({'顶层' if layer == max(layers.keys()) else '中间层' if layer > 0 else '底层'}):")
            for idx in layers[layer]:
                accessible = "✓ 可抓取" if stacking_info['accessible'][idx] else "✗ 被遮挡"
                print(f"    - 药盒 #{idx + 1}: {accessible}")
        
        print("\n" + "=" * 70)
        print("🎯 推荐抓取顺序:")
        print("=" * 70)
        
        for rank, grasp in enumerate(grasps[:5], 1):
            print(f"\n#{rank}. 评分: {grasp.score:.3f}")
            print(f"   类型: {grasp.grasp_type}")
            print(f"   位置: ({grasp.grasp_pose.pose.position.x:.3f}, "
                  f"{grasp.grasp_pose.pose.position.y:.3f}, "
                  f"{grasp.grasp_pose.pose.position.z:.3f})")
            
            if "_blocked" in grasp.grasp_type:
                print(f"   ⚠️ 警告: 此药盒被遮挡，不建议抓取")
            elif "_top" in grasp.grasp_type:
                print(f"   ✓ 推荐: 顶层药盒，优先抓取")
        
        print("\n" + "=" * 70 + "\n")
    
    def run(self):
        """运行"""
        rospy.spin()


if __name__ == '__main__':
    try:
        planner = StackedBoxGraspPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
