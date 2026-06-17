#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云结构分析脚本
分析RealSense点云的实际结构和特性
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class PointCloudAnalyzer:
    def __init__(self):
        rospy.init_node('pointcloud_analyzer')
        
        # 订阅点云
        rospy.Subscriber('/camera/depth/color/points_relay', PointCloud2, self.analyze_pointcloud)
        
        rospy.loginfo("点云结构分析器启动，等待点云数据...")
        self.analyzed = False
    
    def analyze_pointcloud(self, msg):
        if self.analyzed:
            return
        
        print("\n" + "="*60)
        print("📊 点云结构分析报告")
        print("="*60)
        
        print(f"基本信息:")
        print(f"  宽度: {msg.width}")
        print(f"  高度: {msg.height}")
        print(f"  点数: {msg.width * msg.height}")
        print(f"  是否有序: {'是' if msg.width > 1 and msg.height > 1 else '否'}")
        print(f"  数据大小: {len(msg.data)} 字节")
        print(f"  点步长: {msg.point_step}")
        print(f"  行步长: {msg.row_step}")
        
        print(f"\n字段信息:")
        for field in msg.fields:
            print(f"  {field.name}: 偏移={field.offset}, 类型={field.datatype}, 计数={field.count}")
        
        # 读取一些实际点来验证
        print(f"\n实际点云采样 (前10个有效点):")
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        total_points = len(points)
        print(f"  总有效点数: {total_points}")
        
        for i, point in enumerate(points[:10]):
            # 如果是有序点云，计算对应的像素位置
            if msg.width > 1 and msg.height > 1:
                # 这里我们假设读取的点保持了原始顺序
                # 但实际上skip_nans可能会打乱顺序
                pixel_u = i % msg.width
                pixel_v = i // msg.width
                print(f"  点{i}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) -> 像素({pixel_u}, {pixel_v})")
            else:
                print(f"  点{i}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
        
        # 检查无效点的比例
        all_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False))
        invalid_points = len(all_points) - total_points
        print(f"\n数据完整性:")
        print(f"  理论点数: {msg.width * msg.height}")
        print(f"  实际读取: {len(all_points)}")
        print(f"  有效点数: {total_points}")
        print(f"  无效点数: {invalid_points}")
        print(f"  有效率: {total_points/len(all_points)*100:.1f}%")
        
        # 验证有序性
        if msg.width > 1 and msg.height > 1:
            print(f"\n有序性验证:")
            # 尝试读取特定位置的点
            try:
                center_u = msg.width // 2
                center_v = msg.height // 2
                center_points = list(pc2.read_points(
                    msg, field_names=("x", "y", "z"), 
                    skip_nans=True, uvs=[(center_u, center_v)]
                ))
                if center_points:
                    print(f"  中心点({center_u}, {center_v}): {center_points[0]}")
                else:
                    print(f"  中心点({center_u}, {center_v}): 无效")
                
                # 尝试读取几个边界框内的点
                bbox = [100, 100, 200, 200]  # 示例边界框
                bbox_points = []
                for v in range(bbox[1], min(bbox[3], msg.height)):
                    for u in range(bbox[0], min(bbox[2], msg.width)):
                        points_at_uv = list(pc2.read_points(
                            msg, field_names=("x", "y", "z"), 
                            skip_nans=True, uvs=[(u, v)]
                        ))
                        if points_at_uv:
                            bbox_points.append(points_at_uv[0])
                
                print(f"  边界框{bbox}内有效点数: {len(bbox_points)}")
                
            except Exception as e:
                print(f"  有序性验证失败: {e}")
        
        print("="*60)
        self.analyzed = True
        rospy.signal_shutdown("分析完成")

if __name__ == '__main__':
    try:
        analyzer = PointCloudAnalyzer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass