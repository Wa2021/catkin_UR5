#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置 joint_state_publisher_gui 的初始位置
"""

import rospy
import yaml
import os


def set_initial_joints():
    """从配置文件读取并设置初始关节位置"""
    rospy.init_node('set_initial_joints', anonymous=True)
    
    # 加载初始姿态配置
    config_path = "/home/xsh/catkin_UR5/src/universal_robot/ur_description/config/initial_pose.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        initial_pose = config['initial_pose']
        
        # 设置 ROS 参数
        for joint_name, angle in initial_pose.items():
            param_name = f"/joint_state_publisher/zeros/{joint_name}"
            rospy.set_param(param_name, float(angle))
            rospy.loginfo(f"Set {joint_name}: {angle}")
            
        rospy.loginfo("初始关节位置已设置")
        
    except Exception as e:
        rospy.logerr(f"加载配置文件失败: {e}")


if __name__ == '__main__':
    set_initial_joints()
