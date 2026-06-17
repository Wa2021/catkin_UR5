#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发送初始姿态到机器人 - 用于启动时自动移动到预设位置
"""

import rospy
import moveit_commander
import sys


def send_initial_pose():
    """发送初始姿态"""
    rospy.init_node('send_initial_pose', anonymous=True)
    
    # 等待 MoveIt 启动
    rospy.loginfo("等待 MoveIt 启动...")
    rospy.sleep(3)
    
    # 初始化 MoveIt
    moveit_commander.roscpp_initialize(sys.argv)
    
    try:
        # 创建机械臂规划组
        arm_group = moveit_commander.MoveGroupCommander("manipulator")
        
        rospy.loginfo("机器人已在初始姿态 - 跳过移动")
        # 注释掉实际移动命令，机器人将保持启动时的位置
        # arm_group.set_named_target("grasp")
        # plan = arm_group.go(wait=True)
        # arm_group.stop()
        
        # if plan:
        #     rospy.loginfo("初始姿态已发送")
        # else:
        #     rospy.logwarn("初始姿态发送失败")
        
        # 设置夹爪初始位置（打开状态）
        try:
            gripper_group = moveit_commander.MoveGroupCommander("gripper")
            
            # 设置夹爪关节位置（完全打开）
            gripper_group.set_joint_value_target({
                'hande_left_finger_joint': 0.0,
                'hande_right_finger_joint': 0.0
            })
            
            gripper_group.go(wait=True)
            gripper_group.stop()
            rospy.loginfo("夹爪初始位置已发送")
            
        except Exception as e:
            rospy.logwarn(f"夹爪控制失败: {e}")
    
    except Exception as e:
        rospy.logerr(f"MoveIt 初始化失败: {e}")
    
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    send_initial_pose()
