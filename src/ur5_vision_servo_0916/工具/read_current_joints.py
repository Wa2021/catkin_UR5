#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取并显示当前机械臂关节角度
"""
import rospy
from sensor_msgs.msg import JointState

def joint_callback(msg):
    # 找到UR5的6个关节
    ur_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    print("\n" + "="*60)
    print("当前关节角度:")
    print("="*60)
    
    rad_values = []
    deg_values = []
    
    for joint in ur_joints:
        if joint in msg.name:
            idx = msg.name.index(joint)
            rad = msg.position[idx]
            deg = rad * 180.0 / 3.14159265359
            rad_values.append(rad)
            deg_values.append(deg)
            print(f"{joint:25s}: {rad:8.6f} rad  ({deg:7.2f}°)")
    
    print("\n弧度列表 (可直接复制到SRDF):")
    print([round(v, 6) for v in rad_values])
    print("\n角度列表:")
    print([round(v, 2) for v in deg_values])
    print("="*60 + "\n")
    
    rospy.signal_shutdown("读取完成")

if __name__ == '__main__':
    try:
        rospy.init_node('read_joints', anonymous=True)
        print("等待关节状态...")
        rospy.Subscriber('/joint_states', JointState, joint_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
