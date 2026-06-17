#!/usr/bin/env python3
"""
测试 MoveIt Servo 的简单脚本
发送带正确时间戳的 TwistStamped 命令
"""

import rospy
from geometry_msgs.msg import TwistStamped

def main():
    rospy.init_node('servo_test_publisher', anonymous=True)
    
    pub = rospy.Publisher('/servo_server/delta_twist_cmds', TwistStamped, queue_size=10)
    
    rate = rospy.Rate(30)  # 30 Hz
    
    rospy.loginfo("开始发送伺服命令... 按 Ctrl+C 停止")
    rospy.loginfo("机械臂将沿 Z 轴（垂直向上）以 0.02 m/s 移动")
    
    while not rospy.is_shutdown():
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()  # 关键：使用当前时间戳
        cmd.header.frame_id = "base_link"  # 改用 base_link 坐标系
        
        # 沿 Z 轴（垂直）移动 0.02 m/s - 更安全的方向
        cmd.twist.linear.x = 0.0
        cmd.twist.linear.y = 0.0
        cmd.twist.linear.z = 0.02  # 向上移动
        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = 0.0
        
        pub.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
