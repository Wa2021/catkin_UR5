#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HandE 夹爪控制脚本
直接修改下面的参数值，然后运行脚本即可控制夹爪

使用方法：
==========

方式 1：只控制夹爪（推荐）
---------------------------
1. 启动夹爪驱动：
   roslaunch ur_planning gripper_only.launch robot_ip:=192.168.0.1

2. 修改下面 gripper() 函数括号中的参数

3. 运行此脚本：
   test_gripper
   或
   python3 test_gripper.py


方式 2：启动完整机器人系统（包含机械臂+夹爪）
----------------------------------------------
1. 启动完整系统：
   roslaunch ur_planning start_robot_only.launch robot_ip:=192.168.0.1

2. 修改下面 gripper() 函数括号中的参数

3. 运行此脚本：
   test_gripper


参数说明：
==========
- position: 0-255 (0=完全打开, 255=完全闭合)
- speed:    1-255 (速度，数值越大越快)
- force:    1-255 (力度，数值越大力度越大，建议50-150)

终端命令对照：
==============
# 打开夹爪
rosservice call /gripper_service "position: 0
speed: 255
force: 100"

# 闭合夹爪
rosservice call /gripper_service "position: 255
speed: 255
force: 100"

# 半开状态
rosservice call /gripper_service "position: 128
speed: 200
force: 100"
"""

import rospy
from robotiq_hande_ros_driver.srv import gripper_service

# ============================================================
# 修改这里的参数来控制夹爪位置
# ============================================================
def gripper(position=0, speed=255, force=100):
    """
    控制夹爪函数
    修改括号中的参数：
    - position: 0-255 (0=打开, 255=闭合)
    - speed:    1-255 (速度)
    - force:    1-255 (力度)
    """
    rospy.init_node('test_gripper', anonymous=True)
    
    # 等待夹爪服务
    print("等待夹爪服务...")
    rospy.wait_for_service('/gripper_service', timeout=10.0)
    gripper_srv = rospy.ServiceProxy('/gripper_service', gripper_service)
    print("✅ 夹爪服务已连接")
    
    # 发送控制指令
    print(f"控制夹爪: position={position}, speed={speed}, force={force}")
    response = gripper_srv(position=position, speed=speed, force=force)
    print(f"响应: {response.response}")
    print("✅ 完成")


if __name__ == '__main__':
    try:
        # ============================================================
        # 在这里调用 gripper() 并修改参数，类似 test_gohome.py
        # ============================================================
        
        # 示例1：打开夹爪
        #gripper(position=0, speed=255, force=100)
        
        # 示例2：闭合夹爪（注释掉上面的，取消注释下面这行）
        gripper(position=255, speed=255, force=100)
        
        # 示例3：半开状态
        # gripper(position=128, speed=200, force=100)
        
        # 示例4：轻柔抓取
        # gripper(position=200, speed=100, force=50)
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"❌ 错误: {e}")
