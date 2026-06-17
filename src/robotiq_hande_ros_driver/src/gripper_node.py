#!/usr/bin/env python3

import rospy 
from std_msgs.msg import Int32
import robotiq_gripper
from robotiq_hande_ros_driver.srv import gripper_service, gripper_serviceResponse
from sensor_msgs.msg import JointState
import threading

class HandEGripper:
    def __init__(self):
        rospy.init_node("hand_e_gripper_node", anonymous=False)
        # get the IP
        ip = rospy.get_param('~robot_ip')
        # initialize the gripper
        self.gripper = robotiq_gripper.RobotiqGripper()
        rospy.loginfo("Connecting to the gripper.....")
        self.gripper.connect(ip, 63352)
        rospy.loginfo("Activating the gripper.....")
        self.gripper.activate(auto_calibrate=False)
        
        # 发布夹爪关节状态（只发布夹爪，不订阅机械臂）
        self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        
        # 启动线程定期发布夹爪状态
        self.publish_rate = rospy.Rate(50)  # 50 Hz
        self.publishing_thread = threading.Thread(target=self.publish_gripper_state)
        self.publishing_thread.daemon = True
        self.publishing_thread.start()
        
        # set up server
        self.gripper_server = rospy.Service('gripper_service', gripper_service, self.serverCallback)
        rospy.loginfo("Gripper ready to receive service request...")
    
    def publish_gripper_state(self):
        """
        定期发布夹爪关节状态
        读取真实夹爪位置并发布，这样 RViz 可以显示实时状态
        """
        while not rospy.is_shutdown():
            try:
                js = JointState()
                js.header.stamp = rospy.Time.now()
                js.name = ['hande_left_finger_joint']
                
                # 获取夹爪真实位置 (0-255) 并转换为关节角度
                # HandE 夹爪最大开度约 50mm，对应 position=0
                # 完全闭合对应 position=255
                try:
                    gripper_pos = self.gripper.get_current_position()  # 0-255
                    # 转换为关节角度：0(打开) -> 0.0 rad, 255(闭合) -> 0.025 rad
                    joint_angle = gripper_pos * 0.0001  # 归一化到合理范围
                    js.position = [joint_angle]
                except Exception as e:
                    rospy.logwarn_throttle(10, f"Failed to get gripper position: {e}")
                    js.position = [0.0]  # 默认值
                
                js.velocity = []
                js.effort = []
                
                # 发布夹爪关节状态
                self.joint_state_pub.publish(js)
                
            except Exception as e:
                rospy.logerr_throttle(10, f"Error publishing gripper state: {e}")
            
            self.publish_rate.sleep()
    
    def serverCallback(self, request):
        pos = request.position
        speed = request.speed
        force = request.force
        if speed > 255 or speed <=0:
            return(gripper_serviceResponse('invalid speed value. Valid in range (0,255]'))
        if force > 255 or force <=0:
            return(gripper_serviceResponse('invalid force value. Valid in range (0,255]'))
        if pos > 255 or pos < 0:
            return(gripper_serviceResponse('invalid position value. Valid in range [0,255]'))

        rospy.loginfo("moving the gripper. positino = {}, speed={}, force={}".format(pos, speed, force))
        self.gripper.move_and_wait_for_pos(pos, speed, force)
        return(gripper_serviceResponse('Done'))

if __name__ == '__main__':
    gripper_obj = HandEGripper()
    rospy.spin()