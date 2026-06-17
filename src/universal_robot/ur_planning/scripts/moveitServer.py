#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 导入基本ros和moveit库
import rospy, sys
import moveit_commander#最重要的
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from moveit_msgs.msg import  PlanningScene, ObjectColor,CollisionObject, AttachedCollisionObject,Constraints,OrientationConstraint
from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import quaternion_from_euler#将四源数转换成欧拉角
from copy import deepcopy
import numpy as np
import math
from copy import deepcopy

# robotiq-85夹爪库调用，没用使用ros控制夹爪
#from real.robotiq_gripper import RobotiqGripper

# 负责接收深度学习传入的抓取位姿，并反馈抓取结果
from ur_planning.srv import grasp_pose,grasp_poseRequest,grasp_poseResponse

class MoveIt_Control:
    # 初始化程序
    def __init__(self, is_use_gripper=False):
        # Init ros config  主要是使用cpp做的，所以需要有一个接口对接到cpp
        moveit_commander.roscpp_initialize(sys.argv)

        # 初始化ROS节点
        rospy.init_node('moveit_control_server', anonymous=False)
        self.arm = moveit_commander.MoveGroupCommander('manipulator')#MoveGroupCommander最重要的库
        #两个group，一个是manipulator，管得是机械臂，一个是endeffector，管得是夹爪（Moveit！Setup Assistant中配置的）
        #如果有夹爪，会有形如下方的代码 
        #self.gripper = moveit_commander.MoveGroupCommander('gripper')#还要设置夹爪的开和关的两个姿态，与up，home平级
        self.arm.set_goal_joint_tolerance(0.001)#误差容忍范围 
        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.01)

        self.end_effector_link = self.arm.get_end_effector_link()#机械臂的link链的最后是这个，把他拿出来，后面会用到ee_link
        # 设置机械臂基座的参考系
        self.reference_frame = 'base'#base和baselink只差了一个角度问题，他们z轴重合，只要绕z轴旋转180度即可从baselink到base,根据自己的实际情况，也就算机械臂方向去选择base还是baselink，一般来说是base
        self.arm.set_pose_reference_frame(self.reference_frame)

        # 设置最大规划时间和是否允许重新规划
        self.arm.set_planning_time(5)
        self.arm.allow_replanning(True)
        self.arm.set_planner_id("RRTstar")#可以设置，在universal_robot/ur5_moveit_config/config/ompl中可以选择各种算法
        #在rviz和Moveit！Setup Assistant可以设置

        # 设置允许的最大速度和加速度（范围：0~1）比例因子
        self.arm.set_max_acceleration_scaling_factor(1)#在universal_robot/ur5_moveit_config/config/joint中可以查看速度设置
        self.arm.set_max_velocity_scaling_factor(1)

        # 机械臂初始姿态
        self.go_home()

        # 如果使用robotiq-85夹爪，会初始化夹爪，默认不使用
        self.is_use_gripper = is_use_gripper
        if is_use_gripper:
          self.gripper = RobotiqGripper()
          self.gripper.connect("192.168.50.100", 63352)
          self.gripper._reset()
          rospy.loginfo("Activating gripper...")
          self.gripper.activate()
          rospy.sleep(1.5)
        # 发布场景
        self.set_scene()  # set table
        #self.arm.set_workspace([-2,-2,0,2,2,2])  #[minx miny minz maxx maxy maxz]
        # 测试专用
        self.testRobot()
        # 抓取服务端，负责接收抓取位姿并执行运动
        # server = rospy.Service("moveit_grasp",grasp_pose,self.grasp_callback )
        # rospy.spin()

    def close(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    # 测试程序用
    def testRobot(self):
        try:
            print("Test for robot...")
            
            # --- 1. 关节空间规划 (move_j) ---
            # 作用：快速移动到大概位置，不考虑末端路径
            rospy.loginfo("Step 1: Joint Move (move_j) to Ready Position")
            # 这是一个比较典型的等待位姿，手臂弯曲在前方
            # 关节角度: [base, shoulder, elbow, wrist1, wrist2, wrist3]
            ready_joints = [0, -1.57, 1.57, -1.57, -1.57, 0] 
            self.move_j(ready_joints)
            rospy.sleep(1)

            
            # --- 2. 笛卡尔空间点到点规划 (move_p) ---
            # 作用：移动到指定的空间坐标，路径可能是曲线，但保证终点准确
            rospy.loginfo("Step 2: Pose Move (move_p) to Start Point of Rectangle")
            # 定义朝下的姿态 (RX, RY, RZ)
            down_orientation = [-np.pi, 0, 0]
            # 目标位置: x=0.4, y=-0.2, z=0.4 (矩形左上角)
            p1 = [0.4, -0.2, 0.4] + down_orientation
            self.move_p(p1)
            rospy.sleep(1)


            # --- 3. 笛卡尔空间直线规划 (move_l) ---
            # 作用：走直线，适合焊接、涂胶、画图等需要精确路径的动作
            rospy.loginfo("Step 3: Linear Move (move_l) - Drawing a Rectangle")
            
            # P2: 右上 (y 变大)
            p2 = [0.4, 0.2, 0.4] + down_orientation
            self.move_l(p2)
            
            # P3: 右下 (x 变小, 拉近)
            p3 = [0.2, 0.2, 0.4] + down_orientation
            self.move_l(p3)
            
            # P4: 左下 (y 变回 -0.2)
            p4 = [0.2, -0.2, 0.4] + down_orientation
            self.move_l(p4)
            
            # 回到 P1 (闭合矩形)
            self.move_l(p1)
            rospy.sleep(1)
            
            # 结束演示，回到 Home
            rospy.loginfo("Test Finished! Going Home...")
            self.go_home()
            
        except Exception as e:
            print("Test fail! Error: ", e)
            import traceback
            traceback.print_exc()

    # 在机械臂下方添加一个table，使得机械臂只能够在上半空间进行规划和运动
    # 避免碰撞到下方的桌子等其他物体
    #设置了一个场景，然后发布之后，机械臂就知道这里有一个障碍物
    def set_scene(self):
        ## set table
        self.scene = PlanningSceneInterface()
        self.scene_pub = rospy.Publisher('planning_scene', PlanningScene, queue_size=5)
        self.colors = dict()
        rospy.sleep(1)
        table_id = 'table'
        self.scene.remove_world_object(table_id)
        rospy.sleep(1)
        table_size = [2, 2, 0.01]
        table_pose = PoseStamped()
        table_pose.header.frame_id = self.reference_frame
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -table_size[2]/2 -0.02#这边得改阿,这个z轴可以到达-0.02m
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box(table_id, table_pose, table_size)
        self.setColor(table_id, 0.5, 0.5, 0.5, 1.0)#调板子（table）的颜色RGB/透明度
        self.sendColors()

    # 关节规划，输入6个关节角度（单位：弧度）
    def move_j(self, joint_configuration=None,a=1,v=1):
        # 设置机械臂的目标位置，使用六轴的位置数据进行描述（单位：弧度）
        if joint_configuration==None:
            joint_configuration = [0, -1.5707, 0, -1.5707, 0, 0]#以弧度为单位
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)
        self.arm.set_joint_value_target(joint_configuration)
        rospy.loginfo("move_j:"+str(joint_configuration))
        self.arm.go()
        rospy.sleep(1)

    # 空间规划，输入xyzRPY
    def move_p(self, tool_configuration=None,a=1,v=1):
        if tool_configuration==None:
            tool_configuration = [0.3,0,0.3,0,-np.pi/2,0]#以弧度为单位，需要修改
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = tool_configuration[0]
        target_pose.pose.position.y = tool_configuration[1]
        target_pose.pose.position.z = tool_configuration[2]
        #将rpy转换成四元数
        q = quaternion_from_euler(tool_configuration[3],tool_configuration[4],tool_configuration[5])
        target_pose.pose.orientation.x = q[0]
        target_pose.pose.orientation.y = q[1]
        target_pose.pose.orientation.z = q[2]
        target_pose.pose.orientation.w = q[3]

        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        rospy.loginfo("move_p:" + str(tool_configuration))
        plan = self.arm.plan()
        
        # ROS Noetic plan() returns a tuple (success, plan, planning_time, error_code)
        # ROS Melodic plan() returns just the plan
        if isinstance(plan, tuple):
            traj = plan[1]
        else:
            traj = plan
            
        self.arm.execute(traj)
        rospy.sleep(1)

    # 空间直线运动，输入(x,y,z,R,P,Y,x2,y2,z2,R2,...)waypoints_number要和输入的xyzrpy组数相同
    # 默认仅执行一个点位，可以选择传入多个点位
    def move_l(self, tool_configuration,waypoints_number=1,a=0.5,v=0.5):
        if tool_configuration==None:
            tool_configuration = [0.3,0,0.3,0,-np.pi/2,0]
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)

        # 设置路点
        waypoints = []
        for i in range(waypoints_number):
            target_pose = PoseStamped()
            target_pose.header.frame_id = self.reference_frame
            target_pose.header.stamp = rospy.Time.now()
            target_pose.pose.position.x = tool_configuration[6*i+0]
            target_pose.pose.position.y = tool_configuration[6*i+1]
            target_pose.pose.position.z = tool_configuration[6*i+2]
            q = quaternion_from_euler(tool_configuration[6*i+3],tool_configuration[6*i+4],tool_configuration[6*i+5])
            target_pose.pose.orientation.x = q[0]
            target_pose.pose.orientation.y = q[1]
            target_pose.pose.orientation.z = q[2]
            target_pose.pose.orientation.w = q[3]
            waypoints.append(target_pose.pose)
        rospy.loginfo("move_l:" + str(tool_configuration))
        self.arm.set_start_state_to_current_state()
        fraction = 0.0  # 路径规划覆盖率
        maxtries = 100  # 最大尝试规划次数
        attempts = 0  # 已经尝试规划次数

        # 设置机器臂当前的状态作为运动初始状态
        self.arm.set_start_state_to_current_state()

        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
        while fraction < 1.0 and attempts < maxtries:
            # 根据报错信息 Boost.Python signature: (list, double, bool)
            # 这意味着 jump_threshold 参数已经被移除了，只需要传入 waypoints, eef_step, avoid_collisions
            (plan, fraction) = self.arm.compute_cartesian_path(
                waypoints,  # waypoint poses
                0.001,      # eef_step
                True)       # avoid_collisions (bool)
            attempts += 1
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            self.arm.execute(plan)
            rospy.loginfo("Path execution complete.")
        else:
            rospy.loginfo(
                "Path planning failed with only " + str(fraction) +
                " success after " + str(maxtries) + " attempts.")
        rospy.sleep(1)

    def move_c(self,pose_via,tool_configuration,k_acc=1,k_vel=1,r=0,mode=0):
        pass

    def go_home(self,a=1,v=1):
        self.arm.set_max_acceleration_scaling_factor(a)
        self.arm.set_max_velocity_scaling_factor(v)
        # “up”为自定义姿态，你可以使用“home”或者其他姿态
        self.arm.set_named_target('up')#在universal_robot/ur5_moveit_config/config/ur5.stdf中可以看到配置好的位置，是在#Moveit！Setup Assistant工具中生成的
        self.arm.go()
        rospy.sleep(1)

    def setColor(self, name, r, g, b, a=0.9):
        # 初始化moveit颜色对象
        color = ObjectColor()
        # 设置颜色值
        color.id = name
        color.color.r = r
        color.color.g = g
        color.color.b = b
        color.color.a = a
        # 更新颜色字典
        self.colors[name] = color

    # 将颜色设置发送并应用到moveit场景当中
    def sendColors(self):
        # 初始化规划场景对象
        p = PlanningScene()
        # 需要设置规划场景是否有差异
        p.is_diff = True
        # 从颜色字典中取出颜色设置
        for color in self.colors.values():
            p.object_colors.append(color)
        # 发布场景物体颜色设置
        self.scene_pub.publish(p)
    
    ## robotiq85夹爪相关的函数，可以不使用，报错可以删掉
    ## ----------------------------------------------------
    def log_gripper_info(self):
        rospy.loginfo("Pos: "+str(self.gripper.get_current_position()   ))
    def close_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(255, speed, force)
        # self.log_gripper_info()
        # print("gripper had closed!")
        rospy.sleep(1.2)
    def open_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(0, speed, force)
        # self.log_gripper_info()
        # print("gripper had opened!")
        rospy.sleep(1.2)
    def check_grasp(self):
        # if the robot grasp object ,then the gripper is not open
        return self.get_current_tool_pos()>5
    def get_current_tool_pos(self):
        return self.gripper.get_current_position() 
    ## ----------------------------------------------------

    def some_useful_function_you_may_use(self):
        # return the robot current pose 获取当前的xyz和角度
        current_pose = self.arm.get_current_pose()
        # rospy.loginfo('current_pose:',current_pose)
        # return the robot current joints获取当前六个关节的值
        current_joints = self.arm.get_current_joint_values()
        # rospy.loginfo('current_joints:',current_joints)

        #self.arm.set_planner_id("RRTConnect")设置算法
        self.arm.set_planner_id("TRRT")
        plannerId = self.arm.get_planner_id()
        rospy.loginfo(plannerId)

        planning_frame = self.arm.get_planning_frame()
        rospy.loginfo(planning_frame)

        # stop the robot
        self.arm.stop()
  

    # 最重要的函数，与深度学习对接用的
    # 接收grasp_pose（xyzRPY） ,执行相应的动作
    def grasp_callback(self,req):

        result =True
        # 执行动作
        self.move_p([req.grasppose_x,req.grasppose_y,req.grasppose_z,req.grasppose_R,req.grasppose_P,req.grasppose_Y])
        # 本人仅使用了move_p,其他动作并未使用moveit执行
        # 你可以改写grasp函数，然后调用grasp函数如下:
        # result =  self.grasp([req.grasppose_x,req.grasppose_y,req.grasppose_z],[req.grasppose_R,req.grasppose_P,req.grasppose_Y],is_use_gripper=self.is_use_gripper)
        # 设置返回值，反馈执行动作是否成功
        response = grasp_poseResponse()
        response.success = bool(result)
        return response

    #将之前的函数进行一个组合，实现一个小动作
    # 你可以改写此函数，主要更改动作步骤和夹爪控制相关程序
    # 函数参数分别为：位置，姿态，是否使用robotiq85，夹爪打开的大小，机械臂运行的最大加速度、速度，夹爪的速度、力
    def grasp(self, position, rpy=None,open_size=0.65, k_acc=0.8, k_vel=0.8, speed=255, force=125):

        rospy.loginfo('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # 准备工作
        grasp_home = [15.96 / 180 * np.pi,-78.97/ 180 * np.pi,-82.33/ 180 * np.pi,
-108.58/ 180 * np.pi,89.90/ 180 * np.pi,-74.08/ 180 * np.pi]  # you can change me
        self.move_j(grasp_home, k_acc, k_vel)
        if self.is_use_gripper:
            open_pos = int(-300 * open_size + 255)  # open size:0~0.85cm --> open pos:255~0
            self.gripper.move_and_wait_for_pos(open_pos, speed, force)
            self.log_gripper_info()

        # 首先，到达预抓取位置，即抓取位置上方10cm
        pre_position = deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        self.move_p(pre_position + rpy, k_acc, k_vel)

        # 第二，达到抓取位姿
        self.move_l(position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        # 关闭夹爪
        if self.is_use_gripper:
            self.close_gripper(speed, force)
        # 抓取完成后回到预抓取姿态
        self.move_l(pre_position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        # 检查抓取是否成功，失败则回到机械臂初始位姿
        if (self.is_use_gripper and not self.check_grasp()):
            print("Check grasp fail! ")
            self.move_p(grasp_home)
            return False
        # 第三，将物体放入指定区域
        box_position = [0.6, 0, 0.35, -np.pi, 0, 0]  # you can change me!
        self.move_p(box_position, k_acc, k_vel)
        box_position[2] = 0.15  # 在桌面上方15cm出打开夹爪
        self.move_l(box_position, k_acc, k_vel)
        if self.is_use_gripper:
            self.open_gripper(speed, force)
        box_position[2] = 0.35
        self.move_l(box_position, k_acc, k_vel)
        self.move_j(grasp_home)
        print("grasp success!")


if __name__ =="__main__":
    moveit_server = MoveIt_Control(is_use_gripper=False)


