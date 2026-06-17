#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <tf2/LinearMath/Quaternion.h>
#include <vector>
#include <string>
#include <iostream>

using namespace std;


class MoveIt_Control
{
public:
	//传进去的三个参数：nodehandle,ros大管家；
	//后面两个是一起的，机械臂moveit配置中的group，分为机械臂和夹爪
	MoveIt_Control(const ros::NodeHandle &nh,moveit::planning_interface::MoveGroupInterface &arm,const string &PLANNING_GROUP) {
		
		this->arm_ = &arm;
		this->nh_ =nh;
		

		//误差容忍度
		arm_->setGoalPositionTolerance(0.001);
		arm_->setGoalOrientationTolerance(0.01);
		arm_->setGoalJointTolerance(0.001);
		//速度系数0-1
		arm_->setMaxAccelerationScalingFactor(0.5);
		arm_->setMaxVelocityScalingFactor(0.5);

		const moveit::core::JointModelGroup* joint_model_group =
			arm_->getCurrentState()->getJointModelGroup(PLANNING_GROUP);

		this->end_effector_link = arm_->getEndEffectorLink();

		//参考坐标系
		this->reference_frame = "base";
		arm_->setPoseReferenceFrame(reference_frame);
		
		//规划器相关
		arm_->allowReplanning(true);
		arm_->setPlanningTime(5.0);
		arm_->setPlannerId("TRRTkConfigDefault");

				go_home();

		
		create_table();//搭建一个桌子的环境出来

	}	
	void go_home() {
		// moveit::planning_interface::MoveGroupInterface arm("manipulator");
		arm_->setNamedTarget("up");
		arm_->move();
		sleep(0.5);
	}

	bool move_j(const vector<double> &joint_group_positions) {
		// moveit::planning_interface::MoveGroupInterface arm("manipulator");
		arm_->setJointValueTarget(joint_group_positions);
		arm_->move();
		sleep(0.5);
		return true;
	}

	bool move_p(const vector<double> &pose) {
		// moveit::planning_interface::MoveGroupInterface arm("manipulator");
		// const std::string reference_frame = "base";
		// arm.setPoseReferenceFrame(reference_frame);

		
		geometry_msgs::Pose target_pose;
		target_pose.position.x = pose[0];
		target_pose.position.y = pose[1];
		target_pose.position.z = pose[2];

		//将姿态从rpy转换成四元数，moveit用的就是四元数表示姿态
		tf2::Quaternion myQuaternion;
		myQuaternion.setRPY(pose[3], pose[4], pose[5]);
		target_pose.orientation.x = myQuaternion.getX();
		target_pose.orientation.y = myQuaternion.getY();
		target_pose.orientation.z = myQuaternion.getZ();
		target_pose.orientation.w = myQuaternion.getW();

		
		arm_->setStartStateToCurrentState();
		arm_->setPoseTarget(target_pose);
		
		// 为避障演示增加规划时间和尝试次数
		arm_->setPlanningTime(10.0); // 增加规划时间
		arm_->setNumPlanningAttempts(10); // 设置多次尝试

		//规划
		moveit::planning_interface::MoveGroupInterface::Plan plan;
		moveit::planning_interface::MoveItErrorCode success = arm_->plan(plan);

		ROS_INFO("move_p:%s", success ? "SUCCESS" : "FAILED");
		
		// 恢复默认规划时间
		arm_->setPlanningTime(5.0);

		
		if (success) {
			arm_->execute(plan);
			sleep(1);
			return true;
		}
		return false;
	}

	//限制规划的路径，比如说去抓一瓶水，要求机械臂末端姿态不变（这个函数就是实现的这个）
	bool move_p_with_constrains(const vector<double>& pose) {

		// moveit::planning_interface::MoveGroupInterface arm("manipulator");
		// const std::string reference_frame = "base";
		// arm.setPoseReferenceFrame(reference_frame);
		// arm.setPlannerId("TRRT");
		arm_->setMaxAccelerationScalingFactor(0.5);
		arm_->setMaxVelocityScalingFactor(0.5);

				geometry_msgs::Pose target_pose;
		target_pose.position.x = pose[0];
		target_pose.position.y = pose[1];
		target_pose.position.z = pose[2];

		tf2::Quaternion myQuaternion;
		myQuaternion.setRPY(pose[3], pose[4], pose[5]);
		target_pose.orientation.x = myQuaternion.getX();
		target_pose.orientation.y = myQuaternion.getY();
		target_pose.orientation.z = myQuaternion.getZ();
		target_pose.orientation.w = myQuaternion.getW();

		// geometry_msgs::PoseStamped current_pose_modified = arm.getCurrentPose(this->end_effector_link);
		// current_pose_modified.header.frame_id = "base";
		// current_pose_modified.pose.position.x = -current_pose_modified.pose.position.x ;
		// current_pose_modified.pose.position.y = -current_pose_modified.pose.position.y ;
		// current_pose_modified.pose.orientation.x = myQuaternion.getX();
		// current_pose_modified.pose.orientation.y = myQuaternion.getY();
		// current_pose_modified.pose.orientation.z = myQuaternion.getZ();
		// current_pose_modified.pose.orientation.w = myQuaternion.getW();
		// arm.setPoseTarget(current_pose_modified.pose);arm.move();
		

		//set constraint 
		moveit_msgs::OrientationConstraint ocm;
		ocm.link_name = "ee_link";
		ocm.header.frame_id = "base";
		ocm.orientation.x = myQuaternion.getX();
		ocm.orientation.y = myQuaternion.getY();
		ocm.orientation.z = myQuaternion.getZ();
		ocm.orientation.w = myQuaternion.getW();
		ocm.absolute_x_axis_tolerance = 0.1;//误差容忍
		ocm.absolute_y_axis_tolerance = 0.1;
		ocm.absolute_z_axis_tolerance = 0.1;
		ocm.weight = 1.0;

		// Now, set it as the path constraint for the group.
		moveit_msgs::Constraints test_constraints;
		test_constraints.orientation_constraints.push_back(ocm);
		arm_->setPathConstraints(test_constraints);//将限制加进去

		/*moveit::core::RobotState start_state(*move_group_interface.getCurrentState());
		geometry_msgs::Pose start_pose2;
		start_pose2.orientation.w = 1.0;
		start_pose2.position.x = 0.55;
		start_pose2.position.y = -0.05;
		start_pose2.position.z = 0.8;
		start_state.setFromIK(joint_model_group, start_pose2);
		move_group_interface.setStartState(start_state);*/

		// Now we will plan to the earlier pose target from the new
		// start state that we have just created.
		arm_->setStartStateToCurrentState();
		arm_->setPoseTarget(target_pose);

		// Planning with constraints can be slow because every sample must call an inverse kinematics solver.
		// Lets increase the planning time from the default 5 seconds to be sure the planner has enough time to succeed.
		arm_->setPlanningTime(10.0);//规划时间拉长一点

		moveit::planning_interface::MoveGroupInterface::Plan plan;
		moveit::planning_interface::MoveItErrorCode success = arm_->plan(plan);

		ROS_INFO("move_p_with_constrains :%s", success ? "SUCCESS" : "FAILED");

		arm_->setPlanningTime(5.0);//规划好之后，将时间变回去
		arm_->clearPathConstraints();//清除掉之前的限制
		if (success) {
			arm_->execute(plan);
			sleep(1);
			return true;
		}
		return false;
	}

	//两个版本，c++中可以有重载，这个版本传的是xyzrpy
	bool move_l(const vector<double>& pose) {
		// moveit::planning_interface::MoveGroupInterface arm("manipulator");

		vector<geometry_msgs::Pose> waypoints;
		geometry_msgs::Pose target_pose;
		target_pose.position.x = pose[0];
		target_pose.position.y = pose[1];
		target_pose.position.z = pose[2];

		
		tf2::Quaternion myQuaternion;
		myQuaternion.setRPY(pose[3], pose[4], pose[5]);
		target_pose.orientation.x = myQuaternion.getX();
		target_pose.orientation.y = myQuaternion.getY();
		target_pose.orientation.z = myQuaternion.getZ();
		target_pose.orientation.w = myQuaternion.getW();
		waypoints.push_back(target_pose);

		
		moveit_msgs::RobotTrajectory trajectory;
		const double jump_threshold = 0.0;
		const double eef_step = 0.01;
		double fraction = 0.0;
		int maxtries = 100;   
		int attempts = 0;     

		while (fraction < 1.0 && attempts < maxtries)
		{
			fraction = arm_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
			attempts++;
		}

		if (fraction == 1)
		{
			ROS_INFO("Path computed successfully. Moving the arm.");

			
			moveit::planning_interface::MoveGroupInterface::Plan plan;
			plan.trajectory_ = trajectory;

			
			arm_->execute(plan);
			sleep(1);
			return true;
		}
		else
		{
			ROS_INFO("Path planning failed with only %0.6f success after %d attempts.", fraction, maxtries);
			return false;
		}
	}
	//可以一次性传入多个点
	bool move_l(const vector<vector<double>>& posees) {
		// moveit::planning_interface::MoveGroupInterface arm("manipulator");
		vector<geometry_msgs::Pose> waypoints;
		for (int i = 0; i < posees.size(); i++) {
            geometry_msgs::Pose target_pose;
			target_pose.position.x = posees[i][0];
			target_pose.position.y = posees[i][1];
			target_pose.position.z = posees[i][2];

			
			tf2::Quaternion myQuaternion;
			myQuaternion.setRPY(posees[i][3], posees[i][4], posees[i][5]);
			target_pose.orientation.x = myQuaternion.getX();
			target_pose.orientation.y = myQuaternion.getY();
			target_pose.orientation.z = myQuaternion.getZ();
			target_pose.orientation.w = myQuaternion.getW();
			waypoints.push_back(target_pose);
		}

		
		moveit_msgs::RobotTrajectory trajectory;
		const double jump_threshold = 0.0;
		const double eef_step = 0.01;
		double fraction = 0.0;
		int maxtries = 100;   
		int attempts = 0;     

		while (fraction < 1.0 && attempts < maxtries)
		{
			fraction = arm_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
			attempts++;
		}

		if (fraction == 1)
		{
			ROS_INFO("Path computed successfully. Moving the arm.");

			
			moveit::planning_interface::MoveGroupInterface::Plan plan;
			plan.trajectory_ = trajectory;

			
			arm_->execute(plan);
			sleep(1);
			return true;
		}
		else
		{
			ROS_INFO("Path planning failed with only %0.6f success after %d attempts.", fraction, maxtries);
			return false;
		}
	}


	void create_table() {
		
		// Now let's define a collision object ROS message for the robot to avoid.

		ros::Publisher planning_scene_diff_publisher = nh_.advertise<moveit_msgs::PlanningScene>("planning_scene", 1);
    	ros::WallDuration sleep_t(0.5);
    	while (planning_scene_diff_publisher.getNumSubscribers() < 1)
    	{
     	 sleep_t.sleep();
    	}
    	ROS_INFO("Planning scene publisher ready with %d subscribers", planning_scene_diff_publisher.getNumSubscribers());
		moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
		moveit_msgs::PlanningScene planning_scene;
		
		// 创建桌子
		moveit_msgs::CollisionObject collision_object;
		collision_object.header.frame_id = arm_->getPlanningFrame();
		collision_object.id = "table";

		// 定义了一个桌子并加到世界中，可以根据自己的实际情况去改
		shape_msgs::SolidPrimitive primitive;
		primitive.type = primitive.BOX;
		primitive.dimensions.resize(3);
		primitive.dimensions[primitive.BOX_X] = 2;//长
		primitive.dimensions[primitive.BOX_Y] = 2;//宽
		primitive.dimensions[primitive.BOX_Z] = 0.01;//高

		// 定义桌子的位姿 (specified relative to frame_id)
		geometry_msgs::Pose box_pose;
		box_pose.orientation.w = 1.0;
		box_pose.position.x = 0.0;
		box_pose.position.y = 0.0;
		box_pose.position.z = -0.01/2 -0.02;

		collision_object.primitives.push_back(primitive);
		collision_object.primitive_poses.push_back(box_pose);
		collision_object.operation = collision_object.ADD;
		planning_scene.world.collision_objects.push_back(collision_object);

		// 创建障碍物 - 圆柱体障碍物，用于展示避障功能
		moveit_msgs::CollisionObject obstacle;
		obstacle.header.frame_id = arm_->getPlanningFrame();
		obstacle.id = "cylinder_obstacle";

		shape_msgs::SolidPrimitive cylinder;
		cylinder.type = cylinder.CYLINDER;
		cylinder.dimensions.resize(2);
		cylinder.dimensions[cylinder.CYLINDER_HEIGHT] = 0.6; // 高度60cm，增加高度
		cylinder.dimensions[cylinder.CYLINDER_RADIUS] = 0.1; // 半径10cm (增大半径以便更容易看到)

		// 将障碍物放置在机械臂前方，会阻挡直线路径
		geometry_msgs::Pose cylinder_pose;
		cylinder_pose.orientation.w = 1.0;
		cylinder_pose.position.x = 0.4;   // 前方40cm，更靠近工作区域
		cylinder_pose.position.y = 0.4;   // 正中间
		cylinder_pose.position.z = 0.25;  // 底部高度25cm，稍微抬高

		obstacle.primitives.push_back(cylinder);
		obstacle.primitive_poses.push_back(cylinder_pose);
		obstacle.operation = obstacle.ADD;
		planning_scene.world.collision_objects.push_back(obstacle);
		
		ROS_INFO("Cylinder obstacle created: radius=%.2f, height=%.2f, position=(%.2f, %.2f, %.2f)", 
		         cylinder.dimensions[cylinder.CYLINDER_RADIUS], 
		         cylinder.dimensions[cylinder.CYLINDER_HEIGHT],
		         cylinder_pose.position.x, cylinder_pose.position.y, cylinder_pose.position.z);

    	planning_scene.is_diff = true;
    	planning_scene_diff_publisher.publish(planning_scene);
    	
    	// 等待一下确保场景被正确发布
    	sleep_t.sleep();
    	
    	// 再次发布确保所有订阅者都收到
    	planning_scene_diff_publisher.publish(planning_scene);

		ROS_INFO("Added table and cylinder obstacle into the world");
	}
    
	void some_functions_maybe_useful(){
		// moveit::planning_interface::MoveGroupInterface arm("manipulator");

		geometry_msgs::PoseStamped current_pose = this->arm_->getCurrentPose(this->end_effector_link);//获取当前的位姿xyz，四元数
		ROS_INFO("current pose:x:%f,y:%f,z:%f,Quaternion:[%f,%f,%f,%f]",current_pose.pose.position.x,current_pose.pose.position.y,
		current_pose.pose.position.z,current_pose.pose.orientation.x,current_pose.pose.orientation.y,
		current_pose.pose.orientation.z,current_pose.pose.orientation.w);

		std::vector<double> current_joint_values = this->arm_->getCurrentJointValues();//获得当前的关节角度，六个关节每个关节的角度
		ROS_INFO("current joint values:%f,%f,%f,%f,%f,%f",current_joint_values[0],current_joint_values[1],current_joint_values[2],
		current_joint_values[3],current_joint_values[4],current_joint_values[5]);

		std::vector<double> rpy = this->arm_->getCurrentRPY(this->end_effector_link);//获得末端的rpy
		ROS_INFO("current rpy:%f,%f,%f",rpy[0],rpy[1],rpy[2]);

		string planner = this->arm_->getPlannerId();//用的规划算法的名称
		ROS_INFO("current planner:%s",planner.c_str());
		std::cout<<"current planner:"<<planner<<endl;//ur5_moveit_config/config/ompl这个文件可以看

	}

	
	~MoveIt_Control() {
		
		ros::shutdown();
	}


public:
	
	string reference_frame;
	string end_effector_link;
	ros::NodeHandle nh_;
	moveit::planning_interface::MoveGroupInterface *arm_;
};

int main(int argc, char** argv) {

	ros::init(argc, argv, "moveit_control_server_cpp");
	ros::AsyncSpinner spinner(1);
	ros::NodeHandle nh;
	spinner.start();
	static const std::string PLANNING_GROUP = "manipulator";
	moveit::planning_interface::MoveGroupInterface arm(PLANNING_GROUP);
	

	MoveIt_Control moveit_server(nh,arm,PLANNING_GROUP);

	// ===================================================================================
	// 各种运动演示 - 可以通过注释/取消注释来选择运行哪个演示
	// ===================================================================================

	// 演示1: 关节空间运动演示
	// 功能：控制机械臂各关节角度直接运动到指定位置
	// cout<<"-----------------------test for move_j----------------------"<<endl;
	// vector<double> joints ={0,0,-1.57,0,0,0};
	// moveit_server.move_j(joints);

	// 演示2: 点到点运动和直线运动演示
	// 功能：先用move_p进行点到点运动，再用move_l进行直线运动
	// cout<<"-----------------------test for move_p and move_l---------------------"<<endl;
	// vector<double> xyzrpy={0.3,0.1,0.4,-3.1415,0,0};
	// moveit_server.move_p(xyzrpy);
	// xyzrpy[2]=0.2;
	// moveit_server.move_l(xyzrpy);

	// 演示3: 多点直线运动演示
	// 功能：通过多个路径点进行连续直线运动
	// cout<<"-----------------------test for move_l(more points)----------------------"<<endl;
	// vector<vector<double>> xyzrpys;
	// xyzrpys.push_back(xyzrpy);
	// xyzrpy[1]=0.2;
	// xyzrpys.push_back(xyzrpy);
	// xyzrpy[0]=0.4;
	// moveit_server.move_l(xyzrpys);

	// 演示4: 改进的姿态约束运动演示 
	// 功能：更优化的姿态约束运动，末端垂直向下保持不变
	// cout << "-----------------------test for move_p_with_constrains----------------------" << endl;
	// // 初始点：正前方，适中高度，末端竖直向下
	// vector<double> pose1 = {0.35, 0.0, 0.35, 0, M_PI, 0};
	// moveit_server.move_p(pose1);
	// // 第二点：向右前方移动，略微升高，末端姿态不变
	// vector<double> pose2 = {0.40, -0.15, 0.40, 0, M_PI, 0};
	// moveit_server.move_p_with_constrains(pose2);
	// // 第三点：向左前方移动，继续升高，末端姿态不变
	// vector<double> pose3 = {0.30, 0.15, 0.45, 0, M_PI, 0};
	// moveit_server.move_p_with_constrains(pose3); 

	// 演示5: MoveIt避障演示 - 手动控制演示 
	// 功能：创建障碍物环境，机械臂移动到up位置，用户可手动控制进行避障测试
	// 场景：在机械臂前方(0.4, 0.0, 0.25)位置放置了一个半径10cm、高度60cm的圆柱体障碍物
	// 测试：用户使用RViz手动控制机械臂进行避障测试
	cout << "=======================================================================================" << endl;
	cout << "                              MoveIt Obstacle Avoidance Demo                          " << endl;
	cout << "=======================================================================================" << endl;
	
	// 移动到up位置作为起始状态
	ROS_INFO("Moving robot to 'up' position...");
	moveit_server.go_home(); // 这会调用up位置
	
	// 给系统一些时间来完全初始化场景
	sleep(2);
	ROS_INFO("Robot is now in 'up' position and ready for obstacle avoidance testing");
	ROS_INFO("Obstacle info: Cylinder at (0.4, 0.0, 0.25), radius=0.1m, height=0.6m");
	
	cout << "=======================================================================================" << endl;
	cout << "                     Setup Complete - Ready for Manual Testing                        " << endl;
	cout << "=======================================================================================" << endl;
	cout << "The robot is now in 'up' position and obstacles have been created." << endl;
	cout << "You can now manually control the robot using MoveIt to test obstacle avoidance:" << endl;
	cout << "1. Use RViz Motion Planning plugin to set target poses" << endl;
	cout << "2. Plan and execute motions around the cylinder obstacle" << endl;
	cout << "3. Observe how MoveIt automatically avoids collisions" << endl;
	cout << "4. Try moving from one side of the cylinder to the other" << endl;
	cout << "5. The taller cylinder (60cm height) provides better obstacle visibility" << endl;
	cout << "=======================================================================================" << endl;
	

	// ===================================================================================
	// 演示7: 综合功能演示 (Integrated Demo)
	// ===================================================================================
	// 说明：此演示结合了 Python 版本的动作逻辑，并使用了前文定义的 move_j, move_p, move_l 函数。
	// 流程：
	// 1. Move_j 到 准备姿态
	// 2. Move_p 到 矩形起始点
	// 3. Move_l 画矩形
	
	// ROS_INFO("================= Starting Demo 7: Integrated Robot Test =================");
	
	// // Step 1: Joint Move (move_j) to Ready Position
	// ROS_INFO("Step 1: Joint Move (move_j) to Ready Position");
	// // 关节角度: [base, shoulder, elbow, wrist1, wrist2, wrist3]
	// std::vector<double> ready_joints = {0, -1.57, 1.57, -1.57, -1.57, 0};
	// moveit_server.move_j(ready_joints);
	// ros::Duration(1.0).sleep();

	// // 定义通用朝下姿态 (RX, RY, RZ) - 沿用代码中的欧拉角定义
	// // 注意：XYZRPY 定义顺序通常为 [x, y, z, r, p, y]
	// double rx_val = -3.141592653589793;
	// double ry_val = 0;
	// double rz_val = 0;

	// // Step 2: Pose Move (move_p) to Start Point of Rectangle
	// ROS_INFO("Step 2: Pose Move (move_p) to Start Point of Rectangle");
	// // 目标位置: x=0.4, y=-0.2, z=0.4 (矩形左上角)
	// std::vector<double> p1 = {0.4, -0.2, 0.4, rx_val, ry_val, rz_val};
	// moveit_server.move_p(p1);
	// ros::Duration(1.0).sleep();

	// // Step 3: Linear Move (move_l) - Drawing a Rectangle
	// ROS_INFO("Step 3: Linear Move (move_l) - Drawing a Rectangle");
	
	// // P2: 右上 (y 变大)
	// std::vector<double> p2 = {0.4, 0.2, 0.4, rx_val, ry_val, rz_val};
	// moveit_server.move_l(p2);
	
	// // P3: 右下 (x 变小, 拉近)
	// std::vector<double> p3 = {0.2, 0.2, 0.4, rx_val, ry_val, rz_val};
	// moveit_server.move_l(p3);
	
	// // P4: 左下 (y 变回 -0.2)
	// std::vector<double> p4 = {0.2, -0.2, 0.4, rx_val, ry_val, rz_val};
	// moveit_server.move_l(p4);
	
	// // 回到 P1 (闭合矩形)
	// moveit_server.move_l(p1);
	// ros::Duration(1.0).sleep();
	
	// // 结束演示，回到 Home
	// ROS_INFO("Test Finished! Going Home...");
	// moveit_server.go_home();
	// ROS_INFO("================= Integrated Robot Test Failed =================");

	// ===================================================================================
	// 实用功能测试
	// 功能：测试一些MoveIt的实用功能，如获取当前位姿、关节角度、RPY姿态、规划器名称等
	// ===================================================================================
	cout<<"-----------------------test for other functions----------------------"<<endl;
	moveit_server.some_functions_maybe_useful();
	return 0;
}

