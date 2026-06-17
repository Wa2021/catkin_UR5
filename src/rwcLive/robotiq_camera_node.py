#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robotiq 腕部相机 ROS 节点

功能：
- 从 Robotiq 腕部相机获取图像
- 发布到 ROS topic: /robotiq_camera/image_raw
- 发布相机信息到: /robotiq_camera/camera_info

使用方法：
    rosrun rwcLive robotiq_camera_node.py _robot_ip:=192.168.0.1

或在 launch 文件中：
    <node name="robotiq_camera" pkg="rwcLive" type="robotiq_camera_node.py">
        <param name="robot_ip" value="192.168.0.1"/>
        <param name="publish_rate" value="10"/>
    </node>
"""

import rospy
import requests
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError


class RobotiqCameraNode:
    """Robotiq 腕部相机 ROS 节点"""
    
    def __init__(self):
        """初始化节点"""
        # 初始化 ROS 节点
        rospy.init_node('robotiq_camera_node', anonymous=False)
        
        # 获取参数
        self.robot_ip = rospy.get_param('~robot_ip', '192.168.0.1')
        self.camera_port = rospy.get_param('~camera_port', 4242)
        self.publish_rate = rospy.get_param('~publish_rate', 10)  # Hz
        self.frame_id = rospy.get_param('~frame_id', 'robotiq_camera_optical_frame')
        
        # 打印配置信息
        rospy.loginfo("="*60)
        rospy.loginfo("Robotiq 腕部相机节点已启动")
        rospy.loginfo("="*60)
        rospy.loginfo(f"机器人 IP: {self.robot_ip}")
        rospy.loginfo(f"相机端口: {self.camera_port}")
        rospy.loginfo(f"发布频率: {self.publish_rate} Hz")
        rospy.loginfo(f"坐标系: {self.frame_id}")
        
        # 创建发布者
        self.image_pub = rospy.Publisher(
            '/robotiq_camera/image_raw', 
            Image, 
            queue_size=10
        )
        self.camera_info_pub = rospy.Publisher(
            '/robotiq_camera/camera_info', 
            CameraInfo, 
            queue_size=10
        )
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 相机 URL
        self.camera_url = f"http://{self.robot_ip}:{self.camera_port}/current.jpg?type=color"
        
        # 统计信息
        self.frame_count = 0
        self.error_count = 0
        self.last_success_time = None
        
        # 设置发布频率
        self.rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo("节点初始化完成，开始采集图像...")
        rospy.loginfo("="*60)
    
    def get_image(self):
        """
        从相机获取图像
        
        Returns:
            numpy.ndarray: OpenCV 格式的图像 (BGR)，失败返回 None
        """
        try:
            # 发送 HTTP 请求
            response = requests.get(self.camera_url, timeout=1.0)
            
            if response.status_code == 200:
                # 将字节数据转换为 numpy 数组
                image_data = np.asarray(bytearray(response.content), dtype="uint8")
                
                # 使用 OpenCV 解码图像
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    return image
                else:
                    rospy.logwarn("图像解码失败")
                    return None
            else:
                rospy.logwarn(f"HTTP 响应错误: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            rospy.logwarn_throttle(5.0, "相机连接超时")
            return None
            
        except requests.exceptions.ConnectionError:
            rospy.logwarn_throttle(5.0, f"无法连接到相机 ({self.robot_ip}:{self.camera_port})")
            return None
            
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"获取图像时发生错误: {e}")
            return None
    
    def create_camera_info(self, height, width):
        """
        创建相机信息消息
        
        Args:
            height: 图像高度
            width: 图像宽度
            
        Returns:
            CameraInfo: 相机信息消息
        """
        camera_info = CameraInfo()
        camera_info.header.frame_id = self.frame_id
        camera_info.height = height
        camera_info.width = width
        
        # 这些值需要通过相机标定获得
        # 这里使用默认值，实际使用时应该通过标定获取
        camera_info.distortion_model = "plumb_bob"
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # 畸变系数
        
        # 相机内参矩阵 (需要标定)
        fx = fy = width  # 简化假设
        cx = width / 2.0
        cy = height / 2.0
        
        camera_info.K = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]
        
        camera_info.R = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        camera_info.P = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        return camera_info
    
    def publish_image(self, cv_image):
        """
        发布图像到 ROS topic
        
        Args:
            cv_image: OpenCV 格式的图像
        """
        try:
            # 创建消息头
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.frame_id
            
            # 转换为 ROS Image 消息
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            ros_image.header = header
            
            # 发布图像
            self.image_pub.publish(ros_image)
            
            # 创建并发布相机信息
            camera_info = self.create_camera_info(
                cv_image.shape[0], 
                cv_image.shape[1]
            )
            camera_info.header = header
            self.camera_info_pub.publish(camera_info)
            
            # 更新统计
            self.frame_count += 1
            self.last_success_time = rospy.Time.now()
            
            # 定期打印统计信息
            if self.frame_count % 100 == 0:
                rospy.loginfo(
                    f"已发布 {self.frame_count} 帧图像 "
                    f"(错误: {self.error_count})"
                )
            
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")
            self.error_count += 1
            
        except Exception as e:
            rospy.logerr(f"发布图像时发生错误: {e}")
            self.error_count += 1
    
    def run(self):
        """运行节点主循环"""
        rospy.loginfo("开始采集和发布图像...")
        
        while not rospy.is_shutdown():
            # 获取图像
            image = self.get_image()
            
            if image is not None:
                # 发布图像
                self.publish_image(image)
            else:
                self.error_count += 1
            
            # 按照设定频率休眠
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        # 节点关闭时打印统计信息
        rospy.loginfo("="*60)
        rospy.loginfo("节点关闭")
        rospy.loginfo(f"总共发布帧数: {self.frame_count}")
        rospy.loginfo(f"错误次数: {self.error_count}")
        if self.frame_count > 0:
            success_rate = (self.frame_count / (self.frame_count + self.error_count)) * 100
            rospy.loginfo(f"成功率: {success_rate:.2f}%")
        rospy.loginfo("="*60)


def main():
    """主函数"""
    try:
        node = RobotiqCameraNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr(f"节点运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
