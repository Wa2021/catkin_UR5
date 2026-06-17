#!/usr/bin/env python3
"""
手持RealSense相机YOLO物体检测程序
功能：实时检测各种物体，适合手持使用
按键：'q'退出，'s'保存截图，'c'清空控制台
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os
import datetime

class HandheldYOLODetector:
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.detection_count = 0
        self.frame_count = 0
        
    def initialize_camera(self):
        """初始化RealSense相机"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 配置彩色流 - 使用较高分辨率获得更好的检测效果
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            print("正在启动RealSense相机...")
            self.pipeline.start(config)
            print("✓ 相机启动成功")
            return True
            
        except Exception as e:
            print(f"✗ 相机启动失败: {e}")
            print("请检查：")
            print("1. RealSense相机是否正确连接")
            print("2. USB接口是否正常")
            print("3. 是否有其他程序在使用相机")
            return False
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../../models/yolov8n.pt')
            )
            if not os.path.exists(model_path):
                print(f"✗ 找不到模型文件: {model_path}")
                return False
                
            print("正在加载YOLO模型...")
            self.model = YOLO(model_path)
            print("✓ YOLO模型加载成功")
            return True
            
        except Exception as e:
            print(f"✗ YOLO模型加载失败: {e}")
            return False
    
    def save_detection_result(self, image, detections):
        """保存检测结果"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_result_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, image)
            print(f"✓ 检测结果已保存: {filename}")
            
            # 打印检测到的物体信息
            if len(detections) > 0:
                print("检测到的物体:")
                for i, det in enumerate(detections):
                    if hasattr(det, 'boxes') and det.boxes is not None:
                        boxes = det.boxes
                        if boxes.cls is not None:
                            for j, cls_id in enumerate(boxes.cls):
                                class_name = self.model.names[int(cls_id)]
                                confidence = float(boxes.conf[j]) if boxes.conf is not None else 0
                                print(f"  - {class_name}: {confidence:.2f}")
            
        except Exception as e:
            print(f"✗ 保存失败: {e}")
    
    def run_detection(self):
        """运行实时检测"""
        if not self.initialize_camera():
            return
            
        if not self.load_model():
            self.cleanup()
            return
        
        print("\n" + "="*50)
        print("手持RealSense YOLO检测开始!")
        print("操作说明:")
        print("  'q' - 退出程序")
        print("  's' - 保存当前检测结果")
        print("  'c' - 清空控制台显示")
        print("="*50)
        
        try:
            while True:
                # 获取相机帧
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # 转换为OpenCV格式
                image = np.asanyarray(color_frame.get_data())
                self.frame_count += 1
                
                # YOLO检测
                results = self.model(image, show=False, verbose=False)
                
                # 绘制检测结果
                annotated_frame = results[0].plot()
                
                # 添加状态信息
                status_text = f"Frame: {self.frame_count} | Objects: {len(results[0].boxes) if results[0].boxes is not None else 0}"
                cv2.putText(annotated_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 添加操作提示
                cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", (10, 460), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示结果
                cv2.imshow('Handheld RealSense YOLO Detection', annotated_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出程序")
                    break
                elif key == ord('s'):
                    self.save_detection_result(annotated_frame, results)
                elif key == ord('c'):
                    os.system('clear')  # Linux清屏
                    print("控制台已清空")
                
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        except Exception as e:
            print(f"\n检测过程中出错: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("✓ 程序结束，资源已释放")

def main():
    detector = HandheldYOLODetector()
    detector.run_detection()

if __name__ == "__main__":
    main()
