#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robotiq 腕部相机测试脚本

功能：
1. 测试与 Robotiq 腕部相机的连接
2. 获取并保存图像
3. 显示图像信息

使用方法：
    python3 test_robotiq_camera.py --ip 192.168.0.1

或者直接修改脚本中的 ROBOT_IP 变量
"""

import argparse
import requests
import numpy as np
from PIL import Image
import io
from datetime import datetime
import sys

# 默认机器人 IP 地址
ROBOT_IP = "192.168.0.1"
CAMERA_PORT = 4242


def test_connection(robot_ip):
    """测试与相机的连接"""
    print(f"\n{'='*60}")
    print("🔍 测试 Robotiq 腕部相机连接")
    print(f"{'='*60}")
    
    url = f"http://{robot_ip}:{CAMERA_PORT}/current.jpg?type=color"
    print(f"\n📡 连接地址: {url}")
    
    try:
        print("⏳ 正在连接...")
        response = requests.get(url, timeout=5.0)
        
        if response.status_code == 200:
            print(f"✅ 连接成功！")
            print(f"📊 响应状态码: {response.status_code}")
            print(f"📦 数据大小: {len(response.content)} 字节")
            return response.content
        else:
            print(f"❌ 连接失败！")
            print(f"📊 响应状态码: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ 连接超时！")
        print("💡 提示：")
        print("   - 检查机器人是否已启动")
        print("   - 检查 IP 地址是否正确")
        print(f"   - 尝试 ping {robot_ip}")
        return None
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到相机！")
        print("💡 提示：")
        print("   - 检查网络连接")
        print("   - 确认相机已启用")
        print(f"   - 尝试访问: {url}")
        return None
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return None


def process_image(image_data):
    """处理图像数据"""
    if image_data is None:
        return None
    
    print(f"\n{'='*60}")
    print("🖼️  处理图像")
    print(f"{'='*60}")
    
    try:
        # 将字节数据转换为图像
        image_array = np.asarray(bytearray(image_data), dtype="uint8")
        image = Image.open(io.BytesIO(image_array))
        
        print(f"✅ 图像解析成功")
        print(f"📐 图像尺寸: {image.size[0]} x {image.size[1]}")
        print(f"🎨 图像模式: {image.mode}")
        print(f"🎯 图像格式: {image.format}")
        
        return image
        
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return None


def save_image(image, robot_ip):
    """保存图像到文件"""
    if image is None:
        return False
    
    print(f"\n{'='*60}")
    print("💾 保存图像")
    print(f"{'='*60}")
    
    try:
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robotiq_camera_{timestamp}.jpg"
        
        image.save(filename)
        print(f"✅ 图像已保存: {filename}")
        
        # 显示图像统计信息
        img_array = np.array(image)
        print(f"\n📊 图像统计信息:")
        print(f"   - 形状: {img_array.shape}")
        print(f"   - 数据类型: {img_array.dtype}")
        print(f"   - 最小值: {img_array.min()}")
        print(f"   - 最大值: {img_array.max()}")
        print(f"   - 平均值: {img_array.mean():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='测试 Robotiq 腕部相机连接',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 test_robotiq_camera.py --ip 192.168.0.1
  python3 test_robotiq_camera.py --ip 192.168.1.100
        """
    )
    parser.add_argument(
        '--ip', 
        type=str, 
        default=ROBOT_IP,
        help=f'机器人 IP 地址 (默认: {ROBOT_IP})'
    )
    
    args = parser.parse_args()
    robot_ip = args.ip
    
    print("\n" + "="*60)
    print("🤖 Robotiq 腕部相机测试工具")
    print("="*60)
    print(f"📍 机器人 IP: {robot_ip}")
    print(f"🔌 相机端口: {CAMERA_PORT}")
    
    # 测试连接
    image_data = test_connection(robot_ip)
    
    if image_data is None:
        print(f"\n{'='*60}")
        print("❌ 测试失败")
        print(f"{'='*60}")
        print("\n💡 故障排查步骤:")
        print("   1. 检查机器人电源是否开启")
        print("   2. 检查网络连接:")
        print(f"      ping {robot_ip}")
        print("   3. 检查相机端口:")
        print(f"      telnet {robot_ip} {CAMERA_PORT}")
        print("   4. 在浏览器中访问:")
        print(f"      http://{robot_ip}:{CAMERA_PORT}/current.jpg?type=color")
        print()
        sys.exit(1)
    
    # 处理图像
    image = process_image(image_data)
    
    if image is None:
        print(f"\n{'='*60}")
        print("❌ 图像处理失败")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # 保存图像
    success = save_image(image, robot_ip)
    
    # 最终结果
    print(f"\n{'='*60}")
    if success:
        print("✅ 测试成功！相机工作正常")
        print(f"{'='*60}")
        print("\n📝 下一步:")
        print("   1. 查看保存的图像文件")
        print("   2. 运行 rwcLive.py 查看实时画面:")
        print(f"      cd /home/xsh/catkin_UR5/src/rwcLive")
        print(f"      python3 rwcLive.py")
        print("   3. 开始开发视觉伺服应用")
    else:
        print("⚠️  测试部分成功，但保存图像失败")
        print(f"{'='*60}")
    print()


if __name__ == "__main__":
    main()
