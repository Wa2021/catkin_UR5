# 相机查看器使用说明

## 概述

这个launch文件允许您单独启动RealSense相机并查看图像，无需启动完整的障碍物避障系统。适用于：
- 测试相机功能
- 调试相机设置
- 查看相机视野
- 验证相机安装位置

## 启动方式

```bash
roslaunch ur5_gripper_moveit_config camera_viewer.launch
```

## 功能说明

### 默认启动的组件

1. **RealSense相机节点** - 启动相机硬件驱动
2. **彩色图像查看器** - 自动打开窗口显示彩色图像
3. **相机信息监控** - 在后台记录相机参数信息

### 可选组件（需要手动取消注释）

在launch文件中，您可以取消以下组件的注释来启用：

#### 深度图像查看器
```xml
<node pkg="image_view" type="image_view" name="camera_depth_viewer" output="screen">
  <remap from="image" to="/camera/depth/image_rect_raw"/>
  <param name="autosize" value="true"/>
  <param name="window_name" value="RealSense Depth Image"/>
</node>
```

#### RQT图像查看器（更灵活）
```xml
<node pkg="rqt_image_view" type="rqt_image_view" name="rqt_camera_viewer" output="screen"/>
```

## 使用场景

### 场景1：快速测试相机
```bash
# 只想看看相机是否正常工作
roslaunch ur5_gripper_moveit_config camera_viewer.launch
```

### 场景2：同时查看彩色和深度图像
1. 编辑 `camera_viewer.launch` 文件
2. 取消深度图像查看器的注释
3. 启动launch文件

### 场景3：使用RQT进行更灵活的图像查看
1. 取消RQT相关行的注释
2. 启动后可以在RQT界面中选择不同的图像话题

## 可用的图像话题

启动后，以下话题将可用：

```bash
# 彩色图像
/camera/color/image_raw

# 深度图像
/camera/depth/image_rect_raw

# 点云数据
/camera/depth/color/points

# 相机参数
/camera/color/camera_info
/camera/depth/camera_info
```

## 手动命令替代方案

如果您想手动控制，也可以分步启动：

```bash
# 1. 只启动相机
roslaunch realsense2_camera demo_pointcloud.launch

# 2. 在新终端启动图像查看器
rosrun image_view image_view image:=/camera/color/image_raw

# 3. 或者启动RQT图像查看器
rosrun rqt_image_view rqt_image_view
```

## 与完整系统的区别

| 功能 | camera_viewer.launch | demo_gripper_with_obstacle_avoidance.launch |
|------|---------------------|---------------------------------------------|
| 相机启动 | ✅ | ✅ |
| 图像显示 | ✅ | 需要手动启动 |
| 机器人模型 | ❌ | ✅ |
| MoveIt规划 | ❌ | ✅ |
| 障碍物检测 | ❌ | ✅ |
| 快照控制 | ❌ | ✅ |

## 故障排除

### 1. 相机无法启动
```bash
# 检查相机连接
lsusb | grep Intel

# 检查权限
ls -l /dev/video*
```

### 2. 图像窗口无法显示
```bash
# 检查X11转发（如果使用SSH）
echo $DISPLAY

# 测试图像查看器
rosrun image_view image_view image:=/camera/color/image_raw
```

### 3. 相机话题无数据
```bash
# 检查话题列表
rostopic list | grep camera

# 检查话题数据
rostopic echo /camera/color/image_raw --noarr
```

## 文件位置

```
ur5_gripper_moveit_config/
├── launch/
│   ├── camera_viewer.launch              # 相机查看器启动文件
│   ├── camera_viewer.md                  # 本说明文档
│   └── demo_gripper_with_obstacle_avoidance.launch
```