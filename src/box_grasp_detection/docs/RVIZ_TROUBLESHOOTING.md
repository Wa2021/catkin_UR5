# RViz 崩溃问题解决方案

## 问题描述
RViz 频繁崩溃（exit code -11），特别是在虚拟机环境中。

## 已实施的优化

### 1. RViz 配置优化 (`box_detection.rviz`)
- **减少队列大小**: PointCloud2 Queue Size 从 10 降到 1
- **降低渲染复杂度**: 
  - 点云大小从 3px 降到 2px
  - Style 从 "Flat Squares" 改为 "Points"
  - MarkerArray Queue Size 从 100 降到 10
- **禁用第二个点云**: "Detected Objects" 默认关闭，避免双点云冲突

### 2. Launch 文件优化 (`ur5_eye_in_hand.launch`)
- **自动重启**: `respawn="true"` - RViz 崩溃后自动重启
- **重启延迟**: `respawn_delay="2"` - 等待2秒再重启
- **软件渲染**: `LIBGL_ALWAYS_SOFTWARE=1` - 虚拟机友好

### 3. 安全模式启动 (`rviz_safe.launch`)
提供了一个独立的安全启动文件，包含更多稳定性设置：
```bash
roslaunch box_grasp_detection rviz_safe.launch
```

## 使用方法

### 正常启动（已包含优化）
```bash
roslaunch box_grasp_detection ur5_eye_in_hand.launch
```

### 单独启动安全模式 RViz
如果主launch文件的RViz仍然崩溃，可以先关闭它，然后用安全模式启动：
```bash
# 终端1：启动检测系统（不含RViz）
roslaunch box_grasp_detection ur5_eye_in_hand.launch

# 在另一个终端中手动终止RViz（如果它在崩溃）
# 然后单独启动安全模式RViz

# 终端2：安全模式RViz
roslaunch box_grasp_detection rviz_safe.launch
```

## 其他建议

### 1. 检查显卡驱动
```bash
# 检查当前OpenGL渲染器
glxinfo | grep "OpenGL renderer"

# 如果是虚拟机，应该看到类似：
# OpenGL renderer string: llvmpipe 或 Software Rasterizer
```

### 2. 增加虚拟机显存
在 VMware/VirtualBox 设置中：
- **VMware**: 虚拟机设置 → 显示器 → 3D图形加速 → 分配显存 ≥ 256MB
- **VirtualBox**: 设置 → 显示 → 显存大小 ≥ 128MB

### 3. 降低点云分辨率
编辑 `ur5_eye_in_hand.launch`，在 RealSense 启动中添加：
```xml
<include file="$(find realsense2_camera)/launch/demo_pointcloud.launch">
  <arg name="camera" value="$(arg camera_namespace)" />
  <arg name="filters" value="decimation" />  <!-- 降采样 -->
</include>
```

### 4. 临时禁用 RViz
如果 RViz 持续崩溃，可以只使用 Open3D 可视化：
```bash
# 启动检测器（不启动RViz）
roslaunch box_grasp_detection ur5_eye_in_hand.launch

# Ctrl+C 停止崩溃的RViz

# 使用单帧检测（会启动Open3D窗口）
rosrun box_grasp_detection single_frame_detector.py
```

## 调试技巧

### 查看 RViz 日志
```bash
# 查看最新的RViz日志
tail -f ~/.ros/log/latest/rviz-*.log
```

### 手动启动 RViz（不通过 roslaunch）
```bash
# 使用软件渲染
LIBGL_ALWAYS_SOFTWARE=1 rosrun rviz rviz -d ~/catkin_UR5/src/box_grasp_detection/config/box_detection.rviz
```

### 最小化配置测试
```bash
# 启动空白RViz
LIBGL_ALWAYS_SOFTWARE=1 rosrun rviz rviz

# 然后手动添加显示器，逐个测试哪个导致崩溃
```

## 已知问题

1. **虚拟机OpenGL限制**: 
   - 软件渲染性能较低，但更稳定
   - 大点云（>100K点）可能卡顿

2. **自动重启可能循环**:
   - 如果问题持续，RViz会不断重启
   - 解决：Ctrl+C 停止整个launch，检查根本原因

3. **点云太大**:
   - RealSense D435 原始点云约30-60万点
   - 建议使用体素滤波（已在检测器中实施）

## 验证优化效果

运行系统后观察：
```bash
# 观察RViz是否稳定运行超过30秒
roslaunch box_grasp_detection ur5_eye_in_hand.launch

# 查看资源使用
top -p $(pgrep rviz)
```

成功标志：
- ✅ RViz 能稳定运行 >30秒
- ✅ 点云显示流畅
- ✅ 不出现 "exit code -11" 错误

失败标志：
- ❌ 仍然频繁崩溃（<10秒）
- ❌ "Segmentation fault" 错误
- ❌ 黑屏或无响应

如果仍然失败，考虑使用 Open3D 作为主要可视化工具。
