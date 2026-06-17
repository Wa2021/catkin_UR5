# UR5 Gripper 障碍物避障系统使用说明

## 概述

本系统实现了Eye-in-Hand配置的UR5机械臂障碍物避障功能，相机安装在机械臂末端执行器上。系统支持自动和手动的障碍物地图冻结控制，解决了Eye-in-Hand配置中相机移动导致视野变化的问题。

## 启动方式

### 主系统启动

```bash
roslaunch ur5_gripper_moveit_config demo_gripper_with_obstacle_avoidance.launch
```

### 相机图像监控（可选）

为了在障碍物检测过程中监控相机看到的内容，可以在新终端中启动图像查看器：

```bash
# 查看相机彩色图像
rosrun image_view image_view image:=/camera/color/image_raw

# 或者使用更灵活的图像查看器
rosrun rqt_image_view rqt_image_view
```

### 单独的相机查看器

如果只想测试相机或查看相机图像（不需要障碍物避障功能），可以使用专门的相机查看器：

```bash
# 启动独立的相机查看器
roslaunch ur5_gripper_moveit_config camera_viewer.launch
```

详细说明请参考：[camera_viewer.md](camera_viewer.md)

**建议使用方式**：
- 主窗口运行RViz查看机器人和障碍物地图
- 单独窗口运行image_view监控相机实时图像
- 这样可以同时观察环境检测情况和相机视野

## 系统特性

### 1. 自动冻结机制
- **默认设置**: 启动后5秒自动冻结障碍物更新
- **目的**: 在观察期内建立完整的环境地图，随后冻结以避免抓取时视野变化
- **状态**: 冻结后系统会输出确认信息

### 2. Eye-in-Hand配置
- 相机通过URDF固定在机械臂末端 (`ee_link`)
- 相机与机械臂的变换关系已在URDF中定义
- 无需额外的静态变换发布器

## 手动控制服务

### 可用服务列表

| 服务名称              | 功能描述 |                使用场景 |
|---------|----------|----------|
| `/snapshot/freeze` | 立即冻结障碍物更新           |提前结束观察期 |
| `/snapshot/unfreeze` | 恢复障碍物更新并重置计时器   |重新开始观察 |
| `/snapshot/clear` | 清空当前障碍物地图            | 清除错误的障碍物 |
| `/snapshot/reset_timer` | 重置自动冻结计时器      | 延长观察时间 |

### 使用命令

```bash
# 立即冻结障碍物更新（提前结束观察期）
rosservice call /snapshot/freeze

# 恢复障碍物更新（重新开始5秒倒计时）
rosservice call /snapshot/unfreeze

# 清空当前障碍物地图
rosservice call /snapshot/clear

# 重置计时器（重新开始5秒倒计时，但保持当前冻结状态）
rosservice call /snapshot/reset_timer
```

## 自动冻结与手动控制的关系

### ✅ 手动命令始终有效

**重要**: 即使启用了自动冻结，所有手动控制服务依然完全有效，可以随时覆盖自动行为。

### 优先级关系

1. **手动冻结** > 自动冻结
   - 可以在自动冻结前手动冻结
   - 手动冻结会设置已冻结标志，停止自动冻结计时

2. **手动解冻** > 任何冻结状态
   - 无论是自动还是手动冻结，都可以手动解冻
   - 解冻后会重新开始自动冻结倒计时

3. **重置计时器**的行为
   - 如果当前是冻结状态：先自动解冻，然后重新开始倒计时
   - 如果当前是解冻状态：重新开始倒计时

### 典型使用场景

#### 场景1: 标准工作流程
```bash
# 1. 启动系统
roslaunch ur5_gripper_moveit_config demo_gripper_with_obstacle_avoidance.launch

# 2. 等待5秒自动冻结（或提前手动冻结）
# 系统输出：自动冻结：5秒后自动停止障碍物更新

# 3. 执行抓取任务（基于冻结的地图）

# 4. 任务完成后恢复更新
rosservice call /snapshot/unfreeze
```

#### 场景2: 需要更长观察时间
```bash
# 1. 启动系统后，需要更多时间观察
rosservice call /snapshot/reset_timer

# 2. 再次获得5秒观察时间
```

#### 场景3: 立即冻结
```bash
# 1. 启动系统后，环境已经足够清楚
rosservice call /snapshot/freeze

# 2. 立即进入冻结状态，不等待自动冻结
```

#### 场景4: 清除错误障碍物
```bash
# 1. 发现障碍物检测错误
rosservice call /snapshot/clear

# 2. 重新开始观察
rosservice call /snapshot/reset_timer
```

## 参数配置

### Launch文件参数

```xml
<!-- 是否启用自动冻结 -->
<param name="auto_freeze_enabled" value="true" />

<!-- 自动冻结延迟时间（秒） -->
<param name="freeze_delay" value="5.0" />
```

### 运行时参数修改

```bash
# 修改自动冻结延迟为10秒
rosparam set /snapshot_controller/freeze_delay 10.0

# 禁用自动冻结（仅手动控制）
rosparam set /snapshot_controller/auto_freeze_enabled false
```

## 传感器配置

系统使用RealSense深度相机，相关配置：
- **点云话题**: `/camera/depth/color/points` → `/camera/depth/color/points_relay`
- **最大检测距离**: 1.3米
- **更新频率**: 0.5Hz（降低CPU占用）
- **点云采样**: 1（不降采样）

## 故障排除

### 1. TF循环错误
- **症状**: 报告TF树循环，涉及camera_color_optical_frame和ee_link
- **原因**: static_transform_publisher与URDF定义冲突
- **解决**: 确保static_transform_publisher已被注释掉

### 2. 障碍物不显示
- **检查RViz设置**: 在Motion Planning插件中启用"Show Scene Geometry"
- **检查话题**: 确认`/camera/depth/color/points_relay`有数据发布
- **检查服务**: 确认`rostopic info /camera/depth/color/points_relay`显示MoveIt在订阅

### 3. 自动冻结不工作
- **检查参数**: `rosparam get /snapshot_controller/auto_freeze_enabled`
- **检查日志**: 查看snapshot_controller的输出日志
- **手动测试**: 使用手动服务验证基本功能

## 日志信息

系统运行时的关键日志信息：

```
# 启动成功
Snapshot controller ready!
Auto-freeze enabled: will freeze after 5.0 seconds

# 自动冻结
自动冻结：5秒后自动停止障碍物更新

# 手动操作
手动冻结：障碍物地图已冻结 - 停止更新
障碍物地图已解冻 - 恢复更新，重置自动冻结计时器
计时器已重置，将在5.0秒后自动冻结
```

## 技术说明

### 工作原理
1. **点云中继**: snapshot_control.py作为中继，根据冻结状态决定是否转发点云数据
2. **MoveIt集成**: MoveIt订阅中继话题而非原始相机话题
3. **状态管理**: 使用线程锁确保状态变更的原子性

### 文件结构
```
ur5_gripper_moveit_config/
├── launch/
│   ├── demo_gripper_with_obstacle_avoidance.launch  # 主启动文件
│   └── README_obstacle_avoidance.md               # 本说明文档
├── scripts/
│   └── snapshot_control.py                        # 快照控制节点
└── config/
    └── sensors_3d.yaml                           # 传感器配置
```