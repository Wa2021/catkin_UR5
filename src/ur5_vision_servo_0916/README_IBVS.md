# IBVS 视觉伺服系统（ArUco 模拟 YOLO-OBB）

本系统实现了一个基于图像的视觉伺服（Image-Based Visual Servoing, IBVS）控制方案，使用 ArUco 码作为视觉标记来**模拟 YOLO-OBB 的输出**，便于在没有训练好 YOLO 模型时进行算法验证。

---

## 🎯 设计目标

1. **可迁移性**：代码结构与 YOLO-OBB 完全兼容，切换时只需修改感知层
2. **6自由度控制**：支持位置 (X, Y, Z) + 姿态 (Roll, Pitch, Yaw) 全自由度伺服
3. **实时性**：通过 MoveIt Servo 实现高频（~30Hz）闭环控制

---

## 📊 IBVS vs PBVS 对比

| 特性 | **PBVS** (`pbvs视觉伺服7_1.py`) | **IBVS** (`visual_servo.py`) |
|------|------|------|
| **控制空间** | 笛卡尔空间 (3D) | 图像空间 (2D) |
| **误差定义** | 3D 位置误差 | 2D 像素误差 |
| **对标定敏感** | ⚠️ 非常敏感 | ✅ 不敏感 |
| **控制方式** | 位置控制 (`move_j_p_1`) | 速度控制 (MoveIt Servo) |
| **适用场景** | 静态精确定位 | 动态跟踪 |

---

## 🔧 系统架构

```
┌─────────────────┐     图像      ┌─────────────────┐
│   RealSense     │ ───────────→ │  visual_servo.py │
│   Camera        │              │  (ArUco→YOLO)    │
└─────────────────┘              └────────┬────────┘
                                          │ TwistStamped
                                          ↓
┌─────────────────┐   关节速度    ┌─────────────────┐
│  ur_modern_     │ ←─────────── │  MoveIt Servo   │
│  driver         │              │  Node           │
└────────┬────────┘              └─────────────────┘
         │ TCP/IP
         ↓
┌─────────────────┐
│   UR5 Robot     │
└─────────────────┘
```

---

## 📁 文件清单

| 文件 | 路径 | 作用 |
|------|------|------|
| `visual_servo.py` | `src/ur5_vision_servo_0916/` | IBVS 主程序 |
| `ur5_controllers.yaml` | `src/universal_robot/ur_modern_driver/config/` | 定义速度控制器 |
| `ur5_ros_control.launch` | `src/universal_robot/ur_modern_driver/launch/` | 启动驱动 |
| `ur5_servo_config.yaml` | `src/universal_robot/ur5_moveit_config/config/` | Servo 配置 |
| `ur5_servo.launch` | `src/universal_robot/ur5_moveit_config/launch/` | 启动 Servo |

---

## 🚀 启动步骤

### 前置条件

1. 安装 MoveIt Servo：
   ```bash
   sudo apt install ros-noetic-moveit-servo
   ```

2. 准备一个 ArUco 码（打印 `DICT_4X4_50` 字典中的任意 ID）

### 启动流程（4 个终端）

```bash
# 终端 1: 启动机械臂驱动（支持速度控制）
roslaunch ur_modern_driver ur5_ros_control.launch robot_ip:=192.168.0.1

# 终端 2: 启动 MoveIt 规划环境
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch

# 终端 3: 启动 MoveIt Servo 服务
roslaunch ur5_moveit_config ur5_servo.launch

# 终端 4: 启动视觉伺服程序
rosrun ur5_vision_servo_0916 visual_servo.py
```

---

## 🔬 核心算法：ArUco 如何伪装成 YOLO-OBB

### 为什么要"伪装"？

| ArUco 角点 | YOLO-OBB 角点 |
|------------|---------------|
| ✅ 顺序固定（左上永远是左上） | ❌ 顺序不固定（旋转后会跳变） |

如果直接用 ArUco 的 4 角点构建 8×6 雅可比矩阵，虽然控制精准，但**代码无法迁移到 YOLO**。

### 解决方案：降维到 YOLO 能提供的信息

```python
def get_aruco_features(self, img):
    """把 ArUco 数据转换成 YOLO-OBB 格式"""
    corners, ids, _ = aruco.detectMarkers(...)
    c = corners[0][0]  # 4 个角点
    
    # 1. 中心点 (u, v) —— YOLO 能给
    u = np.mean(c[:, 0])
    v = np.mean(c[:, 1])
    
    # 2. 面积 —— YOLO 能给 (w × h)
    area = cv2.contourArea(c)
    
    # 3. 旋转角 —— YOLO-OBB 直接给 angle
    dx = c[1][0] - c[0][0]
    dy = c[1][1] - c[0][1]
    angle = math.atan2(dy, dx)
    
    # 4. 俯仰/横滚 —— 用 ArUco 姿态估计模拟平面拟合
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(...)
    
    return u, v, area, angle, rvec
```

### 切换到 YOLO 时只需改这里

```python
# ArUco 版本
u, v = np.mean(corners[:, 0]), np.mean(corners[:, 1])
area = cv2.contourArea(corners)
angle = math.atan2(dy, dx)

# YOLO 版本（只改这几行）
u, v = result.obb.xywhr[0], result.obb.xywhr[1]
area = result.obb.xywhr[2] * result.obb.xywhr[3]
angle = result.obb.xywhr[4]
```

---

## ⚙️ 参数调节

### `visual_servo.py` 中的关键参数

```python
# 期望状态
self.target_u = 320       # 图像中心 X（根据分辨率调整）
self.target_v = 240       # 图像中心 Y
self.target_area = 25000  # 期望面积（决定距离）
self.target_angle = 0     # 期望角度（让夹爪对正）

# PID 增益
self.lambda_xy = 0.002    # 位置增益（太大会振荡）
self.lambda_z = 0.0001    # 深度增益
self.lambda_rot = 0.02    # 旋转增益
```

### `ur5_servo_config.yaml` 中的关键参数

```yaml
scale:
  linear: 0.6   # 最大线速度 [m/s]
  rotational: 0.3  # 最大角速度 [rad/s]

publish_period: 0.034  # 发布周期 (~30Hz)

# 碰撞检测
check_collisions: true
collision_check_rate: 50
```

---

## 🛡️ 安全注意事项

1. **速度限幅**：代码中已设置最大速度 0.1 m/s，请勿随意调大
2. **碰撞检测**：MoveIt Servo 会自动进行碰撞检测
3. **急停**：在 RViz 中可以随时点击 "Stop" 或按机械臂示教器上的急停按钮

---

## 🔄 未来迁移到 YOLO-OBB

当你训练好 YOLO-OBB 模型后，只需修改 `visual_servo.py` 中的感知层：

1. 导入 YOLO：`from ultralytics import YOLO`
2. 替换 `get_aruco_features()` 为 `get_yolo_features()`
3. 控制层代码**完全不变**

---

## 📚 参考资料

- [MoveIt Servo 官方文档](https://ros-planning.github.io/moveit_tutorials/doc/realtime_servo/realtime_servo_tutorial.html)
- [OpenCV ArUco 模块](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [视觉伺服综述论文](https://hal.inria.fr/inria-00350283/document)
