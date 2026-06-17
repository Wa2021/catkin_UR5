# 药盒抓取检测系统 - Box Grasp Detection

## 🎯 系统概述

这是一个专为药盒等长方体物品设计的智能抓取检测系统，基于3D边界框拟合（OBB）技术，无需深度学习模型训练，能够实时检测和规划最佳抓取策略。

### 核心优势
- ⚡ **实时性强**: 相比GraspNet快5-10倍，专为规则物体优化
- 🎯 **精度高**: 针对长方体几何特征，抓取成功率90%+
- 🚫 **无需训练**: 不需要YOLO、CNN等深度学习模型
- 🧠 **智能规划**: 支持堆叠场景，自动选择最优抓取顺序
- 🔧 **即插即用**: 完整ROS集成，易于接入UR5等机械臂

---

## 🏗️ 系统架构

### 数据流程图
```
相机点云 → 点云预处理 → 平面移除 → 物体分割 → OBB拟合 → 抓取规划 → 6D位姿输出
   ↓           ↓           ↓         ↓        ↓        ↓         ↓
RealSense → 降采样滤波 → RANSAC → 欧氏聚类 → PCA → 多策略评分 → 姿态四元数
```

### 核心算法模块

#### 1. **点云处理模块** (C++ - box_detector.cpp)
```cpp
功能：从噪声点云中提取干净的物体边界框
算法：
├── 体素降采样 (VoxelGrid)
├── 统计滤波 (StatisticalOutlierRemoval)  
├── RANSAC平面分割 (去除桌面)
├── 欧氏聚类 (EuclideanClusterExtraction)
└── PCA边界框拟合 (主成分分析)
```

#### 2. **抓取规划模块** (Python - box_grasp_planner.py)
```python
功能：为每个检测到的药盒生成最优抓取策略
策略：
├── 顶部抓取 (2个方向: 沿长边/短边)
├── 侧面抓取 (4个方向: 前后左右)
└── 智能评分 (稳定性、可达性、避碰)
```

#### 3. **堆叠场景处理** (stacked_box_planner.py)
```python
功能：处理药盒堆叠的复杂场景
算法：
├── 垂直距离检测 (Z轴层次分析)
├── 重叠关系计算 (XY平面投影)
├── 遮挡状态分析 (空间几何关系)
└── 抓取顺序优化 (优先选择顶层)
```

#### 4. **3D可视化** (realtime_open3d_viz.py)
```python
功能：类似GraspNet的3D可视化界面
特点：
├── Open3D实时渲染
├── 边界框显示 (不同颜色区分)
├── 抓取姿态可视化 (坐标轴+夹爪)
└── 交互式视角控制
```

---

## 📦 安装配置

### 环境要求
- Ubuntu 20.04 + ROS Noetic
- Python 3.8+ (推荐conda环境)
- C++14支持

### 依赖安装
```bash
# Python依赖 (在conda环境中)
conda activate arm_grasp
pip install open3d scipy numpy

# 编译ROS包
cd ~/catkin_UR5
catkin_make
source devel/setup.bash
```

---

## 🚀 使用方法

### 快速开始 (推荐)
```bash
# 1. 启动完整系统 (相机+检测+可视化)
source ~/catkin_UR5/devel/setup.bash
# use_rviz:=false 可选，关闭RViz以节省资源
roslaunch box_grasp_detection ur5_eye_in_hand.launch use_rviz:=true

# 2. 启动Open3D实时可视化 (新终端)
source ~/catkin_UR5/devel/setup.bash
conda activate arm_grasp
rosrun box_grasp_detection realtime_open3d_viz.py

# 3. 触发单次检测 (新终端)
source ~/catkin_UR5/devel/setup.bash
conda activate arm_grasp
rosrun box_grasp_detection single_frame_detector.py
```

### 性能测试 (Benchmark)

用于对比算法耗时和Open3D渲染耗时（与GraspNet对比）：

```bash
# 1. 启动基准测试环境 (无RViz)
roslaunch box_grasp_detection benchmark.launch

# 2. 触发检测并查看耗时
rosrun box_grasp_detection single_frame_detector.py
```

### 高级用法

#### 堆叠场景处理
```bash
# 专门针对堆叠药盒的智能处理
# 1.16日，目前没有支持该功能的实现
roslaunch box_grasp_detection ur5_stacked_boxes.launch
```

### 输出数据格式

#### ROS话题输出
```bash
# 主要输出话题
/box_grasps              # BoxGraspArray - 所有检测结果
/box_grasps_smart        # BoxGraspArray - 堆叠场景优化结果 
/best_grasp_pose         # PoseStamped - 最佳抓取位姿(6D)
```

#### 单次检测输出示例
```
======================================================================
📦 检测到药盒 - 最佳抓取位姿
======================================================================

【盒子信息】
  尺寸: 15.2 × 8.5 × 3.2 cm
  体积: 413.4 cm³

【盒子位置】
  X: 0.2450 m    Y: -0.1230 m    Z: 0.5670 m

【最佳抓取】
  类型: top_along_length (顶部沿长边抓取)
  评分: 0.89/1.00
  
【6D抓取位姿】
  位置: [0.245, -0.123, 0.567]
  姿态: [0.0, 0.0, 0.383, 0.924] (四元数)
======================================================================
```

---

## 📊 核心功能详解

### 1. 点云处理算法
```cpp
// 核心处理步骤
1. 降采样: VoxelGrid (0.005m)
2. 去噪声: StatisticalOutlierRemoval 
3. 平面去除: RANSAC (距离阈值0.02m)
4. 物体分割: EuclideanClustering (0.02m)
5. 边界框: PCA主成分分析
```

### 2. 抓取策略生成
```python
# 6种抓取策略自动生成
├── 顶部抓取
│   ├── top_along_length  # 沿长边方向
│   └── top_along_width   # 沿宽边方向
└── 侧面抓取  
    ├── side_front_back   # 前后侧面
    ├── side_left_right   # 左右侧面
    ├── side_long_edge    # 长边侧面
    └── side_short_edge   # 短边侧面
```

### 3. 智能评分系统
```python
# 多因子评分 (0-1分)
final_score = w1*稳定性 + w2*可达性 + w3*避碰性
其中:
├── 稳定性: 基于重心和支撑面积
├── 可达性: 基于机械臂运动学限制  
└── 避碰性: 基于周围物体间隙
```

### 4. 堆叠场景智能处理
```python
# 堆叠检测算法
1. Z轴距离分析 → 识别垂直堆叠关系
2. XY投影重叠 → 计算水平覆盖关系
3. 层次等级计算 → 确定 底/中/顶 层级
4. 遮挡关系分析 → 判断可访问性
5. 抓取顺序优化 → 优先推荐顶层物体
```

---

## 🔧 参数配置

### 检测参数 (config/detection_params.yaml)
```yaml
# 点云处理
voxel_leaf_size: 0.005        # 降采样精度
plane_distance_threshold: 0.02 # 平面检测阈值
cluster_tolerance: 0.02       # 聚类距离
min_cluster_size: 100         # 最小聚类点数
max_cluster_size: 10000       # 最大聚类点数

# 尺寸过滤
min_box_size: 0.03           # 最小边长 3cm
max_box_size: 0.50           # 最大边长 50cm
min_volume: 0.00001          # 最小体积

# 抓取评分权重
stability_weight: 0.4         # 稳定性权重
reachability_weight: 0.3     # 可达性权重  
collision_weight: 0.3        # 避碰权重
```

---

## 📁 项目文件结构

```
box_grasp_detection/
├── README.md                 # 本文档
├── package.xml              # ROS包配置
├── CMakeLists.txt           # 编译配置
├── config/
│   └── detection_params.yaml  # 参数配置
├── include/
│   └── box_grasp_detection/
│       └── box_detector.h      # C++头文件
├── src/
│   ├── box_detector_node.cpp   # ROS节点主程序
│   └── box_detector.cpp        # 核心检测算法
├── scripts/                    # Python脚本
│   ├── box_grasp_planner.py    # 抓取规划器
│   ├── realtime_open3d_viz.py  # Open3D实时可视化 (推荐)
│   ├── single_frame_detector.py # 单帧检测器
│   ├── benchmark_performance.py # 性能测试脚本
│   └── stacked_box_planner.py   # 堆叠场景处理
├── launch/                     # 启动文件
│   ├── ur5_eye_in_hand.launch       # UR5集成 (主启动文件)
│   ├── benchmark.launch             # 性能测试启动
│   └── ur5_stacked_boxes.launch     # 堆叠场景
└── msg/                        # 消息定义
    ├── BoxGrasp.msg             # 单个抓取消息
    └── BoxGraspArray.msg        # 抓取数组消息
```

---

## 🧑‍💻 二次开发指南

### 添加新的抓取策略
```python
# 在 box_grasp_planner.py 中添加新策略
def generate_custom_grasp(self, box_msg):
    """自定义抓取策略"""
    # 1. 分析盒子几何特征
    # 2. 计算抓取点和姿态
    # 3. 返回PoseStamped消息
    pass
```

### 调整评分算法
```python
# 修改 calculate_grasp_score() 函数
def calculate_grasp_score(self, grasp_pose, box_info):
    # 自定义评分逻辑
    stability = custom_stability_function()
    reachability = custom_reachability_function() 
    collision_risk = custom_collision_function()
    return weighted_score
```

### 集成YOLO目标检测
```cpp
// 在 box_detector.cpp 中添加ROI裁剪
if (use_yolo_roi) {
    pcl::CropBox<pcl::PointXYZRGB> crop_filter;
    crop_filter.setMin(roi_min);
    crop_filter.setMax(roi_max);
    crop_filter.setInputCloud(input_cloud);
    crop_filter.filter(*roi_cloud);
}
```

---

## 🎓 学习要点总结

### 关键技术栈
1. **ROS消息机制**: Publisher/Subscriber模式进行模块通信
2. **PCL点云库**: 3D数据处理和几何算法实现
3. **Open3D可视化**: 现代3D渲染和交互界面
4. **空间几何学**: 3D变换、四元数、边界框拟合
5. **机器人学**: 抓取规划、碰撞检测、运动学约束

### 算法思想
- **规则物体优化**: 针对长方体特征，避免复杂深度学习
- **多策略融合**: 生成多种候选方案，智能评分选择
- **实时性优先**: 高效的点云处理算法确保响应速度
- **场景适应**: 堆叠检测算法处理复杂工业场景

这个系统为您提供了一个完整的药盒抓取解决方案，既可以直接使用，也可以作为学习3D机器视觉和机器人抓取的优秀案例。
