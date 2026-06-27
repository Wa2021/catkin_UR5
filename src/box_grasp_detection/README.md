# box_grasp_detection

UR5 眼在手场景下的盒体抓取位姿检测包。当前保留的是一条简洁的 PCA + OBB 快速路径：

- RealSense 点云输入
- C++ 盒体检测与 OBB 位姿估计
- RViz 显示点云、盒体和抓取坐标系
- 单帧脚本输出当前最佳抓取位姿

## 快速开始

启动相机、检测节点和 RViz：

```bash
cd ~/catkin_UR5
source devel/setup.bash
roslaunch box_grasp_detection ur5_eye_in_hand.launch use_rviz:=true
```

触发单次检测并定格当前结果：

```bash
cd ~/catkin_UR5
conda activate arm_grasp
source devel/setup.bash
rosrun box_grasp_detection single_frame_detector.py
```

常用参数：

```bash
roslaunch box_grasp_detection ur5_eye_in_hand.launch use_rviz:=false
rosrun box_grasp_detection single_frame_detector.py --no-viz
rosrun box_grasp_detection single_frame_detector.py --loop
```

## 主要话题

- `/camera/depth/color/points`: RealSense 点云
- `/box_grasps`: 检测到的抓取候选，类型 `box_grasp_detection/BoxGraspArray`
- `/box_markers`: RViz 显示用 marker
- `/detected_objects`: 检测出的物体点云

## 保留结构

```text
box_grasp_detection/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── box_detection.rviz
├── include/box_grasp_detection/
│   └── box_detector.h
├── launch/
│   └── ur5_eye_in_hand.launch
├── msg/
│   ├── BoxGrasp.msg
│   └── BoxGraspArray.msg
├── scripts/
│   └── single_frame_detector.py
└── src/
    ├── box_detector.cpp
    └── box_detector_node.cpp
```

已删除未接入当前快速路径的 benchmark、YOLO fusion、堆叠盒处理和独立实时 Open3D 可视化代码。
