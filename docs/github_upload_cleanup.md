# GitHub 上传前清理说明

本文档记录当前 `catkin_UR5` 工作空间中哪些文件适合上传 GitHub，哪些应该忽略或删除，以及部分目录的实际用途。

## 体积异常原因

当前工作空间曾接近 998MB，主要不是源码本身导致的，而是 IDE 和构建产物：

- `src/universal_robot/ur_planning/.vscode/browse.vc.db`：约 749MB。
- `build/`：catkin 编译产物，不需要上传。
- `devel/`：catkin 开发空间产物，不需要上传。
- `.vs/`、`.idea/`、`.vscode/`：IDE 配置或索引，不建议上传。
- 各子项目里的 `.git/`：复制开源项目时带来的上游 Git 历史，不是本项目自己的历史。

## `browse.vc.db` 是什么，可以删除吗

`browse.vc.db` 是 VS Code C/C++ 插件或 Visual Studio 生成的代码浏览数据库，用于 IntelliSense、跳转定义、符号索引等本地编辑器功能。

可以删除。它不是源码，不影响 ROS/catkin 编译，也不影响程序运行。删除后编辑器需要时会重新生成。这个文件不应该上传 GitHub。

已在根目录 `.gitignore` 中忽略：

```gitignore
.vscode/
.vs/
*.vc.db
*.VC.db
```

## 建议不要上传的内容

这些内容可以删除或至少不加入 Git：

- `build/`
- `devel/`
- `install/`
- `.vs/`
- `.idea/`
- `.vscode/`
- `*.vc.db`
- `__pycache__/`
- `*.pyc`
- 运行日志、临时文件、ROS bag 数据

如果只是准备上传 GitHub，可以先执行：

```bash
rm -rf build devel install logs .vs .idea .vscode
find src -type d -name .vscode -prune -exec rm -rf {} +
```

## 嵌套 `.git` 历史

当前 `src` 里多个目录包含自己的 `.git/`，例如：

- `src/universal_robot/.git`
- `src/robotiq/.git`
- `src/robotiq_hande_ros_driver/.git`
- `src/ur5e_with_robotiq_hande/.git`
- `src/rwcLive/.git`

这些一般是复制或 clone 开源项目时带来的上游历史。若你打算把整个 `catkin_UR5` 作为一个新仓库上传，并且不使用 Git submodule，建议删除这些嵌套 `.git/`，否则 GitHub 主仓库不会正常追踪这些目录内容，或者会出现嵌套仓库提示。

删除前可以先检查：

```bash
find src -type d -name .git -prune -print
```

确认后再删除：

```bash
find src -type d -name .git -prune -exec rm -rf {} +
```

## YOLO 模型文件

原来有 4 份完全相同的 `yolov8n.pt`：

- `yolov8n.pt`
- `src/universal_robot/ur5_gripper_moveit_config/models/yolov8n.pt`
- `src/ur5_vision_servo_0916/yolov8n.pt`
- `src/ur5_vision_servo_0916/工具/yolov8n.pt`

它们的 SHA256 一致，可以只保留一份。当前统一保留到：

```text
src/models/yolov8n.pt
```

相关代码和 launch 已改为默认引用这个共享路径，同时仍保留 ROS 参数或代码变量覆盖能力。后续如果换成自训练模型，建议也放到 `src/models/`，并通过 launch 参数覆盖模型路径。

注意：`yolov8n-seg.pt` 是分割模型，不等同于 `yolov8n.pt`，当前没有合并。

## `src/robotiq` 是否没用

`src/robotiq` 不是完全没用，但是否需要保留取决于你当前使用哪套夹爪驱动。

当前发现的直接引用：

- `src/universal_robot/ur_planning/launch/start_ur5_with_cam_gripper.launch` 引用了：
- `robotiq_2f_gripper_control`
- `robotiq_2f_gripper_action_server`

因此如果你还需要运行 `start_ur5_with_cam_gripper.launch`，不能直接删除整个 `src/robotiq`。

但如果当前主要使用 Hand-E 夹爪，并通过以下 launch 启动：

- `src/universal_robot/ur_planning/launch/start_robot_only.launch`
- `src/universal_robot/ur_planning/launch/gripper_only.launch`

那它们使用的是：

```text
src/robotiq_hande_ros_driver
```

这时 `src/robotiq` 中很多旧的 2F、FT sensor、EtherCAT、Modbus、可视化包可能不是当前主流程必需。建议先保留，等确认不再运行 `start_ur5_with_cam_gripper.launch` 后再拆分或删除。

## `src/robotiq_hande_ros_driver`

这是当前 Hand-E 夹爪控制相关包。`start_robot_only.launch` 中的夹爪节点使用：

```xml
<node pkg="robotiq_hande_ros_driver" type="gripper_node.py" />
```

如果你当前真实机器人使用 URCap 控制 Hand-E，这个目录应保留。

## `src/rwcLive`

`src/rwcLive` 不是夹爪驱动，而是 Robotiq Wrist Camera 腕部相机工具。它的用途包括：

- `rwcLive.py`：Tkinter GUI，查看腕部相机实时画面。
- `robotiq_camera_node.py`：ROS 节点，从 HTTP 接口读取图像并发布到 `/robotiq_camera/image_raw`。
- `robotiq_camera.launch`：启动相机 ROS 节点。
- `test_robotiq_camera.py`：测试相机连接和保存图像。
- `使用指南.md`：中文使用说明。

文档里写到这个相机最后没用上。如果当前系统使用 RealSense D435，而不是 Robotiq Wrist Camera，可以考虑不上传 `src/rwcLive`，或者保留作为备用工具。

注意：当前 `src/rwcLive` 目录没有 `package.xml`，严格来说不是标准 catkin package。若要使用 `roslaunch rwcLive robotiq_camera.launch`，需要补齐 ROS 包元数据；否则可以直接用文件路径启动或只作为普通 Python 工具保存。

## 建议上传的核心内容

如果目标是上传当前 UR5 + Hand-E + RealSense + 抓取相关代码，建议保留：

- `.gitignore`
- `docs/github_upload_cleanup.md`
- `src/models/yolov8n.pt`
- `src/box_grasp_detection`
- `src/ur5_vision_servo_0916`
- `src/robotiq_hande_ros_driver`
- `src/ur5e_with_robotiq_hande`
- `src/universal_robot/ur_description`
- `src/universal_robot/ur_modern_driver`
- `src/universal_robot/ur5_gripper_moveit_config`
- `src/universal_robot/ur_planning`
- `src/universal_robot/ur_msgs`
- `src/universal_robot/ur_kinematics`

按需保留：

- `src/robotiq`：只有仍需旧 2F gripper control/action server 时保留。
- `src/robotiq_85_gripper`：如果当前没有使用 Robotiq 85 仿真/描述/驱动，可以不作为主流程上传。
- `src/rwcLive`：只有仍需 Robotiq Wrist Camera 时保留。

## 上传前检查命令

```bash
du -h --max-depth=2 . | sort -h | tail -40
find . -type f -size +20M -printf '%s %p\n' | sort -n
find src -type d -name .git -prune -print
git status --short
```

如果还没初始化仓库：

```bash
git init
git add .gitignore docs src
git status --short
```

确认没有 `build/`、`devel/`、`.vscode/`、`.vs/`、`*.vc.db` 后再提交。
