# 独立标定相机内参，供手眼标定流程复用。
# 相机内参是相机的固有属性；不更换镜头、不改变焦距、不剧烈磕碰相机时通常可以重复使用。

import numpy as np
import cv2
import json
import time
import os


class CameraIntrinsicsCalibrator:
    def __init__(self, pattern_size=(8, 5), square_size=0.027,
                 output_dir="calibration_images", output_file="camera_intrinsics.json"):
        """
        初始化相机内参标定器。这个过程不需要机器人。
        :param pattern_size: 标定板角点数 (宽, 高)
        :param square_size: 标定板方格大小(米)
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = self._resolve_path(output_dir)  # 用于存储拍摄的图像
        self.output_file = self._resolve_path(output_file)  # 最终结果保存的文件名

        # 创建存储图像的目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已创建目录: {self.output_dir}")

        # 初始化数据存储
        self.object_points = []  # 3D点 (世界坐标系)
        self.image_points = []  # 2D点 (图像坐标系)

        # 生成标定板3D坐标
        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.pattern_points *= self.square_size

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.script_dir, path)

    def capture_images_from_camera(self, camera_index=0, num_images=30):
        """
        从连接到PC的摄像头实时捕获图像用于标定。
        带有可视化辅助界面，实时显示角点分布。
        :param camera_index: 摄像头的索引，通常为0。
        :param num_images: 计划拍摄的图像数量。
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            return

        # 获取摄像头的分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建一个空白的“角点分布图”，用于可视化
        # 我们用一个比实际分辨率稍大的图，方便观看
        coverage_map = np.zeros((height + 100, width + 100, 3), dtype=np.uint8)
        map_h, map_w, _ = coverage_map.shape
        cv2.rectangle(coverage_map, (50, 50), (map_w - 50, map_h - 50), (255, 255, 255), 1)  # 画出相机视野框
        cv2.putText(coverage_map, "Corner Coverage Map", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        captured_corners = []  # 用来存储所有捕获到的角点
        count = 0

        print("\n=== 开始采集图像 (带可视化辅助) ===")
        print("请移动相机，确保标定板的角点能够覆盖“角点分布图”的各个区域。")
        print("特别是四个角落和边缘区域！")

        try:
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("错误：无法读取摄像头帧。")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 实时检测角点，但不保存，仅用于预览
                found, corners = cv2.findChessboardCorners(
                    gray,
                    self.pattern_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
                )

                display_frame = frame.copy()
                if found:
                    cv2.drawChessboardCorners(display_frame, self.pattern_size, corners, found)

                cv2.putText(display_frame, f"Captured: {len(captured_corners)}/{num_images}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Camera View', display_frame)
                cv2.imshow('Coverage Map', coverage_map)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    found_capture, corners_capture = cv2.findChessboardCorners(
                        gray,
                        self.pattern_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                    )

                    if found_capture:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_capture = cv2.cornerSubPix(gray, corners_capture, (11, 11), (-1, -1), criteria)
                        count += 1

                        for corner in corners_capture:
                            pt_x = int(corner[0][0]) + 50
                            pt_y = int(corner[0][1]) + 50
                            cv2.circle(coverage_map, (pt_x, pt_y), 3, (0, 255, 255), -1)

                        captured_corners.append(corners_capture)

                        img_path = os.path.join(self.output_dir, f"calib_{count:02d}.png")
                        cv2.imwrite(img_path, frame)
                        print(f"成功捕获! 已保存图像: {img_path}")

                    else:
                        print("捕获失败：未检测到角点。")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        print(f"\n图像采集完成，共捕获 {len(captured_corners)} 张有效图像。")

    def calibrate_from_images(self):
        """
        从 self.output_dir 目录下的图像进行标定。
        """
        print(f"\n正在从目录 '{self.output_dir}' 中的图像进行标定...")

        self.object_points = []
        self.image_points = []
        image_files = sorted(
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

        if not image_files:
            print("错误：在目录下未找到任何图像文件。")
            return

        img_size = None
        for fname in image_files:
            img = cv2.imread(fname)
            if img is None:
                print(f"处理图像: {os.path.basename(fname)} ... 失败，无法读取文件")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_size is None:
                img_size = gray.shape[::-1]  # (width, height)

            found, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                # 亚像素优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                self.object_points.append(self.pattern_points)
                self.image_points.append(corners_subpix)
                print(f"处理图像: {os.path.basename(fname)} ... 成功")
            else:
                print(f"处理图像: {os.path.basename(fname)} ... 失败，未找到角点")

        if len(self.object_points) < 5:
            print(f"错误：有效的图像数量 ({len(self.object_points)}) 太少，无法进行标定。")
            return

        print("\n正在计算相机内参和畸变系数...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, img_size, None, None
        )

        if not ret:
            print("相机标定失败！")
            return

        print("\n相机标定成功！")
        print("相机内参矩阵 (Camera Matrix):")
        print(camera_matrix)
        print("\n畸变系数 (Distortion Coefficients):")
        print(dist_coeffs)

        # 计算重投影误差
        mean_error = 0
        for i in range(len(self.object_points)):
            img_points2, _ = cv2.projectPoints(self.object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(self.image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error

        reprojection_error = mean_error / len(self.object_points)
        print(f"\n平均重投影误差 (Reprojection Error): {reprojection_error:.4f} 像素")
        if reprojection_error > 1.0:
            print("警告：重投影误差较大，标定结果可能不准确。请尝试使用更多或更高质量的图像。")

        # 保存结果
        self.save_intrinsics(camera_matrix, dist_coeffs, reprojection_error, img_size)

    def save_intrinsics(self, camera_matrix, dist_coeffs, error, image_size):
        """
        将标定结果保存到JSON文件。
        """
        intrinsics_data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "reprojection_error": error,
            "image_size": list(image_size),
            "pattern_size": list(self.pattern_size),
            "square_size": self.square_size,
            "num_images": len(self.object_points)
        }

        with open(self.output_file, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)

        print(f"\n标定结果已成功保存到: {self.output_file}")


def main():
    print("=== 相机内参标定程序 ===")
    print("步骤1: 采集图像。请手持棋盘格，在摄像头前从不同角度、不同距离展示。")
    print("         确保棋盘格出现在画面的中心、四周角落和边缘。")
    print("步骤2: 程序会自动从采集的图像中计算内参。")

    calibrator = CameraIntrinsicsCalibrator(
        pattern_size=(8, 5),
        square_size=0.027
    )

    # 模式一：实时从摄像头捕获图像
    calibrator.capture_images_from_camera(camera_index=0, num_images=30)

    # 模式二：从已经拍摄好的图像文件夹进行标定
    # 如果你已经有了图像，可以注释掉上面的 capture_images...，然后直接运行下面的 calibrate...
    calibrator.calibrate_from_images()


if __name__ == '__main__':
    main()
