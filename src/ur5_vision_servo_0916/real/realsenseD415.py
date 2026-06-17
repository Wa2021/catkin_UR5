import numpy as np
import pyrealsense2 as rs
import cv2

class Camera(object):

    def __init__(self,width=640,height=480,fps=15):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.intrinsics = None
        self.scale = None
        self.pipeline = None
        self.connect()
        # color_img, depth_img = self.get_data()
        #print(color_img, depth_img)


    def connect(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # Start streaming
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(rgb_profile)
        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print("camera depth scale:",self.scale)
        print("D415 have connected ...")

    def get_data(self):
        # 等待一帧对齐后的彩色和深度图
        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        # 获取对齐后的彩色和深度图帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 转换为 numpy 格式，注意：这里用 uint16 读取原始深度值（单位是设备内部单位）
        depth_raw = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.uint16)

        # 获取深度缩放因子（单位转换：如0.001代表毫米→米）
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        #print(f"[INFO] Depth scale: {depth_scale}")  # 推荐打印一次验证

        # 应用缩放因子：转换为“米”单位的深度图
        depth_image = depth_raw * depth_scale
        depth_image = np.expand_dims(depth_image, axis=2)  # [H, W, 1]

        # 彩色图像转换
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def plot_image(self):
        color_image,depth_image = self.get_data()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        cv2.waitKey(1)

    def get_intrinsics(self,rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera intrinsics:", raw_intrinsics)
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3) #640 480
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)

        return intrinsics
if __name__== '__main__':
    mycamera = Camera()
    # mycamera.get_data()
    mycamera.plot_image()
    # print(mycamera.intrinsics)