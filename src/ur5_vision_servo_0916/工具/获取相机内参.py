import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取相机内参
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

print("fx:", intrinsics.fx)
print("fy:", intrinsics.fy)
print("cx:", intrinsics.ppx)
print("cy:", intrinsics.ppy)
print("coeffs:", intrinsics.coeffs)
