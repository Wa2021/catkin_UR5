# Single box grasp pose with score
Header header

# Box dimensions (in meters)
float64 length  # x-axis
float64 width   # y-axis
float64 height  # z-axis

# Box center pose in the point cloud frame
geometry_msgs/Pose box_pose

# Grasp pose (position and orientation for end-effector)
geometry_msgs/PoseStamped grasp_pose

# Grasp quality score (0-1, higher is better)
float64 score

# Grasp type: "top", "side_long", "side_short"
string grasp_type

# Object point cloud
sensor_msgs/PointCloud2 object_cloud
