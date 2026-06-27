# UR and RealSense Implementation Notes

This note was converted from the old `改动.docx` scratch document. It keeps the useful implementation reminders in text form so Git can diff it cleanly.

## RealSense D415 `get_data()`

The `get_data()` method belongs at the same indentation level as `connect()` in the RealSense D415 camera class.

It waits for a coherent pair of depth and color frames, aligns depth to the color stream, converts both frames into NumPy arrays, expands the depth image to `H x W x 1`, and returns `(color_image, depth_image)`.

```python
def get_data(self):
    frames = self.pipeline.wait_for_frames()

    align = rs.align(align_to=rs.stream.color)
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(
        aligned_depth_frame.get_data(),
        dtype=np.float32,
    )
    depth_image = np.expand_dims(depth_image, axis=2)
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image
```

The older non-aligned version was:

```python
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
```

## UR `move_j_p()`

`move_j_p()` sends a URScript program through the robot TCP socket. URScript programs must start with `def name():` and end with `end`.

The command converts RPY to a rotation vector inside URScript, then calls:

```text
movej(get_inverse_kin(p[x, y, z, rx, ry, rz]), a=..., v=..., t=..., r=...)
```

Reference implementation:

```python
def move_j_p(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
    print(f"movej_p([{tool_configuration}])")

    tcp_command = "def process():\n"
    tcp_command += "  array = rpy2rotvec([%f,%f,%f])\n" % (
        tool_configuration[3],
        tool_configuration[4],
        tool_configuration[5],
    )
    tcp_command += (
        "  movej(get_inverse_kin(p[%f,%f,%f,array[0],array[1],array[2]]),"
        "a=%f,v=%f,t=%f,r=%f)\n"
    ) % (
        tool_configuration[0],
        tool_configuration[1],
        tool_configuration[2],
        k_acc * self.joint_acc,
        k_vel * self.joint_vel,
        t,
        r,
    )
    tcp_command += "end\n"

    self.tcp_socket.send(str.encode(tcp_command))

    state_data = self.tcp_socket.recv(1500)
    actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")

    while not all([
        np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j]
        for j in range(3)
    ]):
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, "cartesian_info")
        time.sleep(0.01)

    time.sleep(1.5)
    self.tcp_socket.close()
```
