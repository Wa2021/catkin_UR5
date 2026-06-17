'''
URrobot中也有这个函数
movej_p([(-0.5566893888848234, -0.08242294655147536, 0.1272106288956553, 3.0706061790195527, 0.06793653319256784, -1.5790871351208524)])
算出来的位置是上面的那个，我在示教器上读出来的实际位置（xyz,rx,ry,rz）大概是

-0.552，-0.019，0.10555,2.235，-2.209，-0.057
[输出] xyzrpy: [-0.552, -0.019, 0.10555, 3.117055726745874, 0.026752726823974785, -1.5594235410764643]
'''

import numpy as np
import cv2

def rotvec_to_rpy(x, y, z, rx, ry, rz):

    rot_vec = np.array([rx, ry, rz], dtype=np.float64).reshape(3, 1)
    R_mat, _ = cv2.Rodrigues(rot_vec)
    pitch = np.arcsin(-R_mat[2, 0])

    if np.isclose(np.cos(pitch), 0.0, atol=1e-6):
        roll = 0.0
        yaw = np.arctan2(-R_mat[0, 1], R_mat[1, 1])
    else:
        roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])
        yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])

    return x, y, z, roll, pitch, yaw


pose_xyzrxryrz = [-0.478, -0.0678, 0.336, 2.222, -2.22, -0.140]
pose_xyzrpy =rotvec_to_rpy(*pose_xyzrxryrz)

print("[输入] xyzrxryrz:", pose_xyzrxryrz)
print("[输出] xyzrpy:", pose_xyzrpy)
