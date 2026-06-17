# socket通讯所需的包
import socket

# 定义了UR机器人的地址和端口
target_ip = ("192.168.0.1", 30003)

# 建立一个socket对象
sk = socket.socket()

# 建立连接
sk.connect(target_ip)

# 这是发送给UR机器人的一个脚本指令，注意，这是xyzrxryrz的格式
#movej(p[-0.483, -0.065, 0.297, 2.211, -2.224, -0.154],a=1, v=1, t=0, r=0)
send_data1 = '''
def svt():
    movej(p[-0.478, -0.0678, 0.336, 2.222, -2.22, -0.140],a=0.5, v=0.5, t=0, r=0)
    
end
svt()  # 调用函数
'''

# 发送指令，并将字符串转变格式
sk.send(send_data1.encode('utf8'))