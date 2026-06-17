import socket
import time
import threading

class RobotiqHandE:
    def __init__(self):
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname, port):
        """Connect to the gripper."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        print("Connected to Hand-E gripper")

    def disconnect(self):
        """Disconnect from the gripper."""
        if self.socket:
            self.socket.close()

    def _send_command(self, data):
        """Send a command to the gripper."""
        with self.command_lock:
            self.socket.send(data)
            time.sleep(0.1)
            return self.socket.recv(1024)

    def activate(self):
        """Activate the gripper."""
        # Activation command for Hand-E
        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self._send_command(command)
        time.sleep(0.1)
        
        # Set action request
        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        self._send_command(command)
        print("Hand-E gripper activated")

    def _reset(self):
        """Reset the gripper."""
        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self._send_command(command)
        print("Hand-E gripper reset")

    def move(self, position, speed, force):
        """
        Move the gripper to a specified position.
        :param position: Position (0-255), 0 is fully open, 255 is fully closed
        :param speed: Speed (0-255)
        :param force: Force (0-255)
        """
        position = max(min(position, self._max_position), self._min_position)
        speed = max(min(speed, self._max_speed), self._min_speed)
        force = max(min(force, self._max_force), self._min_force)

        command = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, position, speed, force, 0x00])
        self._send_command(command)

    def move_and_wait_for_pos(self, position, speed, force):
        """Move to position and wait until the move is completed."""
        self.move(position, speed, force)
        time.sleep(0.5)  # Give time for the gripper to start moving
        
        # Wait until the gripper stops moving
        while self.is_moving():
            time.sleep(0.1)

    def get_current_position(self):
        """Get the current position of the gripper."""
        # Read status registers
        command = bytes([0x09, 0x03, 0x07, 0xD0, 0x00, 0x01])
        response = self._send_command(command)
        if len(response) >= 4:
            return response[3]  # Position is in the fourth byte
        return 0

    def is_moving(self):
        """Check if the gripper is currently moving."""
        # Read status registers
        command = bytes([0x09, 0x03, 0x07, 0xD0, 0x00, 0x01])
        response = self._send_command(command)
        if len(response) >= 4:
            status = response[3]
            return (status & 0x20) != 0  # Check if the gripper is in motion
        return False 