import sys
import termios
import tty
import select
import numpy as np
import teleop


class KeyboardTeleop:
    def __init__(self):
        # Store original terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)

    def __enter__(self):
        # Switch terminal to raw mode
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        # Restore terminal settings on exit
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        """Non-blocking key reader. Returns a single character or None."""
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None


def enemy_keyboard_torque(max_torque=10.0):
    """
    Reads continuous key presses (non-blocking).
    Mapping:
        q / a = joint1 + / -
        w / s = joint2 + / -
    """
    tau = np.zeros(2)

    key = teleop.get_key()
    if key is None:
        return tau

    if key == 'q':
        tau[0] = +max_torque
    elif key == 'a':
        tau[0] = -max_torque

    if key == 'w':
        tau[1] = +max_torque
    elif key == 's':
        tau[1] = -max_torque

    return tau
