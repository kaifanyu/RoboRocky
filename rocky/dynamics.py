# rocky/dynamics.py

import numpy as np
from typing import Tuple
from dataclasses import dataclass, field

@dataclass
class ArmParams:
    # Geometry and inertias
    l1: float = 0.50
    l2: float = 0.50
    m1: float = 0.60
    m2: float = 0.50
    I1: float = 0.0020
    I2: float = 0.0020

    damping: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1], dtype=float))
    # We treat g as positive magnitude here, direction handled in logic
    gravity: float = 0

    torque_limit: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0], dtype=float))

    def clip_u(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, -self.torque_limit, self.torque_limit)

def forward_kinematics(q: np.ndarray, params: ArmParams) -> np.ndarray:
    q1, q2 = q
    l1, l2 = params.l1, params.l2
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    z = -l1 * np.sin(q1) - l2 * np.sin(q1 + q2) 
    return np.array([x, z])

def jacobian(q: np.ndarray, params: ArmParams) -> np.ndarray:
    q1, q2 = q
    l1, l2 = params.l1, params.l2
    s1  = np.sin(q1)
    c1  = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    J = np.array(
        [
            [-l1 * s1 - l2 * s12,   -l2 * s12],    # d x / d q
            [-l1 * c1 - l2 * c12,   -l2 * c12],    # d z / d q
        ]
    )
    return J

def mass_matrix(q: np.ndarray, params: ArmParams) -> np.ndarray:
    q1, q2 = q
    l1, l2 = params.l1, params.l2
    m1, m2 = params.m1, params.m2
    I1, I2 = params.I1, params.I2
    lc1 = 0.5 * l1
    lc2 = 0.5 * l2
    c2 = np.cos(q2)

    m11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    m12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    m22 = I2 + m2 * lc2**2
    return np.array([[m11, m12], [m12, m22]])

def coriolis_matrix(q: np.ndarray, dq: np.ndarray, params: ArmParams) -> np.ndarray:
    q2 = q[1]
    dq1, dq2 = dq
    h = -params.m2 * params.l1 * 0.5 * params.l2 * np.sin(q2)
    return np.array([[h * dq2, h * (dq1 + dq2)], [-h * dq1, 0.0]])

def gravity_torque(q: np.ndarray, params: ArmParams) -> np.ndarray:
    q1, q2 = q
    g = params.gravity
    l1, l2 = params.l1, params.l2
    m1, m2 = params.m1, params.m2
    lc1 = 0.5 * l1
    lc2 = 0.5 * l2
    
    # FIXED: Gravity signs flipped to match z = -sin(q) convention.
    # Because z is negative down, and q increases down, G must be negative
    # relative to the restoring torque.
    g1 = -((m1 * lc1 + m2 * l1) * g * np.cos(q1) + m2 * lc2 * g * np.cos(q1 + q2))
    g2 = -(m2 * lc2 * g * np.cos(q1 + q2))
    return np.array([g1, g2])

def continuous_dynamics(x: np.ndarray, u: np.ndarray, params: ArmParams) -> np.ndarray:
    q = x[:2]
    dq = x[2:]
    u_clipped = params.clip_u(u)
    M = mass_matrix(q, params)
    C = coriolis_matrix(q, dq, params)
    G = gravity_torque(q, params)
    damping = params.damping * dq
    
    # Equation: M*ddq + C*dq + G + damping = u
    # Therefore: M*ddq = u - C*dq - G - damping
    ddq = np.linalg.solve(M, u_clipped - C @ dq - G - damping)
    return np.hstack([dq, ddq])

def discrete_dynamics(
    x: np.ndarray, u: np.ndarray, dt: float, params: ArmParams
) -> np.ndarray:
    # Simple Euler integration
    return x + dt * continuous_dynamics(x, u, params)

def linearize_dynamics(
    x: np.ndarray, u: np.ndarray, dt: float, params: ArmParams, eps: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    nx = x.shape[0]
    nu = u.shape[0]
    fx = np.zeros((nx, nx))
    fu = np.zeros((nx, nu))
    base = discrete_dynamics(x, u, dt, params)

    for i in range(nx):
        dx = np.zeros_like(x)
        dx[i] = eps
        fx[:, i] = (discrete_dynamics(x + dx, u, dt, params) - base) / eps

    for j in range(nu):
        du = np.zeros_like(u)
        du[j] = eps
        fu[:, j] = (discrete_dynamics(x, u + du, dt, params) - base) / eps

    return fx, fu