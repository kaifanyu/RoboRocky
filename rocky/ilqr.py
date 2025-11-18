"""
Simple iLQR + MPC controller for the 2-DOF boxing robot.

State  x = [q1, q2, q1_dot, q2_dot]
Input  u = [tau_shoulder, tau_elbow]
Forward kinematics are limited to the planar X-Z motion described by the URDF
(joints rotate about +y). Dynamics are a light-weight planar 2-link model,
good enough for controller prototyping without running Drake inside the loop.

Typical usage:
    from rocky.ilqr import ArmParams, CostParams, ILQRController
    params = ArmParams()
    cost   = CostParams()
    ilqr   = ILQRController(horizon=40, dt=0.02, params=params, cost=cost)
    result = ilqr.solve(x0, target_pos=np.array([0.9, 0.8]),
                        enemy_pos=np.array([0.9, 0.8]))
    # For MPC, repeatedly call `step_mpc`, feeding back the latest state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np


class ArmParams:
    # Geometry and inertias pulled from rocky.urdf
    l1: float = 0.50
    l2: float = 0.50
    m1: float = 0.60
    m2: float = 0.50
    I1: float = 0.0020
    I2: float = 0.0020
    damping: np.ndarray = np.array([0.1, 0.1], dtype=float)
    gravity: float = 9.81
    torque_limit: np.ndarray = np.array([40.0, 30.0], dtype=float)

    # Meant to clip torque limits
    def clip_u(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, -self.torque_limit, self.torque_limit)


class CostParams:
    Q: np.ndarray = np.diag([5.0, 5.0, 0.1, 0.1])
    R: np.ndarray = 0.01 * np.eye(2)
    Qf: np.ndarray = np.diag([40.0, 40.0, 1.0, 1.0])
    target_weight: float = 200.0
    avoid_weight: float = 150.0
    avoid_sigma: float = 0.10  # meters
    x_goal: np.ndarray = np.zeros(4)


@dataclass
class ILQRResult:
    x_trj: np.ndarray
    u_trj: np.ndarray
    cost: float
    iterations: int
    converged: bool


def forward_kinematics(q: np.ndarray, params: ArmParams) -> np.ndarray:
    """Return end-effector position in the plane (x, z)."""
    q1, q2 = q
    l1, l2 = params.l1, params.l2
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    z = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, z])


def jacobian(q: np.ndarray, params: ArmParams) -> np.ndarray:
    """2x2 planar Jacobian for the glove."""
    q1, q2 = q
    l1, l2 = params.l1, params.l2
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    J = np.array(
        [
            [-l1 * s1 - l2 * s12, -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12],
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
    g1 = (m1 * lc1 + m2 * l1) * g * np.cos(q1) + m2 * lc2 * g * np.cos(q1 + q2)
    g2 = m2 * lc2 * g * np.cos(q1 + q2)
    return np.array([g1, g2])


def continuous_dynamics(x: np.ndarray, u: np.ndarray, params: ArmParams) -> np.ndarray:
    q = x[:2]
    dq = x[2:]
    u_clipped = params.clip_u(u)
    M = mass_matrix(q, params)
    C = coriolis_matrix(q, dq, params)
    G = gravity_torque(q, params)
    damping = params.damping * dq
    ddq = np.linalg.solve(M, u_clipped - C @ dq - G - damping)
    return np.hstack([dq, ddq])


def discrete_dynamics(
    x: np.ndarray, u: np.ndarray, dt: float, params: ArmParams
) -> np.ndarray:
    return x + dt * continuous_dynamics(x, u, params)


def linearize_dynamics(
    x: np.ndarray, u: np.ndarray, dt: float, params: ArmParams, eps: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobians of the discrete dynamics."""
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


def _target_cost_terms(
    q: np.ndarray,
    target_pos: np.ndarray,
    params: ArmParams,
    weight: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    p = forward_kinematics(q, params)
    J = jacobian(q, params)
    err = p - target_pos
    cost = 0.5 * weight * err.T @ err
    grad_q = weight * J.T @ err
    hess_q = weight * (J.T @ J)
    return cost, grad_q, hess_q


def _avoidance_terms(
    q: np.ndarray,
    enemy_pos: np.ndarray,
    params: ArmParams,
    weight: float,
    sigma: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    p = forward_kinematics(q, params)
    J = jacobian(q, params)
    rel = p - enemy_pos
    r2 = float(rel.T @ rel)
    if r2 < 1e-10 or sigma <= 0.0 or weight <= 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    scale = weight * np.exp(-0.5 * r2 / (sigma**2))
    grad_p = -(scale / (sigma**2)) * rel
    # Gauss-barrier Hessian approximation in task space
    hess_p = scale * (
        (np.outer(rel, rel) / (sigma**4)) - (np.eye(2) / (sigma**2))
    )
    grad_q = J.T @ grad_p
    hess_q = J.T @ hess_p @ J
    return scale, grad_q, hess_q


def running_cost(
    x: np.ndarray,
    u: np.ndarray,
    cost: CostParams,
    params: ArmParams,
    target_pos: np.ndarray,
    enemy_pos: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return l, l_x, l_u, l_xx, l_uu, l_ux for a single time step."""
    nx, nu = x.size, u.size
    l = 0.0
    l_x = np.zeros(nx)
    l_u = np.zeros(nu)
    l_xx = np.zeros((nx, nx))
    l_uu = np.zeros((nu, nu))
    l_ux = np.zeros((nu, nx))

    # State/control quadratic penalties
    # Quadratic Cost Function
    dx = x - cost.x_goal
    l += 0.5 * dx.T @ cost.Q @ dx + 0.5 * u.T @ cost.R @ u
    # Gradients 
    l_x += cost.Q @ dx
    l_u += cost.R @ u
    # Hessians
    l_xx += cost.Q
    l_uu += cost.R

    # Task-space target tracking
    c_target, g_q, H_q = _target_cost_terms(
        x[:2], target_pos, params, cost.target_weight
    )
    l += c_target
    l_x[:2] += g_q
    l_xx[:2, :2] += H_q

    # Obstacle avoidance (enemy end-effector)
    c_avoid, g_q, H_q = _avoidance_terms(
        x[:2], enemy_pos, params, cost.avoid_weight, cost.avoid_sigma
    )
    l += c_avoid
    l_x[:2] += g_q
    l_xx[:2, :2] += H_q

    return l, l_x, l_u, l_xx, l_uu, l_ux


def terminal_cost(
    x: np.ndarray,
    cost: CostParams,
    params: ArmParams,
    target_pos: np.ndarray,
    enemy_pos: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    l = 0.5 * (x - cost.x_goal).T @ cost.Qf @ (x - cost.x_goal)
    l_x = cost.Qf @ (x - cost.x_goal)
    l_xx = cost.Qf.copy()

    c_target, g_q, H_q = _target_cost_terms(
        x[:2], target_pos, params, cost.target_weight
    )
    l += c_target
    l_x[:2] += g_q
    l_xx[:2, :2] += H_q

    c_avoid, g_q, H_q = _avoidance_terms(
        x[:2], enemy_pos, params, cost.avoid_weight, cost.avoid_sigma
    )
    l += c_avoid
    l_x[:2] += g_q
    l_xx[:2, :2] += H_q

    return l, l_x, l_xx


class ILQRController:
    def __init__(
        self,
        horizon: int,
        dt: float,
        params: ArmParams | None = None,
        cost: CostParams | None = None,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg: float = 1e-6,
    ):
        self.N = horizon
        self.dt = dt
        self.params = params or ArmParams()
        self.cost = cost or CostParams()
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.alpha_list = [1.0, 0.5, 0.25, 0.1, 0.05]

    # Main iLQR solve -----------------------------------------------------
    def solve(
        self,
        x0: np.ndarray,
        target_pos: np.ndarray,
        enemy_pos: np.ndarray,
        u_init: Optional[np.ndarray] = None,
    ) -> ILQRResult:
        nx, nu = 4, 2
        # Give an input trajectory guess
        U = np.zeros((self.N, nu)) if u_init is None else np.array(u_init, copy=True)
        # X Trajectory (N+1 for final state where there are no inputs)
        X = np.zeros((self.N + 1, nx))
        X[0] = x0

        def rollout(U_try: np.ndarray) -> Tuple[np.ndarray, float]:
            x_trj = np.zeros_like(X)
            x_trj[0] = x0
            total_cost = 0.0
            for k in range(self.N):
                u_k = self.params.clip_u(U_try[k])
                l, _, _, _, _, _ = running_cost(
                    x_trj[k], u_k, self.cost, self.params, target_pos, enemy_pos
                )
                total_cost += l
                x_trj[k + 1] = discrete_dynamics(
                    x_trj[k], u_k, self.dt, self.params
                )
            lf, _, _ = terminal_cost(
                x_trj[-1], self.cost, self.params, target_pos, enemy_pos
            )
            total_cost += lf
            return x_trj, total_cost

        X, J = rollout(U)

        for it in range(self.max_iter):
            # Backward pass
            V_x, V_xx = terminal_cost(
                X[-1], self.cost, self.params, target_pos, enemy_pos
            )[1:]
            k_seq = np.zeros((self.N, nu))
            K_seq = np.zeros((self.N, nu, nx))
            diverged = False

            for k in reversed(range(self.N)):
                xk = X[k]
                uk = U[k]
                l, l_x, l_u, l_xx, l_uu, l_ux = running_cost(
                    xk, uk, self.cost, self.params, target_pos, enemy_pos
                )
                f_x, f_u = linearize_dynamics(xk, uk, self.dt, self.params)

                Q_x = l_x + f_x.T @ V_x
                Q_u = l_u + f_u.T @ V_x
                Q_xx = l_xx + f_x.T @ V_xx @ f_x
                Q_ux = l_ux + f_u.T @ V_xx @ f_x
                Q_uu = l_uu + f_u.T @ V_xx @ f_u + self.reg * np.eye(nu)

                try:
                    Q_uu_inv = np.linalg.inv(Q_uu)
                except np.linalg.LinAlgError:
                    diverged = True
                    break

                k_ff = -Q_uu_inv @ Q_u
                K_fb = -Q_uu_inv @ Q_ux

                k_seq[k] = k_ff
                K_seq[k] = K_fb

                V_x = Q_x + K_fb.T @ Q_uu @ k_ff + K_fb.T @ Q_u + Q_ux.T @ k_ff
                V_xx = Q_xx + K_fb.T @ Q_uu @ K_fb + K_fb.T @ Q_ux + Q_ux.T @ K_fb

            if diverged:
                break

            # Forward line search
            improved = False
            for alpha in self.alpha_list:
                U_new = np.zeros_like(U)
                X_new = np.zeros_like(X)
                X_new[0] = x0
                cost_new = 0.0
                for k in range(self.N):
                    dx = X_new[k] - X[k]
                    U_new[k] = self.params.clip_u(
                        U[k] + alpha * k_seq[k] + K_seq[k] @ dx
                    )
                    l, _, _, _, _, _ = running_cost(
                        X_new[k], U_new[k], self.cost, self.params, target_pos, enemy_pos
                    )
                    cost_new += l
                    X_new[k + 1] = discrete_dynamics(
                        X_new[k], U_new[k], self.dt, self.params
                    )
                lf, _, _ = terminal_cost(
                    X_new[-1], self.cost, self.params, target_pos, enemy_pos
                )
                cost_new += lf

                if cost_new < J:
                    U = U_new
                    X = X_new
                    J = cost_new
                    improved = True
                    break

            if not improved or abs(J - cost_new) < self.tol:
                return ILQRResult(X, U, J, it + 1, improved)

        return ILQRResult(X, U, J, self.max_iter, False)

    # MPC loop ------------------------------------------------------------
    def step_mpc(
        self,
        x_current: np.ndarray,
        target_pos: np.ndarray,
        enemy_pos: np.ndarray,
        prev_u_trj: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ILQRResult]:
        """Compute one MPC step and roll the dynamics forward by dt."""
        u_init = self._shift_controls(prev_u_trj)
        result = self.solve(x_current, target_pos, enemy_pos, u_init=u_init)
        u0 = result.u_trj[0]
        x_next = discrete_dynamics(x_current, u0, self.dt, self.params)
        return x_next, result

    @staticmethod
    def _shift_controls(u_trj: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if u_trj is None:
            return None
        shifted = np.vstack([u_trj[1:], u_trj[-1]])
        return shifted
