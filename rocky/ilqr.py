# rocky/ilqr.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from rocky.dynamics import (
    ArmParams,
    linearize_dynamics,
    forward_kinematics,
    jacobian,
    discrete_dynamics,
)


# ---------------- Cost definitions ----------------

@dataclass
class CostParams:
    # State and input penalties
    Q: np.ndarray = field(default_factory=lambda: np.diag([2.0, 2.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: 0.01 * np.eye(2))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([40.0, 40.0, 1.0, 1.0]))

    # Task-space weights
    target_weight: float = 200.0  # move EE to target
    avoid_weight: float = 150.0   # avoid enemy EE
    avoid_sigma: float = 0.10     # “radius” of avoidance (m)

    # Optional state goal if you want posture regularization
    x_goal: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class ILQRResult:
    x_trj: np.ndarray
    u_trj: np.ndarray
    cost: float
    iterations: int
    converged: bool


# ---------------- Cost helpers ----------------

def _target_cost_terms(
    q: np.ndarray,
    target_pos: np.ndarray,
    params: ArmParams,
    weight: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Quadratic cost in end-effector space: 0.5 * w * ||p(q) - target||^2
    Return cost, gradient wrt q, and Hessian wrt q.
    """
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
    """
    Gaussian barrier around the enemy EE:
      cost ~ weight * exp(-0.5 * ||p(q) - enemy_pos||^2 / sigma^2)
    High cost near enemy, ~0 far away.
    """
    p = forward_kinematics(q, params)
    J = jacobian(q, params)
    rel = p - enemy_pos
    r2 = float(rel.T @ rel)
    r = np.linalg.norm(rel)
    if r2 < 1e-10 or sigma <= 0.0 or weight <= 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))

    scale = weight * np.exp(-0.5 * r2 / (sigma**2))

    # gradient wrt p
    grad_p = -(scale / (sigma**2)) * rel
    # approximate Hessian wrt p
    hess_p = scale * (
        (np.outer(rel, rel) / (sigma**4)) - (np.eye(2) / (sigma**2))
    )
    # Extra hard penalty inside radius r_min
    r_min = 0.80
    if r < r_min:
        # barrier_strength can be another hyperparameter
        barrier_strength = 500.0
        dr = r_min - r
        # d(0.5 * k * dr^2)/dp = k * dr * (-rel / r)
        grad_p += barrier_strength * dr * (-rel / r)
        # You can approximate the Hessian or just add a diagonal term
        hess_p += barrier_strength * np.eye(2)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    l, l_x, l_u, l_xx, l_uu, l_ux for one time step.
    """
    nx, nu = x.size, u.size
    l   = 0.0
    l_x = np.zeros(nx)
    l_u = np.zeros(nu)
    l_xx = np.zeros((nx, nx))
    l_uu = np.zeros((nu, nu))
    l_ux = np.zeros((nu, nx))

    # 1) Quadratic state/control penalty
    dx = x - cost.x_goal
    l += 0.5 * dx.T @ cost.Q @ dx + 0.5 * u.T @ cost.R @ u
    l_x += cost.Q @ dx
    l_u += cost.R @ u
    l_xx += cost.Q
    l_uu += cost.R

    # 2) Target tracking in task space
    c_target, g_q, H_q = _target_cost_terms(
        x[:2], target_pos, params, cost.target_weight
    )
    l += c_target
    l_x[:2] += g_q
    l_xx[:2, :2] += H_q

    # 3) Enemy avoidance in task space
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
    """
    Terminal cost at final time:
      0.5 (x - x_goal)^T Qf (x - x_goal)
      + task-space target tracking
      + enemy avoidance.
    """
    l   = 0.5 * (x - cost.x_goal).T @ cost.Qf @ (x - cost.x_goal)
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


# ---------------- iLQR controller ----------------

class ILQRController:
    def __init__(
        self,
        horizon: int,
        dt: float,
        params: ArmParams | None = None,
        cost: CostParams | None = None,
        max_iter: int = 60,
        tol: float = 1e-3,
        reg: float = 1e-4,
        verbose: bool = False,
    ):
        self.N = horizon
        self.dt = dt
        self.params = params or ArmParams()
        self.cost = cost or CostParams()
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.alpha_list = [1.0, 0.5, 0.25, 0.1, 0.05]
        self.verbose = verbose

    def __repr__(self) -> str:
        p, c = self.params, self.cost
        arr = lambda a: np.array2string(np.asarray(a), precision=3, floatmode="fixed")
        lines = [
            "ILQRController(",
            f"  horizon={self.N}, dt={self.dt}, max_iter={self.max_iter}, tol={self.tol}, reg={self.reg}, verbose={self.verbose}",
            f"  params: l1={p.l1}, l2={p.l2}, m1={p.m1}, m2={p.m2}, I1={p.I1}, I2={p.I2}, damping={arr(p.damping)}, gravity={p.gravity}, torque_limit={arr(p.torque_limit)}",
            f"  cost: Q={arr(c.Q)}, R={arr(c.R)}, Qf={arr(c.Qf)}, target_w={c.target_weight}, avoid_w={c.avoid_weight}, avoid_sigma={c.avoid_sigma}, x_goal={arr(c.x_goal)}",
            f"  alphas={self.alpha_list}",
            ")",
        ]
        return "\n".join(lines)

    def solve(
        self,
        x0: np.ndarray,
        target_pos: np.ndarray,
        enemy_pos: np.ndarray,
        u_init: Optional[np.ndarray] = None,
    ) -> ILQRResult:
        """
        Solve a finite-horizon optimal control problem from initial state x0.
        """
        nx, nu = 4, 2

        # Initial control guess
        U = np.zeros((self.N, nu)) if u_init is None else np.array(u_init, copy=True)
        X = np.zeros((self.N + 1, nx))
        X[0] = x0

        # Forward rollout helper
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
        if self.verbose:
            print(f"[iLQR] init cost {J:.4f}")

        for it in range(self.max_iter):
            # --- Backward pass ---
            V_x, V_xx = terminal_cost(
                X[-1], self.cost, self.params, target_pos, enemy_pos
            )[1:]
            k_seq = np.zeros_like(U)         # feedforward
            K_seq = np.zeros((self.N, 2, 4)) # feedback
            diverged = False

            for k in reversed(range(self.N)):
                xk = X[k]
                uk = U[k]
                l, l_x, l_u, l_xx, l_uu, l_ux = running_cost(
                    xk, uk, self.cost, self.params, target_pos, enemy_pos
                )
                f_x, f_u = linearize_dynamics(xk, uk, self.dt, self.params)

                Q_x  = l_x  + f_x.T @ V_x
                Q_u  = l_u  + f_u.T @ V_x
                Q_xx = l_xx + f_x.T @ V_xx @ f_x
                Q_ux = l_ux + f_u.T @ V_xx @ f_x
                Q_uu = l_uu + f_u.T @ V_xx @ f_u + self.reg * np.eye(2)


                # regularize and symmetrize Q_uu, V_xx
                Q_uu = 0.5 * (Q_uu + Q_uu.T) + self.reg * np.eye(2)

                try:
                    Q_uu_inv = np.linalg.inv(Q_uu)
                except np.linalg.LinAlgError:
                    diverged = True
                    break

                k_ff = -Q_uu_inv @ Q_u
                K_fb = -Q_uu_inv @ Q_ux

                k_seq[k] = k_ff
                K_seq[k] = K_fb

                V_x  = Q_x + K_fb.T @ Q_uu @ k_ff + K_fb.T @ Q_u + Q_ux.T @ k_ff
                V_xx = Q_xx + K_fb.T @ Q_uu @ K_fb + K_fb.T @ Q_ux + Q_ux.T @ K_fb
                
                V_xx = 0.5 * (V_xx + V_xx.T)

            if diverged:
                if self.verbose:
                    print(f"[iLQR] backward diverged at iter {it}")
                break

            # --- Forward line search ---
            improved = False
            for alpha in self.alpha_list:
                U_new = np.zeros_like(U)
                X_new = np.zeros_like(X)
                X_new[0] = x0
                J_new = 0.0

                for k in range(self.N):
                    dx = X_new[k] - X[k]
                    U_new[k] = self.params.clip_u(
                        U[k] + alpha * k_seq[k] + K_seq[k] @ dx
                    )
                    l, _, _, _, _, _ = running_cost(
                        X_new[k], U_new[k], self.cost, self.params, target_pos, enemy_pos
                    )
                    J_new += l
                    X_new[k + 1] = discrete_dynamics(
                        X_new[k], U_new[k], self.dt, self.params
                    )

                lf, _, _ = terminal_cost(
                    X_new[-1], self.cost, self.params, target_pos, enemy_pos
                )
                J_new += lf

                if J_new < J:
                    U, X, J = U_new, X_new, J_new
                    improved = True
                    if self.verbose:
                        print(f"[iLQR] iter {it} alpha {alpha:.2f} -> cost {J:.4f}")
                    break

            if not improved:
                if self.verbose:
                    print(f"[iLQR] iter {it} no improvement (cost {J_new:.4f})")
                return ILQRResult(X, U, J, it + 1, improved)

            if abs(J - J_new) < self.tol:
                if self.verbose:
                    print(f"[iLQR] converged at iter {it} cost {J:.4f}")
                return ILQRResult(X, U, J, it + 1, True)

        return ILQRResult(X, U, J, self.max_iter, False)

    # -------- MPC wrapper --------
    def step_mpc(
        self,
        x_current: np.ndarray,
        target_pos: np.ndarray,
        enemy_pos: np.ndarray,
        prev_u_trj: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ILQRResult]:
        """
        Compute one MPC step:
          - warm-start with prev_u_trj shifted,
          - solve iLQR,
          - return predicted next state (model) and full ILQRResult.
        """
        u_init = self._shift_controls(prev_u_trj)
        result = self.solve(x_current, target_pos, enemy_pos, u_init=u_init)

        # If solve failed or cost is not finite, fall back safely
        if (not result.converged) or (not np.isfinite(result.cost)):
            if self.verbose:
                print("[MPC] iLQR failed; using zero torque.")
            u0 = np.zeros(2)
        else:
            u0 = result.u_trj[0]

        x_next = discrete_dynamics(x_current, u0, self.dt, self.params)
        if self.verbose:
            print(f"[MPC] x={x_current}, target={target_pos}, u0={u0}, cost={result.cost}")
        return x_next, result

    @staticmethod
    def _shift_controls(u_trj: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if u_trj is None:
            return None
        return np.vstack([u_trj[1:], u_trj[-1]])
