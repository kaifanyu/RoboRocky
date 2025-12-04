"""
Direct-collocation controller for the planar arm.

Rewritten to force use of pydrake.systems.trajectory_optimization.DirectCollocation,
avoid Context constructor mismatches, and ensure the system is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from rocky.dynamics import ArmParams, continuous_dynamics
from rocky.ilqr import CostParams

# ================================================================
# Drake Imports (clean + deterministic)
# ================================================================
try:
    from pydrake.all import LeafSystem, PiecewisePolynomial, Solve
    from pydrake.symbolic import cos as sym_cos, sin as sym_sin
except Exception as exc:
    raise ImportError("Drake must be installed") from exc

# Force the trajectory_optimization version.
try:
    from pydrake.systems.trajectory_optimization import DirectCollocation
except Exception:
    # If unavailable, fallback to planning version (rare)
    from pydrake.planning import DirectCollocation


# ================================================================
# Helpers
# ================================================================

def _quad_form(vec: np.ndarray, mat: np.ndarray) -> float:
    return sum(mat[i, j] * vec[i] * vec[j] for i in range(len(vec)) for j in range(len(vec)))


def _ee_pos_expr(q: np.ndarray, params: ArmParams):
    q1, q2 = q
    l1, l2 = params.l1, params.l2
    x = l1 * sym_cos(q1) + l2 * sym_cos(q1 + q2)
    z = -l1 * sym_sin(q1) - l2 * sym_sin(q1 + q2)
    return x, z


# ================================================================
# Planar Arm System (clean + guaranteed valid)
# ================================================================

class _PlanarArmSystem(LeafSystem):
    """Continuous-time Drake system wrapper around ArmParams dynamics."""

    def __init__(self, params: ArmParams):
        super().__init__()
        self._params = params

        # State x = [q1, q2, dq1, dq2]
        self.DeclareContinuousState(4)

        # Input u = [u1, u2]
        self.DeclareVectorInputPort("u", 2)

        # Output (passthrough)
        self.DeclareVectorOutputPort(
            "x",
            4,
            lambda context, output: output.SetFromVector(
                context.get_continuous_state_vector().CopyToVector()
            ),
        )

    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().CopyToVector()
        u = self.get_input_port(0).Eval(context)
        xdot = np.asarray(continuous_dynamics(x, u, self._params), dtype=float)

        derivatives.get_mutable_vector().SetFromVector(xdot)


# ================================================================
# Result Dataclass
# ================================================================

@dataclass
class CollocationResult:
    x_trj: np.ndarray
    u_trj: np.ndarray
    cost: float
    success: bool
    message: str = ""


# ================================================================
# Collocation Controller
# ================================================================

class CollocationController:
    def __init__(
        self,
        horizon: int,
        dt: float,
        params: Optional[ArmParams] = None,
        cost: Optional[CostParams] = None,
        verbose: bool = False,
    ):
        self.N = int(horizon)
        self.dt = float(dt)
        self.params = params or ArmParams()
        self.cost = cost or CostParams()
        self.verbose = verbose

    # ------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        target_pos: np.ndarray,
        enemy_pos: Optional[np.ndarray] = None,
    ) -> CollocationResult:

        # Create the system (LeafSystem)
        system = _PlanarArmSystem(self.params)

        # IMPORTANT: Let DirectCollocation construct its own Context internally
        try:
            dircol = DirectCollocation(
                system,
                self.N + 1,    # num time samples
                self.dt,       # min dt
                self.dt,       # max dt
                0              # input port index
            )
        except TypeError as exc:
            return CollocationResult(
                x_trj=np.zeros((self.N + 1, 4)),
                u_trj=np.zeros((self.N, 2)),
                cost=np.inf,
                success=False,
                message=f"DirectCollocation constructor failed:\n{exc}",
            )

        dircol.AddEqualTimeIntervalsConstraints()

        # Initial state
        dircol.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())

        # Input limits
        u_lim = self.params.torque_limit
        for k in range(self.N):
            dircol.AddBoundingBoxConstraint(-u_lim, u_lim, dircol.input(k))

        # Build costs ---------------------------------------------
        x = dircol.state()
        u = dircol.input()

        q = x[:2]
        ee_x, ee_z = _ee_pos_expr(q, self.params)

        dx = x - self.cost.x_goal
        cost_expr = 0.5 * _quad_form(dx, self.cost.Q)
        cost_expr += 0.5 * _quad_form(u, self.cost.R)

        # Target-tracking cost
        target_err = np.array([ee_x - target_pos[0], ee_z - target_pos[1]])
        cost_expr += 0.5 * self.cost.target_weight * sum(e * e for e in target_err)

        # Avoid enemy (optional)
        if enemy_pos is not None and self.cost.avoid_weight > 0:
            enemy_err = np.array([ee_x - enemy_pos[0], ee_z - enemy_pos[1]])
            dist2 = sum(e * e for e in enemy_err)
            cost_expr += self.cost.avoid_weight / (self.cost.avoid_sigma + dist2)

        dircol.AddRunningCost(cost_expr)

        # Terminal cost
        xf = dircol.final_state()
        qf = xf[:2]
        ee_xf, ee_zf = _ee_pos_expr(qf, self.params)

        dxf = xf - self.cost.x_goal
        final_cost = 0.5 * _quad_form(dxf, self.cost.Qf)

        target_err_f = np.array([ee_xf - target_pos[0], ee_zf - target_pos[1]])
        final_cost += 0.5 * self.cost.target_weight * sum(e * e for e in target_err_f)

        dircol.AddFinalCost(final_cost)

        # Initial guess: constant state, zero torque
        times = np.linspace(0.0, self.N * self.dt, self.N + 1)
        x_init = np.repeat(x0.reshape(-1, 1), repeats=self.N + 1, axis=1)
        u_init = np.zeros((2, self.N + 1))
        dircol.SetInitialTrajectory(
            PiecewisePolynomial.ZeroOrderHold(times, x_init),
            PiecewisePolynomial.ZeroOrderHold(times, u_init),
        )

        # Solve ----------------------------------------------------
        try:
            result = Solve(dircol)
        except Exception as exc:
            return CollocationResult(
                x_trj=np.zeros((self.N + 1, 4)),
                u_trj=np.zeros((self.N, 2)),
                cost=np.inf,
                success=False,
                message=f"Solver error:\n{exc}",
            )

        success = result.is_success()

        x_trj = np.vstack([
            result.GetSolution(dircol.state(i)) for i in range(dircol.num_time_samples())
        ])
        u_trj = np.vstack([
            result.GetSolution(dircol.input(i)) for i in range(dircol.num_time_samples() - 1)
        ])

        return CollocationResult(
            x_trj=x_trj,
            u_trj=u_trj,
            cost=result.get_optimal_cost(),
            success=success,
            message="OK" if success else "Solver failed.",
        )