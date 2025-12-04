import numpy as np
import os
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    DirectCollocation, Solve, PiecewisePolynomial
)

class CollocationDefender:
    def __init__(self, urdf_path, num_time_samples=21, time_horizon=0.5):
        """
        Args:
            urdf_path: Path to rocky.urdf
            num_time_samples: Number of collocation points (nodes).
            time_horizon: How far into the future to plan (seconds).
        """
        self.dt_horizon = time_horizon
        self.N = num_time_samples

        # 0. Resolve absolute path to avoid relative path issues in Parser
        urdf_abspath = os.path.abspath(urdf_path)

        # 1. Build the internal plant/diagram ONCE (AutoDiff-capable)
        self.builder = DiagramBuilder()
        # Use time_step=0.0 for Continuous plant (required for DirectCollocation)
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.0
        )
        self.parser = Parser(self.plant)

        # Fix: Use AddModels (plural) and take the first instance
        models = self.parser.AddModels(urdf_abspath)
        if not models:
            raise ValueError(f"No models found in {urdf_abspath}")
        self.model = models[0]

        # Weld wall to world (same as simulation)
        W = self.plant.world_frame()
        try:
            wall = self.plant.GetFrameByName("wall", self.model)
        except:
             raise ValueError("Could not find frame 'wall' in the loaded model.")
             
        self.plant.WeldFrames(W, wall)

        self.plant.Finalize()

        # Sanity Check: Ensure actuators loaded (prevents IndexError later)
        if self.plant.num_actuators() == 0:
            raise ValueError(
                "Plant has 0 actuators. The URDF loaded, but transmissions were not found. "
                "Check that your URDF has <transmission> tags and the path is correct."
            )

        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)

        # Frames for constraints
        self.world_frame = self.plant.world_frame()
        self.glove_frame = self.plant.GetFrameByName("glove", self.model)

        # Input limits
        self.tau_limit = 50.0  

        # Cache the last trajectory to warm-start the next solve (optional)
        self.initial_guess_x = None
        self.initial_guess_u = None

    def plan(self, x_current, target_pos_2d):
        """
        Solves the trajectory optimization problem.

        Args:
            x_current: np.array [q1, q2, dq1, dq2]
            target_pos_2d: np.array [x, z] in Ally Wall Frame

        Returns:
            u0: The first control input to apply (MPC style)
            res: The solver result object
        """
        # 1. Setup Direct Collocation on the already-built diagram
        context = self.diagram.CreateDefaultContext()
        dircol = DirectCollocation(
            self.diagram,
            context,
            num_time_samples=self.N,
            minimum_time_step=self.dt_horizon / (self.N - 1),
            maximum_time_step=self.dt_horizon / (self.N - 1),
        )

        prog = dircol.prog()

        # 2. Constraints

        # A) Initial state constraint
        prog.AddBoundingBoxConstraint(
            x_current, x_current, dircol.initial_state()
        )

        # B) Input limits (torque)
        u = dircol.input()
        # The IndexError likely happened here if u was empty (0 actuators)
        dircol.AddConstraintToAllKnotPoints(u[0] <= self.tau_limit)
        dircol.AddConstraintToAllKnotPoints(u[0] >= -self.tau_limit)
        dircol.AddConstraintToAllKnotPoints(u[1] <= self.tau_limit)
        dircol.AddConstraintToAllKnotPoints(u[1] >= -self.tau_limit)

        # C) Final position cost
        target_pos_3d = np.array([target_pos_2d[0], 0.0, target_pos_2d[1]])
        final_state = dircol.final_state()  # [q1, q2, dq1, dq2]

        def final_pose_cost(q):
            # q is [q1, q2] (AutoDiff / symbolic)
            self.plant.SetPositions(self.plant_context, q[:2])
            tf = self.plant.CalcRelativeTransform(
                self.plant_context, self.world_frame, self.glove_frame
            )
            pos = tf.translation()
            err_x = pos[0] - target_pos_3d[0]
            err_z = pos[2] - target_pos_3d[2]
            return (err_x**2 + err_z**2) * 5000.0

        prog.AddCost(final_pose_cost, final_state[:2])

        # D) Minimize velocity at end
        prog.AddBoundingBoxConstraint(0.0, 0.0, final_state[2:])

        # 3. Running cost (effort)
        R = 0.1
        dircol.AddRunningCost(R * (u[0]**2 + u[1]**2))

        # 4. Initial guess (simple stationary)
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
            [0.0, self.dt_horizon],
            np.column_stack((x_current, x_current))
        )
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

        # 5. Solve
        result = Solve(prog)

        if not result.is_success():
            return np.zeros(2), result

        # 6. Extract first control
        u_traj = dircol.ReconstructInputTrajectory(result)
        u0 = u_traj.value(0.0).flatten()
        return u0, result