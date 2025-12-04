import numpy as np
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    MultibodyPlant, DirectCollocation, PiecewisePolynomial,
    Solve, SnoptSolver, MathematicalProgram, RevoluteJoint,
    BoundingBoxConstraint
)

class CollocationDefender:
    """
    Trajectory Optimization-based controller using Direct Collocation.
    Plans a path for the Ally robot to intercept a blocking target.
    """
    def __init__(self, urdf_path: str, num_time_samples: int = 21, time_horizon: float = 0.5):
        self.urdf_path = urdf_path
        self.N = num_time_samples
        self.T = time_horizon

        # 1. Build a internal 'float' plant for basic context creation
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(self.plant)
        self.model = parser.AddModels(self.urdf_path)[0]
        
        # Manually add actuators because the URDF lacks transmission tags
        for j_index in self.plant.GetJointIndices(self.model):
            joint = self.plant.get_joint(j_index)
            if isinstance(joint, RevoluteJoint):
                self.plant.AddJointActuator(f"{joint.name()}_act", joint)
        
        # Weld wall to world so the arm is fixed
        self.plant.WeldFrames(
            self.plant.world_frame(), 
            self.plant.GetFrameByName("wall", self.model)
        )
        self.plant.Finalize()
        self.diagram = builder.Build()
        
        # 2. Create an AutoDiff version of the plant for optimization gradients
        self.plant_ad = self.plant.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()
        
        # Cache frames for the glove to compute FK in cost
        self.glove_frame_ad = self.plant_ad.GetFrameByName("glove", self.model)
        self.wall_frame_ad = self.plant_ad.GetFrameByName("wall", self.model)


    def plan(self, x_start: np.ndarray, target_pos_2d: np.ndarray):
        # Create a fresh context for this plan
        context = self.plant.CreateDefaultContext()

        # Explicitly get actuation port
        act_port = self.plant.get_actuation_input_port()

        # Initialize Direct Collocation
        dircol = DirectCollocation(
            self.plant,
            context,
            num_time_samples=self.N,
            minimum_time_step=self.T / self.N,
            maximum_time_step=self.T / self.N,
            input_port_index=act_port.get_index(),
        )

        prog = dircol.prog()

        # 1. Initial State Constraint
        prog.AddBoundingBoxConstraint(x_start, x_start, dircol.initial_state())

        # 2. Input (Torque) Constraints  <-- FIXED
        dircol.AddInputBounds(-50.0, 50.0)

        # 3. Running Cost (Minimum Effort)
        u = dircol.input()

        # Note: u is (N,1) vector of symbolic vars for one knot
        # Add this constraint at every knot point:
        prog.AddBoundingBoxConstraint(
            -50 * np.ones(2),
            +50 * np.ones(2),
            u
        )


        # 4. Final Position Cost (Blocking)
        def terminal_blocking_cost(x_state_ad):
            q_ad = x_state_ad[:2]
            self.plant_ad.SetPositions(self.context_ad, self.model, q_ad)
            X_WG = self.glove_frame_ad.CalcPoseInWorld(self.context_ad)
            p_G_W = X_WG.translation()
            diff_x = p_G_W[0] - target_pos_2d[0]
            diff_z = p_G_W[2] - target_pos_2d[1]
            weight = 1000.0
            return weight * (diff_x**2 + diff_z**2)

        prog.AddCost(terminal_blocking_cost, dircol.final_state())

        xf = dircol.final_state()
        prog.AddCost(10.0 * xf[2:].dot(xf[2:]))

        # 5. Initial Guess
        t_knots = [0.0, self.T]
        initial_u = PiecewisePolynomial.ZeroOrderHold(
            t_knots,
            np.zeros((2, 2)),
        )
        initial_x = PiecewisePolynomial.FirstOrderHold(
            t_knots,
            np.column_stack((x_start, x_start)),
        )
        dircol.SetInitialTrajectory(initial_u, initial_x)

        # 6. Solve
        result = Solve(prog)

        if result.is_success():
            u_traj = dircol.ReconstructInputTrajectory(result)
            u0 = u_traj.value(0.0).flatten()
            return u0, result
        else:
            return np.zeros(2), result
