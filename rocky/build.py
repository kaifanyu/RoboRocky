# rocky/build.py
from dataclasses import dataclass
import numpy as np
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    RigidTransform, RotationMatrix, AddDefaultVisualization, Simulator
)
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.geometry import Sphere


@dataclass
class SimBundle:
    builder: DiagramBuilder
    plant: any
    scene_graph: any
    diagram: any
    context: any
    plant_context: any
    simulator: Simulator
    ally: int | None = None
    enemy: int | None = None

# Helper function
def _finalize(builder, plant, gravity_vec, meshcat):
    plant.mutable_gravity_field().set_gravity_vector(gravity_vec)   #set a gravity vector
    plant.Finalize()
    if meshcat is not None:
        AddDefaultVisualization(builder, meshcat=meshcat)   # visualize with meshcat
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sim = Simulator(diagram, context)
    sim.set_publish_every_time_step(True)
    return diagram, context, plant_context, sim

def build_robot_diagram_one(urdf_path, time_step=1e-3, gravity_vec=(0.,0.,0.), meshcat=None) -> SimBundle:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)

    # Load one model (URDF should NOT weld to world; no wall_to_world joint)
    ally = parser.AddModels(urdf_path)[0]

    # Place/weld the single instance
    W = plant.world_frame()
    A_wall = plant.GetFrameByName("wall", model_instance=ally)
    plant.WeldFrames(W, A_wall, RigidTransform([0., 0., 0.]))

    diagram, context, plant_context, sim = _finalize(builder, plant, gravity_vec, meshcat)
    return SimBundle(builder, plant, scene_graph, diagram, context, plant_context, sim, ally=ally)

# def build_robot_diagram_two(urdf_path, time_step=1e-3, gravity_vec=(0.,0.,0.), meshcat=None,) -> SimBundle:
def build_robot_diagram_two(
    urdf_path, time_step=1e-3, gravity_vec=(0.,0.,0.), meshcat=None,
    enemy_target_xyz=(0.07, 0.0, 1.0),   # 7 cm off the wall, z = 1.0 m
    enemy_target_rpy=(0.0, 0.0, 0.0),    # roll, pitch, yaw in radians
    enemy_target_radius=0.03,            # 3 cm sphere marker
    enemy_target_rgba=(0.95, 0.2, 0.2, 1.0),  # visible color
    ) -> SimBundle:

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)

    # Two copies of the same model
    parser.SetAutoRenaming(True)          # will create rocky(1), rocky(2), etc.
    ally  = parser.AddModels(urdf_path)[0]
    enemy = parser.AddModels(urdf_path)[0]

    # Place them (enemy rotated 180° about z and shifted to +y)
    W = plant.world_frame()
    A_wall = plant.GetFrameByName("wall", model_instance=ally)
    E_wall = plant.GetFrameByName("wall", model_instance=enemy)

    # Build rotation from rpy (X*Y*Z order is fine here)
    Rx = RotationMatrix.MakeXRotation(enemy_target_rpy[0])
    Ry = RotationMatrix.MakeYRotation(enemy_target_rpy[1])
    Rz = RotationMatrix.MakeZRotation(enemy_target_rpy[2])
    R_target = Rz @ Ry @ Rx
    X_wall_target = RigidTransform(R_target, np.array(enemy_target_xyz, dtype=float))

    # 1) A named frame the controller can query
    enemy_target = FixedOffsetFrame(
        name="enemy_target",
        P=E_wall,
        X_PF=X_wall_target
    )
    plant.AddFrame(enemy_target)

    # 2) A tiny visual marker so you can see it in Meshcat
    enemy_wall_body = plant.GetBodyByName("wall", enemy)
    plant.RegisterVisualGeometry(
        enemy_wall_body,
        X_wall_target,
        Sphere(enemy_target_radius),
        "enemy_target_vis",
        np.array(enemy_target_rgba, dtype=float)  # Drake wants a numpy RGBA
    )


    X_WA = RigidTransform([0.0, 0.0, 0.0])
    X_WE = RigidTransform(RotationMatrix.MakeZRotation(np.pi), [1.5, 0.0, -0.5])

    plant.WeldFrames(W, A_wall, X_WA)
    plant.WeldFrames(W, E_wall, X_WE)

    diagram, context, plant_context, sim = _finalize(builder, plant, gravity_vec, meshcat)
    return SimBundle(builder, plant, scene_graph, diagram, context, plant_context, sim,
                     ally=ally, enemy=enemy)

    # diagram, context, plant_context, sim = _finalize(builder, plant, gravity_vec, meshcat)
    # return SimBundle(builder, plant, scene_graph, diagram, context, plant_context, sim, ally=ally, enemy=enemy)
