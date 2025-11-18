# rocky/build.py
from dataclasses import dataclass
import numpy as np
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    RigidTransform, RotationMatrix, AddDefaultVisualization, Simulator
)

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

def _finalize(builder, plant, gravity_vec, meshcat):
    plant.mutable_gravity_field().set_gravity_vector(gravity_vec)
    plant.Finalize()
    if meshcat is not None:
        AddDefaultVisualization(builder, meshcat=meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sim = Simulator(diagram, context)
    sim.set_publish_every_time_step(True)
    return diagram, context, plant_context, sim

def build_robot_diagram_two(
    urdf_path: str,
    time_step: float = 1e-3,
    gravity_vec=(0., 0., -9.81),
    meshcat=None,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)  # creates rocky(1), rocky(2)

    ally  = parser.AddModels(urdf_path)[0]
    enemy = parser.AddModels(urdf_path)[0]

    W = plant.world_frame()
    A_wall = plant.GetFrameByName("wall", model_instance=ally)
    E_wall = plant.GetFrameByName("wall", model_instance=enemy)

    # Place the two robots apart in Y. No rotation needed.
    X_WA = RigidTransform([0.0, 0.0, 0.0])
    X_WE = RigidTransform(RotationMatrix.MakeZRotation(np.pi), [2.0, 0.0, 0.0])


    plant.WeldFrames(W, A_wall, X_WA)
    plant.WeldFrames(W, E_wall, X_WE)

    diagram, context, plant_context, sim = _finalize(builder, plant, gravity_vec, meshcat)
    return SimBundle(builder, plant, scene_graph, diagram, context, plant_context, sim,
                     ally=ally, enemy=enemy)
