# rocky/build.py
from dataclasses import dataclass
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, AddDefaultVisualization, Simulator

@dataclass
class SimBundle:
    builder: DiagramBuilder
    plant: any
    scene_graph: any
    diagram: any
    context: any           # root diagram context
    plant_context: any     # <-- add this
    simulator: Simulator

def build_robot_diagram(urdf_path, time_step=1e-3, gravity_vec=(0.,0.,0.), meshcat=None) -> SimBundle:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    Parser(plant).AddModels(urdf_path)
    plant.mutable_gravity_field().set_gravity_vector(gravity_vec)
    plant.Finalize()

    if meshcat is not None:
        AddDefaultVisualization(builder, meshcat=meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)   # <-- key line

    sim = Simulator(diagram, context)
    sim.set_publish_every_time_step(True)
    sim.Initialize()

    return SimBundle(builder, plant, scene_graph, diagram, context, plant_context, sim)
