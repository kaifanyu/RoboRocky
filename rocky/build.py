# rocky/build.py
from dataclasses import dataclass
import numpy as np
from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    RigidTransform, RotationMatrix, AddDefaultVisualization, Simulator,
    Rgba, Sphere, CoulombFriction
)
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint
from pydrake.geometry import ProximityProperties, AddContactMaterial

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


# finalizes plant, builds diagram, creates context and simulator
def _finalize(builder, plant, gravity_vec, meshcat):
    plant.mutable_gravity_field().set_gravity_vector(gravity_vec)       # set gravity vector
    plant.Finalize()                                                    # finalizes plant
    if meshcat is not None:
        AddDefaultVisualization(builder, meshcat=meshcat)               # add visualization
    diagram = builder.Build()                                           # build diagram
    context = diagram.CreateDefaultContext()                            # build context
    plant_context = plant.GetMyContextFromRoot(context)                 # extract plant_context from context
    sim = Simulator(diagram, context)                                   # wraps in simulator, enables publishing every time stamp
    sim.set_publish_every_time_step(True)
    return diagram, context, plant_context, sim


def build_robot_diagram_two(
    urdf_path: str,
    time_step: float = 1e-3,
    gravity_vec=(0., 0., 0.),
    meshcat=None,
):
    builder = DiagramBuilder()      # create a builder, holds plant and scene

    # adds a plant: core object that represents my robot in the world with physics engine
    # builds it with timestep, scene_graph is for visualization
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)  
    parser = Parser(plant)      # loads URDF into plant


    # creates rocky(1), rocky(2)
    parser.SetAutoRenaming(True)
    ally  = parser.AddModels(urdf_path)[0]
    enemy = parser.AddModels(urdf_path)[0]

    W = plant.world_frame() # W is the global world frame
    A_wall = plant.GetFrameByName("wall", model_instance=ally)      # frame named wall that belongs to ally
    E_wall = plant.GetFrameByName("wall", model_instance=enemy)     # frame named wall for enemy

    # Translate and rotate ally and enemy
    X_WA = RigidTransform([0.0, 0.0, 0.0])      # place first at 0 0 0 
    X_WE = RigidTransform(RotationMatrix.MakeZRotation(np.pi), [1.1, 0.0, 0.0]) # rotate by pi and +2x

    plant.WeldFrames(W, A_wall, X_WA)
    plant.WeldFrames(W, E_wall, X_WE)


    # for each model, add the joint
    for model in [ally, enemy]:
        for j_index in plant.GetJointIndices(model):
            joint = plant.get_joint(j_index)
            if isinstance(joint, RevoluteJoint):
                # Name doesn't matter much, just must be unique
                plant.AddJointActuator(f"{joint.name()}_act", joint)



    # add ally (player) target frame on enemy side
    target_pos_W = np.array([1.0, 0.0, 1.25])  # (x,y,z) in world
    X_WT = RigidTransform(target_pos_W)

    # fix it to frame "enemy_target"
    plant.AddFrame(FixedOffsetFrame("enemy_target", W, X_WT))

    # visual sphere 
    radius = 0.05
    diffuse = np.array([1.0, 0.0, 0.0, 1.0])  # RGBA as numpy array

    plant.RegisterVisualGeometry(
        plant.world_body(),   # anchored to world body
        X_WT,
        Sphere(radius),
        "enemy_target_visual",
        diffuse
    )
    # register friction to target to get collision
    friction = CoulombFriction(static_friction=0.8, dynamic_friction=0.5)
    props = ProximityProperties()
    # Example numbers; tune these:
    elastic_modulus = 5e7      # larger -> stiffer, more bounce
    dissipation = 0.1          # smaller -> less energy loss, more bounce
    AddContactMaterial(dissipation, elastic_modulus, friction, props)

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WT,
        Sphere(radius),
        "enemy_target_collision",
        props,
    )

    # add enemy target frame on ally side so enemy can strike back
    enemy_target_pos_W = np.array([0.15, 0.0, 1.25])
    X_WT_enemy = RigidTransform(enemy_target_pos_W)
    plant.AddFrame(FixedOffsetFrame("ally_target", W, X_WT_enemy))

    enemy_diffuse = np.array([0.0, 0.4, 1.0, 1.0])
    props_enemy = ProximityProperties()
    AddContactMaterial(dissipation, elastic_modulus, friction, props_enemy)
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_WT_enemy,
        Sphere(radius),
        "ally_target_visual",
        enemy_diffuse,
    )

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WT_enemy,
        Sphere(radius),
        "ally_target_collision",
        props_enemy,
    )
    
    diagram, context, plant_context, sim = _finalize(builder, plant, gravity_vec, meshcat)
    
    return SimBundle(builder, plant, scene_graph, diagram, context, plant_context, sim,
                     ally=ally, enemy=enemy)
