# rocky/utils.py
import numpy as np


def set_joint_positions(plant, plant_context, q):
    # Sets the entire generalized position vector (all models).
    plant.SetPositions(plant_context, np.asarray(q))

# set joint angles 
def set_angles_for_instance(plant, plant_context, model_instance, shoulder, elbow):
    plant.GetJointByName("shoulder", model_instance).set_angle(plant_context, shoulder)
    plant.GetJointByName("elbow",    model_instance).set_angle(plant_context, elbow)

def make_actuator_map(plant, model_instance):
    """Returns a list of velocity indices for actuated DOFs."""
    vel_indices = []
    for joint_index in plant.GetJointIndices(model_instance):
        joint = plant.get_joint(joint_index)
        if joint.num_velocities() > 0:
            vel_indices.append(joint.velocity_start())
    return vel_indices


def make_control_applier(plant, model_instance):
    """Creates a function apply_control(ctx, u_small) that:
       - maps u_small (2D) to the correct velocity indices
       - pads zeros for everything else
       - writes to applied_generalized_force port
    """
    vel_map = make_actuator_map(plant, model_instance)

    def apply_control(context, u_small):
        u_full = np.zeros(plant.num_velocities())
        for i, idx in enumerate(vel_map):
            u_full[idx] = u_small[i]
        plant.get_applied_generalized_force_input_port().FixValue(context, u_full)

    return apply_control
