# rocky/utils.py
import numpy as np


def set_joint_positions(plant, plant_context, q):
    # Sets the entire generalized position vector (all models).
    plant.SetPositions(plant_context, np.asarray(q))


def set_zero_torques(plant, plant_context):
    u0 = np.zeros(plant.num_actuators())
    plant.get_actuation_input_port().FixValue(plant_context, u0)


# set joint angles 
def set_angles_for_instance(plant, plant_context, model_instance, shoulder, elbow):
    plant.GetJointByName("shoulder", model_instance).set_angle(plant_context, shoulder)
    plant.GetJointByName("elbow",    model_instance).set_angle(plant_context, elbow)