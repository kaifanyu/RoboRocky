# rocky/utils.py
import numpy as np

def set_joint_positions(plant, plant_context, q):
    plant.SetPositions(plant_context, np.asarray(q))

def set_zero_torques(plant, plant_context):
    u0 = np.zeros(plant.num_actuators())
    plant.get_actuation_input_port().FixValue(plant_context, u0)

def advance(sim, T):
    sim.AdvanceTo(T)
