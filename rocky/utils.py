# rocky/utils.py
import numpy as np
from pydrake.multibody.tree import RevoluteJoint
from .dynamics import forward_kinematics

# --- Joint helpers ---
def get_revolute_joints_for_instance(plant, instance):
    joints = [plant.get_joint(j) for j in plant.GetJointIndices(instance)]
    return [j for j in joints if isinstance(j, RevoluteJoint)]

def set_angles_for_instance(plant, ctx, instance, angles_rad):
    revs = get_revolute_joints_for_instance(plant, instance)
    for j, q in zip(revs, angles_rad):
        j.set_angle(ctx, float(q))


def set_zero_torques(plant, ctx):
    plant.get_actuation_input_port().FixValue(ctx, np.zeros(plant.num_actuators()))

def set_joint_positions(plant, plant_context, q):
    # Sets the entire generalized position vector (all models).
    plant.SetPositions(plant_context, np.asarray(q))


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



def has_hit_target(pos, target, R_target=0.1, R_ee=0.005):
    return np.linalg.norm(pos - target) <= (R_target + R_ee)

def get_block_point_2d(enemy_ee_2d: np.ndarray,
                       enemy_target_2d: np.ndarray,
                       lam: float = 0.5) -> np.ndarray:
    """
    Midpoint blocking: point between enemy EE and its target.
    lam=0.5 gives true midpoint, lam<0.5 closer to EE, lam>0.5 closer to target.
    """
    return enemy_ee_2d + lam * (enemy_target_2d - enemy_ee_2d)

def clamp_radius(p, r_min, r_max):
    r = np.linalg.norm(p)
    if r < 1e-6:
        return p
    r_clamped = np.clip(r, r_min, r_max)
    return p * (r_clamped / r)


def get_ally_state(plant, ctx, ally):
    q = plant.GetPositions(ctx, ally)[:2]
    dq = plant.GetVelocities(ctx, ally)[:2]
    return np.hstack([q, dq])


def get_enemy_state(plant, ctx, enemy):
    q = plant.GetPositions(ctx, enemy)[:2]
    dq = plant.GetVelocities(ctx, enemy)[:2]
    return np.hstack([q, dq])


def get_shoulder_in_world(plant, ctx, wall_frame, W):
    X_W_wall = plant.CalcRelativeTransform(ctx, W, wall_frame)
    shoulder_in_wall = np.array([0.0, 0.0, 1.0])   # joint origin in wall frame
    return X_W_wall.translation() + X_W_wall.rotation().matrix() @ shoulder_in_wall


def express_point_in_wall_2d(plant, ctx, wall_frame, p_W_point, p_W_shoulder, W):
    # Point in wall frame
    X_wall_W = plant.CalcRelativeTransform(ctx, wall_frame, W)
    p_wall_point    = X_wall_W.multiply(p_W_point)
    p_wall_shoulder = X_wall_W.multiply(p_W_shoulder)
    rel = p_wall_point - p_wall_shoulder
    # 2D planar coordinates: (x,z)
    return np.array([rel[0], rel[2]])




def get_ally_target_pos_2d(plant, ctx, W, ally_wall_frame, enemy_target_frame):
    """
    Ally's target: the 'enemy_target' sphere on the ENEMY side,
    expressed in Ally wall frame (2D x,z).
    """
    p_W_shoulder = get_shoulder_in_world(plant, ctx, ally_wall_frame, W)
    p_W_T = plant.CalcRelativeTransform(ctx, W, enemy_target_frame).translation()
    return express_point_in_wall_2d(plant, ctx, ally_wall_frame, p_W_T, p_W_shoulder, W)


def get_enemy_pos_2d(plant, ctx, W, ally_wall_frame, enemy_glove_frame):
    """
    Enemy EE expressed in Ally wall frame (2D x,z).
    Used when Ally is attacking or defending.
    """
    p_W_shoulder = get_shoulder_in_world(plant, ctx, ally_wall_frame, W)
    p_W_EE = plant.CalcRelativeTransform(ctx, W, enemy_glove_frame).translation()
    return express_point_in_wall_2d(plant, ctx, ally_wall_frame, p_W_EE, p_W_shoulder, W)


def get_enemy_target_pos_2d_for_ally(plant, ctx, W, ally_wall_frame, ally_target_frame):
    """
    Enemy's target ('ally_target' sphere near Ally wall),
    expressed in Ally wall frame (2D x,z).
    Used when Ally is blocking the Enemy's strike.
    """
    p_W_shoulder = get_shoulder_in_world(plant, ctx, ally_wall_frame, W)
    p_W_T = plant.CalcRelativeTransform(ctx, W, ally_target_frame).translation()
    return express_point_in_wall_2d(plant, ctx, ally_wall_frame, p_W_T, p_W_shoulder, W)



def get_enemy_target_pos_2d_for_enemy(plant, ctx, W, enemy_wall_frame, ally_target_frame):
    """
    Enemy's target: the 'ally_target' sphere near the ALLY wall,
    expressed in Enemy wall frame (2D x,z).
    """
    p_W_shoulder = get_shoulder_in_world(plant, ctx, enemy_wall_frame, W)
    p_W_T = plant.CalcRelativeTransform(ctx, W, ally_target_frame).translation()
    return express_point_in_wall_2d(plant, ctx, enemy_wall_frame, p_W_T, p_W_shoulder, W)


def get_ally_pos_2d_for_enemy(plant, ctx, W, enemy_wall_frame, ally_glove_frame):
    """
    Ally EE expressed in Enemy wall frame (2D x,z).
    Used for Enemy's avoidance / planning.
    """
    p_W_shoulder = get_shoulder_in_world(plant, ctx, enemy_wall_frame, W)
    p_W_ally = plant.CalcRelativeTransform(ctx, W, ally_glove_frame).translation()
    return express_point_in_wall_2d(plant, ctx, enemy_wall_frame, p_W_ally, p_W_shoulder, W)


def get_enemy_pos_2d_for_enemy(plant, ctx, W, enemy_wall_frame, enemy_glove_frame):
    """
    Enemy EE expressed in Enemy wall frame (2D x,z).
    Used both for control and hit detection on Enemy side.
    """
    p_W_shoulder = get_shoulder_in_world(plant, ctx, enemy_wall_frame, W)
    p_W_enemy = plant.CalcRelativeTransform(ctx, W, enemy_glove_frame).translation()
    return express_point_in_wall_2d(plant, ctx, enemy_wall_frame, p_W_enemy, p_W_shoulder, W)

def get_ally_pos_2d_from_model(x, params):
    return forward_kinematics(x[:2], params)


def get_enemy_pos_2d_from_model(x, params):
    return forward_kinematics(x[:2], params)


def apply_combined_torques(plant, ctx, u_ally, u_enemy, ally_act_idxs, enemy_act_idxs):
    tau = np.zeros(plant.num_actuators())
    if u_ally is not None:
        tau[ally_act_idxs[0]] = u_ally[0]
        tau[ally_act_idxs[1]] = u_ally[1]
    if u_enemy is not None:
        tau[enemy_act_idxs[0]] = u_enemy[0]
        tau[enemy_act_idxs[1]] = u_enemy[1]
    plant.get_actuation_input_port().FixValue(ctx, tau)


def apply_ally_torque(plant, ctx, ally, u2):
    apply_combined_torques(u_ally=u2, u_enemy=None)


def apply_enemy_torque(plant, ctx, enemy, u2):
    apply_combined_torques(u_ally=None, u_enemy=u2)



def randomize_enemy_configuration(
    clearance_buffer=0.02,
    clearance_target=None,
    clearance_ally=None,
    max_tries=200,
    plant=None,
    enemy=None,
    ctx=None,
    W=None,
    enemy_glove_frame=None,
    enemy_target_frame=None,
    ally_glove_frame=None,
    enemy_elbow_frame=None,
    enemy_wall_frame=None
):
    rng = np.random.default_rng()
    
    """Sample random enemy joint angles that keep the glove away from target/ally."""
    glove_radius = 0.05
    target_radius = 0.03
    clearance_target = clearance_target or (glove_radius + target_radius + clearance_buffer)
    clearance_ally = clearance_ally or (2 * glove_radius + clearance_buffer)

    joints = get_revolute_joints_for_instance(plant, enemy)
    limits = []
    for j in joints:
        lo = j.position_lower_limit()
        hi = j.position_upper_limit()
        if not np.isfinite(lo):
            lo = -np.pi
        if not np.isfinite(hi):
            hi = np.pi
        limits.append((lo, hi))

    for attempt in range(1, max_tries + 1):
        candidate = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        set_angles_for_instance(plant, ctx, enemy, candidate)

        p_enemy = plant.CalcRelativeTransform(ctx, W, enemy_glove_frame).translation()
        p_target = plant.CalcRelativeTransform(ctx, W, enemy_target_frame).translation()
        p_ally = plant.CalcRelativeTransform(ctx, W, ally_glove_frame).translation()
        p_elbow = plant.CalcRelativeTransform(ctx, W, enemy_elbow_frame).translation()
        wall_x = plant.CalcRelativeTransform(ctx, W, enemy_wall_frame).translation()[0]

        if not (p_elbow[0] < wall_x and p_enemy[0] < wall_x):
            continue
        if np.linalg.norm(p_enemy - p_target) < clearance_target:
            continue
        if np.linalg.norm(p_enemy - p_ally) < clearance_ally:
            continue
        print(f"Enemy randomized in {attempt} tries.")
        return candidate

    raise RuntimeError("Failed to sample a collision-free enemy configuration")
