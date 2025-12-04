# **Two-Robot Boxing Simulator with MPC + iLQR (Drake + Python)**

*A real-time dual-arm interaction system combining physics-based simulation, differentiable dynamics, and optimal control.*

---

## **Overview**

This repository implements a full simulation and control stack for a **two-link robotic arm boxing environment** built on **Drake**.
The system models two planar robot arms (“Ally” and “Enemy”), places them facing each other, and allows them to perform **attacks** and **defensive parries** using **Model Predictive Control (MPC)** driven by **Iterative Linear Quadratic Regulator (iLQR)** optimization.

The project contains:

* A complete physics simulation (URDF-based, with collisions and targets).
* Differentiable analytical dynamics for a 2-DoF arm.
* A full iLQR solver (derivatives, backward pass, forward rollout, line search).
* MPC wrapper for real-time receding-horizon execution.
* Utilities for integrating with Drake actuators.
* Example notebooks and simulation demos.

> **This README includes placeholders for MP4 demos—add them once uploaded.**

---

## **Demo Videos**

Coming soon.

* **✓ Ally attack sequence** (MP4 placeholder)
* **✓ Enemy attack + Ally parry (defensive iLQR)** (MP4 placeholder)
* **✓ Side-by-side MPC visualizations** (MP4 placeholder)

---

# **Repository Structure**

```
rocky/
│
├── build.py          # Builds the Drake diagram (plant, scene graph, targets)
├── dynamics.py       # Analytic dynamics, FK, Jacobian, linearization
├── ilqr.py           # Full iLQR solver + MPC wrapper
├── utils.py          # Utility functions for joint control and Drake wiring
├── rocky.urdf        # Mechanical model of the 2-link robot
├── simulate.ipynb    # Example simulation notebook
└── notes.cpp         # Concept notes for MPC + iLQR
```

---

# **1. Drake Environment Construction**

### **File:** `build.py` 

This module constructs the full Drake simulation:

* Creates a `MultibodyPlant` and `SceneGraph`
* Loads two identical robots from `rocky.urdf` (“ally” and “enemy”)
* Welds each robot to opposite sides of the world
* Places target frames and visual markers on each robot’s side
* Registers both **visual** and **collision** geometry
* Adds actuators to each revolute joint
* Optionally filters out link–link collisions for cleaner interactions
* Finalizes the plant and returns a `SimBundle` containing:

  * `diagram`, `context`, `plant_context`, `simulator`
  * Robot instance handles: `ally`, `enemy`

This module fully sets up the environment for all control experiments.

---

# **2. Analytical Dynamics Model**

### **File:** `dynamics.py` 

Defines an analytical, differentiable 2-DoF arm model used inside iLQR and MPC.

### **Key Components**

* **`ArmParams` dataclass**
  Arm geometry, masses, inertias, damping, gravity, torque limits.

* **Forward Kinematics**

  ```python
  forward_kinematics(q, params) → [x, z]
  ```

  Computes planar EE coordinates.

* **Jacobian**
  Provides analytic `∂p/∂q` used for gradient/Hessian in cost terms.

* **Dynamics Components**

  * Mass matrix `M(q)`
  * Coriolis matrix `C(q, dq)`
  * Gravity torque `G(q)`
  * Joint damping

* **Continuous and Discrete Dynamics**
  Euler integration for `x_{k+1} = x_k + dt * f(x,u)`.

* **Finite-difference Linearization**
  Produces linearized dynamics `(A = fx, B = fu)` for the iLQR backward pass.

The dynamics model is independent of Drake and runs purely in NumPy.

---

# **3. iLQR Solver + MPC**

### **File:** `ilqr.py` 

This module implements the complete control stack:

---

## **3.1 Cost Functions**

The system optimizes a mixture of:

* **Quadratic state deviation**
* **Quadratic control penalty**
* **Task-space target reaching** (move EE to a desired point)
* **Task-space Gaussian avoidance** (keep distance from the opponent arm)
* **Hard barrier inside a minimum radius**

Both running and terminal costs include analytical gradients and Hessians.

---

## **3.2 iLQR Optimization**

The solver executes:

1. **Forward rollout** to compute nominal trajectory and cost
2. **Backward Riccati recursion** to compute feedback gains `(k, K)`
3. **Forward line search** across multiple step sizes
4. **Convergence checks & regularization** for robustness

The solver returns an `ILQRResult` containing:

* optimized state trajectory
* optimized control trajectory
* total cost
* iteration count
* convergence flag

---

## **3.3 MPC Wrapper**

The method `step_mpc()`:

* Shifts previous control trajectory (warm-start)
* Calls iLQR for the next prediction window
* Extracts the first control input as the MPC action
* Advances the system using discrete dynamics

This enables **real-time** control of the robot.

---

# **4. Utilities**

### **File:** `utils.py` 

Small but essential helper functions:

* **Set joint positions or angles** for any robot instance
* **Generate actuator index maps** for linking Drake actuators to our 2-DoF controls
* **Create a control applier** that maps `[τ1, τ2]` into Drake’s full generalized force vector

Useful whenever integrating MPC+iLQR with Drake simulation.

---

# **5. Notes and Formulas**

### **File:** `notes.cpp` 

Not code—summary notes that describe:

* State representation
* Control representation
* Dynamics equations
* Cost structure

Helpful when extending the system with new behaviors.

---

# **6. Simulation Notebook**

### **File:** `simulate.ipynb`

Contains example experiments such as:

* Ally attacking the target
* Enemy attacking while Ally defends
* MPC visualization through a simulated time loop
* Plotting trajectories of EE and targets

This notebook is the easiest starting point to run the system end-to-end.

---

# **7. How to Run**

### **Prerequisites**

* Python 3.10+
* `pydrake`
* `numpy`
* `matplotlib` (optional, for plotting)
* Jupyter for running `simulate.ipynb`

### **Build and Simulate**

```python
from rocky.build import build_robot_diagram_two
from rocky.ilqr import ILQRController, CostParams
from rocky.dynamics import ArmParams
from rocky.utils import set_angles_for_instance

bundle = build_robot_diagram_two("rocky.urdf", time_step=1e-3)
sim = bundle.simulator
sim.AdvanceTo(2.0)
```

### **Running MPC loop example**

```python
ctrl = ILQRController(horizon=50, dt=0.02)
x = np.zeros(4)   # initial [q1, q2, dq1, dq2]

for t in range(200):
    x, result = ctrl.step_mpc(
        x_current=x,
        target_pos=np.array([1.0, 0.2]),
        enemy_pos=np.array([0.3, 0.1]),
    )
```

---

# **8. Extending This Project**

You can extend this environment into:

* Two-arm adversarial RL
* Contact-based blocking/striking experiments
* Nonlinear MPC with different cost shaping
* Robotic fencing, martial arts, or cooperative manipulation
* Higher-DoF robot assets (Franka, Kinova, Unitree arms, etc.)

The modular structure makes such extensions straightforward.

---
