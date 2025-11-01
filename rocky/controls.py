import numpy as np
import time
from scipy.signal import cont2discrete
from typing import List, Tuple

# import quad_sim
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp


class iLQR(object):

    def __init__(
        self,
        x_goal: np.ndarray,
        N: int,   #horizon
        dt: float, # timestamp
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
    ):
        # M, H, C, B are defined using pydrake MultibodyPlant
        # forwad dynamics when given (q, q_dot, u) -> q_ddot
        self.nx = 4    # state x = [theta1, theta2, theta1_dot, theta2_dot]
        self.nu = 2    # input u = [torque1, torque2]
        
        # iLQR constants
        self.N = N     # horizon
        self.dt = dt   # delta time / timestamp

        # regularization
        self.rho = 1e-8
        self.alpha_set = 1
        self.max_iter = 1000
        self.tol = 1e-4 

        # target state
        self.x_goal = x_goal
        self.u_goal = 0.5 * 9.81 * np.ones((2,))    

        # Cost terms
        self.Q = Q
        self.R = R
        self.Qf = Qf

        # symbolic linearized dynamics
        self.A, self.B, self.xdot = self.get_symbolic_linearized_dynamics()

    def simulate_dynamics(s): 
        pass

    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> float:
        # running cost incurred by xk, uk
        lqr_cost = (
            self.dt * 0.5
        )

        return lqr_cost
    
    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        # gradient of running cost [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        grad = np.zeros((self.nx + self.nu,))

        dldx = self.Q @ (xk - self.x_goal)
        dldu = self.R @ (uk - self.u_goal)

        grad[:self.nx] = dldx
        grad[self.nx:] = dldu

        return grad
    
    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        # return: The hessian of the running cost [[∂²l/∂x², ∂²l/∂x∂u], [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        H = np.zeros((self.nx + self.nu, self.nx + self.nu))

        dldx2 = self.Q
        # dldxu = 0
        # dldux = 0
        dldu2 = self.R

        H[:self.nx, :self.nx] = dldx2
        H[self.nx:, self.nx:] = dldu2

        return H
    
    def terminal_cost(self, xf: np.ndarray) -> float:
        # return Lf(xf), running cost incurred by xf
        return 0.5 * (xf - self.x_goal).T @ self.Qf @ (xf - self.x_goal)
    
    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        # return: ∂Lf/∂xf
        grad = np.zeros((self.nx))
        
        grad = np.array(self.Qf @ (xf - self.x_goal))
        return grad