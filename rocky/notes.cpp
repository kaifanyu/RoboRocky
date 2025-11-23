/*

MPC + iLQR set up:
We need in terms of:
- current state
- what we control (joint 1, joint 2)
- how arm evolves forward in time
- what cost we are minimizing over short horizon


state vector x = [q1, q2, q1_dot, q2_dot]
control vector u = [t1, t2] 
    - two joint torques

Need:

function x_k+1 = f(x, u)    # discrete-time dynamics

forward kinematics to glove
    - given state vector x [q1, q2, q1_dot, q2_dot]
    - return glove position wrt world frame [1, 2, 3]

target position doesn't change

cost function over a horizon
J = ...








*/