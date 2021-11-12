from casadi import *
'''
This function returns the integration error, following a backward Euler
integration scheme. Constraints in the optimization problem impose this
error to be nul.

If x is the state value at time t, then xplus is the state value at t+1.
The backward Euler equation can be formulated as:
u(t+1) = (x(t+1) - x(t))/dt, and therefore the error is given by:
error = (x(t+1) - x(t)) - u(t+1)dt.
'''

def eulerIntegrator(x,xplus,uplus,dt):

    return (xplus - x) - uplus*dt


def rkIntegrator(X,Xp,U,f,dt):
    
    k1 = f(X,         U)
    k2 = f(X+dt/2*k1, U)
    k3 = f(X+dt/2*k2, U)
    k4 = f(X+dt*k3,   U)
    x_next = X + dt/6*(k1+2*k2+2*k3+k4) 
    return (Xp - X)