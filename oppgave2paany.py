import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class RK:
    def __init__(self, M=1, L=1):
        """Initializes the attributes of the class

        Arguments
        -------------
        M : float
            The mass of the pendulum

        L : float
            The length of the rod

        theta : float
            The position of the pendulum
        """
        self.M = M
        self.L = L

    def __call__(self, t, y):
        """Returns the right hand side
        of the equations of motion

        Arguments
        -------------
        t : float
            Time

        y : tuple of floats
            The state variable
        """

        omega_dt = (-g / self.L) * np.sin(y[0])
        theta_dt = y[1]
        return theta_dt, omega_dt

    def solve(self, y0, T, dt, angles="rad"):
        """Solves the equations of motion

        Arguments
        -------------
        y0 : tuple of floats
            Initial state

        T : float
            Time

        dt : float
            Time step
        """
        if angles == "deg":
            y0[0] * (np.pi / 180)
            y0[1] * (np.pi / 180)
        t_ = np.arange(0, T, dt)
        solv = solve_ivp(self, [0, T], (y0), t_eval=t_)
        self._time = solv.t
        self._theta = solv.y[0]
        self._omega = solv.y[1]