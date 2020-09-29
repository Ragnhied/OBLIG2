import numpy as np


def dydt(y, t):
    return np.cos(t)


def runge_kutta_method(t0, y0, t, dt):
    n = np.arange(0, t, dt)
    y = y0

    for i in range(1, n + 1):
        rk_1 = dt * dydt(t0, y)
        rk_2 = dt * dydt((t0 + dt / 2), (y + rk_1 / 2))
        rk_3 = dt * dydt((t0 + dt / 2), (y + rk_2 / 2))
        rk_4 = dt * dydt((t0 + dt / 2), (y + rk_3))

        y = y + (1 / 6) * (rk_1 + 2 * rk_2 + 2 * rk_3 + rk_4)
        t0 = t0 + dt
    return y

def test_runge_kutta_method():
    expected = 0.540302
    f = 
    computed = runge_kutta_method()
