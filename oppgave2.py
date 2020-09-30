import numpy as np
import matplotlib.pyplot as plt


def dydt(y, t):
    return f


def runge_kutta_method(t0, y0, t, n):
    dt = (max(t) - t0) / float(n)
    # dt = 0.01
    n = int(n)
    y = y0

    for i in range(
        1, int(n + 1)
    ):  # her skal det egentlig stå n + 1 men fikk feilmelding.
        rk_1 = dt * dydt(t0, y)
        rk_2 = dt * dydt((t0 + (dt / 2)), (y + (rk_1 / 2)))
        rk_3 = dt * dydt((t0 + (dt / 2)), (y + (rk_2 / 2)))
        rk_4 = dt * dydt((t0 + (dt / 2)), (y + rk_3))

        y = y + (1 / 6) * (rk_1 + 2 * rk_2 + 2 * rk_3 + rk_4)
        t0 = t0 + dt
    return y


"""
def test_runge_kutta_method():
    expected = np.cos(2 * np.pi)
    # f = np.cos(2 * np.pi)
    dydt(np.cos(2 * np.pi), 2 * np.pi)
    computed = runge_kutta_method(0, 0, 2 * np.pi, 100)
    # print(computed)
    # print(expected)
    # plt.plot(1, expected)
    # plt.plot(1, computed)
    # plt.show()
    tol = 1e-9
    # assert abs(computed - expected) < tol
"""

t = np.linspace(0, 2 * np.pi, 50)
f = np.cos(t)
y = runge_kutta_method(0, 0, t, 50)
# print(np.cos(2 * np.pi))
print(y)  # printer 0.5403
# print(f)
# test_runge_kutta_method()
# plt.plot(t, f, label="expected", linewidth=4.3)
# plt.plot(t, o, label="computed")
# plt.legend()
# plt.show()


def dcndt(c, t):
    dt = 0.05
    who = 1.4
    w = 1.0
    h_ = 1
    me = 1
    e_ = 1
    En = h_ * w * (n + 0.5)
    Em = h_ * w * (m + 0.5)
    vdt = Vnm(t) * np.exp((i * (En - Em) * t) / h_)
