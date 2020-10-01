import numpy as np
import matplotlib.pyplot as plt


def f(t, y):
    return np.cos(t)


def runge_kutta_method(tn, yn, dt, f):
    rk_1 = dt * f(tn, yn)
    rk_2 = dt * f((tn + (dt / 2)), (yn + (rk_1 / 2)))
    rk_3 = dt * f((tn + (dt / 2)), (yn + (rk_2 / 2)))
    rk_4 = dt * f((tn + (dt / 2)), (yn + rk_3))
    y = yn + (1 / 6) * (rk_1 + 2 * rk_2 + 2 * rk_3 + rk_4)
    return y


"""
n = 1000
yn = np.zeros(n)
tn = np.linspace(0, 2 * np.pi, n)
dt = (tn[-1] - tn[0]) / n

for i in range(n - 1):
    yn[i + 1] = runge_kutta_method(tn[i], yn[i], dt, f)

plt.title("f(t,y) = cos(t)")
plt.plot(tn, np.sin(tn), label="expected", linewidth=4.3)
plt.plot(tn, yn, label="computed")
plt.legend()
plt.savefig("Oppgave 2c")
plt.show()

"""


def dcndt(c, t):

    h_ = 1
    q = -1
    me = 1
    e_ = 1
    En = h_ * w * (n + 0.5)
    Em = h_ * w * (m + 0.5)
    vdt = Vnm(t) * np.exp((i * (En - Em) * t) / h_)

    x_0 = np.sqrt(h_ / (me * w))

    diag = np.sqrt(np.arange(1, n))
    V0 = np.zeros([n, n], dtype=complex)
    V0 += np.diag(diag, 1) + np.diag(diag, -1)
    V0 = V0 * (x0 / (np.sqrt(2)))

    def V_dt(t):
        V = V0 * (-q) * e_0 * np.cos(w * t) * np.exp((1j(En - Em) * t) / h_)

    init = np.zeros(n)


# 1j = "i" i python
"""
dt = 0.01
who = 1.4
w = 1.0
e0_1 = 0.05
e0_2 = 0.2
e0_3 = 0.5
"""