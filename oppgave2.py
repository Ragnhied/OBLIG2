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


# n = 1000
# yn = np.zeros(n)
# tn = np.linspace(0, 2 * np.pi, n)
# dt = (tn[-1] - tn[0]) / n

# for i in range(n - 1):
#     yn[i + 1] = runge_kutta_method(tn[i], yn[i], dt, f)

# plt.title("f(t,y) = cos(t)")
# plt.plot(tn, np.sin(tn), label="expected", linewidth=4.3)
# plt.plot(tn, yn, label="computed")
# plt.legend()
# plt.savefig("Oppgave 2c")
# plt.show()


who = 1.4
w = 1.0
h_ = 1
q = 1  # samme som e i oppgaveheftet
me = 1
e_ = 1

e0_1 = 0.05
e0_2 = 0.2
e0_3 = 0.5
n_states = 3

En = h_ * who * (n_states + 0.5)


num = 10000
T = 100

x_0 = np.sqrt(h_ / (me * w))

c = np.zeros([num, n_states], dtype=complex)
c[0, 0] = 1
# c1 = c * np.exp((1j * En * t) / h_)
# return -(1j) / h_ * V_dt * c * np.exp((1j * En * t) / h_)


# 1j = "i" i python
tn = np.linspace(0, T, num)
# dt = tn[1] - tn[0]
dt = 0.01


def f1(t, c):
    diag = np.sqrt(np.arange(1, n_states))
    V0 = np.zeros([n_states, n_states], dtype=complex)
    V0 += np.diag(diag, 1) + np.diag(diag, -1)
    # V0 *=
    V = V0 * (
        (x_0 / (np.sqrt(2))) * (-q) * e0_1 * np.cos(w * t) * np.exp((1j * who * t))
    )  # her mangler En osv
    # print(V)
    return (-1j / h_) * np.dot(
        V, c
    )  # * np.exp((1j * En * t) / h_)  # byttet ut En med e_0*cos(wt)


for i in range(num - 1):
    c[i + 1, :] = runge_kutta_method(tn[i], c[i, :], dt, f1)


def absorption_probability(t):
    diag = np.sqrt(np.arange(1, n_states))
    V0 = np.zeros([n_states, n_states], dtype=complex)
    V0 += np.diag(diag, 1) + np.diag(diag, -1)
    W_fi_2 = (V0 * (x_0 / np.sqrt(2)) * ((-q * e0_1) / 2)) ** 2
    return (W_fi_2 * np.sin(0.5 * (who - w) * t) ** 2) / (
        h_ ** 2 * (0.5 * (who - w)) ** 2
    )


c__ = np.abs(c) ** 2


# plt.plot(tn, absorption_probability(tn), label="Absorption probability")
plt.plot(tn, c__[:, 0], label="c0^2")
plt.plot(tn, c__[:, 1], label="c1^2")
plt.plot(tn, c__[:, 2], label="c2^2")
plt.legend()
plt.title("c0^2, c1^2, c2^2 with e0 = 0.05")
plt.savefig("2f e0=0,05.png")
plt.show()
