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


n = 1
m = 0
# dt = 0.01
who = 1.4
w = 1.0
h_ = 1
q = 1  # samme som e i oppgaveheftet
me = 1
e_ = 1

e0_1 = 0.05
e0_2 = 0.2
e0_3 = 0.5

En = h_ * w * (n + 0.5)
Em = h_ * w * (m + 0.5)

x_0 = np.sqrt(h_ / (me * w))
n_states = 100
diag = np.sqrt(np.arange(1, n_states))
V0 = np.zeros([n_states, n_states], dtype=complex)
V0 += np.diag(diag, 1) + np.diag(diag, -1)
V0 *= x_0 / (np.sqrt(2))


def V_dt(t):
    V = V0 * (-q) * e0_1 * np.cos(w * t) * np.exp((1j * (En - Em) * t) / h_)
    return V


# init = np.zeros(n)


def Cn_dt(y, t):
    # må bytte ut cos(t) med -i/h_V't*C

    # t = 0
    # n_timestep = 0.01
    c = np.zeros([n_states, 1])
    c[0, 0] = 1
    # c1 = c * np.exp((1j * En * t) / h_)
    return -(1j) / h_ * V_dt * c * np.exp((1j * En * t) / h_)


# 1j = "i" i python

num = 1000
yn = np.zeros(num)
tn = np.linspace(0, 100, num)
dt = 0.01

for i in range(n - 1):
    yn[i + 1] = runge_kutta_method(tn[i], yn[i], dt, Cn_dt)
print(yn)
plt.title("f(t,y) = cos(t)")

plt.plot(tn, abs(yn) ** 2, label="computed")
plt.legend()
plt.show()

# dt = tn[1] - tn[0]
# c1 = np.zeros([100, n])
# c1[0, 0] = 1
# print(c1)