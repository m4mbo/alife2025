import numpy as np
from util.consts import T
from reservoirpy.datasets import santafe_laser, narma


def narmax(order=10, t=None, discard=20):
    """
    Creates NARMA-X sequence for t timesteps, where X is the order of the system.
    """
    if not t:
        t = T+discard
    # input
    u = np.random.uniform(0, 0.5, (t+order, 1)).astype(np.float64)
    y = narma(n_timesteps=t, order=order, u=u)
    # discard transient effects from first 20 steps
    u = u[discard+order:]
    y = y[discard:]
    return u.T, y.T

def santa_fe(t=None, discard=20):
    """
    Creates NARMA-X sequence for t timesteps, where X is the order of the system.
    """
    if not t:
        t = T+discard
    # input
    u = np.random.uniform(0, 0.5, t).astype(np.float64)
    # NARMA sequence
    y = santafe_laser()[:t].T

    # discard transient effects from first 20 steps
    u = u[np.newaxis, discard:]
    y = y[:, discard:]

    return u, y


    




