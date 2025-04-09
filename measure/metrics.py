import numpy as np
from grow.reservoir import Reservoir


def spectral_radius(res: Reservoir):
    eigenvalues = np.linalg.eigvals(res.A)  
    return max(abs(eigenvalues))

def effective_rank(singular_values: np.ndarray, threshold: float = 0.99) -> int:
    """
    Computes the number of singular values required to 
    capture the specified percentage of the total sum.
    """
    full_sum = np.sum(singular_values)
    cutoff = threshold * full_sum
    tmp_sum = 0.0
    e_rank = 0

    for val in singular_values:
        tmp_sum += val
        e_rank += 1
        if tmp_sum >= cutoff:
            break
    return e_rank

def kernel_rank(res: Reservoir, 
                num_timesteps: int=2000):
    """
    Computes the number of non-zero 
    singular values of the reservoir state matrix.
    """
    # generate random input signal in [-1, 1]
    ui = 2 * np.random.rand(num_timesteps) - 1
    input = np.tile(ui[:, None], (1, res.input_units)).T.astype(np.float64)

    res.reset()
    _ = res.run(input)

    state = res.reservoir_state[:, res.washout:]
    s = np.linalg.svd(state, compute_uv=False)
    return effective_rank(s)
    
def generalization_rank(res: Reservoir, 
                        num_timesteps: int=2000):
    """
    Computes the Magnitude Generalizationr Rank (MGR).
    """
    # generate random input signal in [0.45, 0.55]
    input = 0.5 + 0.1 * np.random.rand(res.input_units, num_timesteps) - 0.05

    res.reset()
    _ = res.run(input)

    state = res.reservoir_state[:, res.washout:]
    s = np.linalg.svd(state, compute_uv=False)
    return effective_rank(s)

def linear_memory_capacity(res: Reservoir,
                           num_timesteps: int=2000,
                           max_delay: int=None,
                           filter: float=0.1):
    """
    Computes the linear memory capacity (MC) of a SISO reservoir
    by training it to reproduce delayed versions of the input.
    """
    if not max_delay:
        max_delay = res.size()
    
    sequence_length = num_timesteps // 2
    total_length = num_timesteps + max_delay + 1

    # random input sequence
    input_signal = 2 * np.random.rand(1, total_length) - 1

    # input and delayed targets
    input_sequence = input_signal[:, max_delay:max_delay + num_timesteps].T
    target_delays = np.zeros((num_timesteps, max_delay))

    for delay in range(1, max_delay + 1):
        target_delays[:, delay - 1] = input_signal[:, max_delay - delay: max_delay + num_timesteps - delay].T[:, 0]

    # training and testing sets
    train_input = input_sequence[:sequence_length].T
    test_input = input_sequence[sequence_length:].T

    train_target = target_delays[:sequence_length].T
    test_target = target_delays[sequence_length:]
    test_target = test_target[res.washout:, :]

    res.reset()
    _ = res.train(input=train_input, target=train_target)
    predictions = res.run(test_input)
    
    # compute mc
    memory_capacities = []

    for i in range(max_delay):
        y_true = test_target[:, i]
        y_pred = predictions[i, :]

        cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
        var_pred = np.var(y_pred)
        var_true = np.var(y_true)


        denom = var_true * var_pred
        mc_k = (cov ** 2) / denom if denom != 0 else 0.0

        memory_capacities.append(mc_k if mc_k > filter else 0.0)

    memory_capacities = np.nan_to_num(memory_capacities, nan=0.0)
    return np.sum(memory_capacities)