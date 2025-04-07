import numpy as np
from grow.reservoir import Reservoir
from util.consts import T


def spectral_radius(res: Reservoir):
    eigenvalues = np.linalg.eigvals(res.A)  
    return max(abs(eigenvalues))

def kernel_rank(res: Reservoir, 
                input: np.ndarray=None, 
                state: np.ndarray=None, 
                num_timesteps: int=T):
    if input is None:
        input = np.random.uniform(-1, 1, (1, num_timesteps)).astype(np.float64)
        input = np.repeat(input, res.input_units, axis=0)
    if state is None:
        res.reset()
        _ = res.run(input)
    return np.linalg.matrix_rank(res.reservoir_state[:, res.washout:])
    
def magnitude_generalization_rank(res: Reservoir,):
    pass

def linear_memory_capacity(res: Reservoir,
                           input: np.ndarray=None,
                           output: np.ndarray=None,
                           num_timesteps: int=T,
                           predictions: np.ndarray=None,
                           filter: float=0.0):
    """
    """
    sequence_length = num_timesteps // 2

    if input is None:
        input = np.random.uniform(-1, 1, (1, num_timesteps)).astype(np.float64)
        input = np.repeat(input, res.input_units, axis=0)

    if output is None:
        output = np.zeros((res.output_units, num_timesteps))
        for i in range(res.output_units):
            if i < num_timesteps:
                output[i, i:num_timesteps] = input[0, 0:num_timesteps - i]
            else:
                output[i, 0:num_timesteps - i] = input[0, i:num_timesteps]  # wrap around index

    # split
    train_input = input[:,:sequence_length]
    train_output = output[:,:sequence_length]

    test_input = input[:,sequence_length:]
    test_output = output[:,sequence_length:]
    test_output = test_output[:,res.washout:]

    # train the reservoir and get predictions if not provided
    if predictions is None:
        res.reset()  # reset the reservoir state
        _ = res.train(train_input, target=train_output) 
        predictions = res.run(test_input)  # run the test input through the reservoir

    # compute the linear memory capacity as the sum of r^2 scores across delays
    
    memory_capacities = []
    for i in range(res.output_units):
        mean_output = np.mean(test_output[i, :])
        mean_predict = np.mean(predictions[i, :])
        sz = predictions.shape[1]

        covariance = np.mean((test_output[i, :] - mean_output) * 
                      (predictions[i, :] - mean_predict) / (sz - 1))
        prediction_variance = np.var(predictions[i, :])
        input_variance = np.var(test_input[i, res.washout:])

        # Memory capacity calculation
        memory_capacity = (covariance ** 2) / (input_variance * prediction_variance)
        if memory_capacity < filter:
            memory_capacity = 0.0
        memory_capacities.append(memory_capacity)

    return np.sum(memory_capacities)
        
    
