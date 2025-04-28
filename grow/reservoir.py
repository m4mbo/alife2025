import numpy as np  
import warnings
import graph_tool.all as gt
from grow.graph import GraphDef
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt


POLARITY_MATRIX = np.array([
    [1, 1, -1],  # state_from 0
    [-1, 1, 1],  # state_from 1
    [1, -1, 1]   # state_from 2
])

# activation functions
def linear(x):
    return x

def tanh(x):
    return np.tanh(x)

ACTIVATION_TABLE = np.array([tanh, tanh, linear])


def check_conditions(res: 'Reservoir',
                     conditions: dict, 
                     verbose: bool=False) -> bool:
    size = res.size()
    conn = res.connectivity()
    frag = res.get_largest_component_frac()
    if 'max_size' in conditions:
        # should already be fine but double check
        if size > conditions['max_size']:
            if verbose:
                print('Reservoir too big (should not happen!)')
            return False
    if 'io_path' in conditions:
        assert res.input_nodes and res.output_nodes
        if not res.io_path() and conditions['io_path']:
            if verbose:
                print('No I/O path.')
            return False
    if 'min_size' in conditions:
        if size < conditions['min_size']:
            if verbose:
                print('Reservoir too small.')
            return False
    if 'min_connectivity' in conditions:
        if conn < conditions['min_connectivity']:
            if verbose:
                print('Reservoir too sparse')
            return False
    if 'max_connectivity' in conditions:
        if conn > conditions['max_connectivity']:
            if verbose:
                print('Reservoir too dense')
            return False
    if 'min_component_frac' in conditions:
        if frag < conditions['min_component_frac']:
            if verbose:
                print('Reservoir too fragmented')
            return False
    if verbose:
        print(f'Reservoir OK: size={size}, conn={conn*100:.2f}%, frag={frag:.2f}')
    return True

def dfs_directed(A: np.ndarray, current: int, visited: set) -> bool:
    """
    Perform a recursive DFS on a directed adjacency matrix
    """
    if A.shape[0] == 0:
        return False
    # current node as visited
    visited.add(current) 
    # visit neighbors
    neighbors = np.nonzero(A[current])[0]  # directed neighbors
    for neighbor in neighbors:
        if neighbor not in visited:
            if dfs_directed(A, neighbor, visited):
                return True
    return False

def get_seed(input_nodes: int, 
             output_nodes: int, 
             n_states: int) -> 'Reservoir':
    
    if input_nodes or output_nodes:
        n_nodes = input_nodes + output_nodes + 1
        A = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # input nodes
        for i in range(input_nodes):
            A[i, -1] = 1
        # output nodes
        for i in range(input_nodes, input_nodes+output_nodes):
            A[-1, i] = 1

        S = np.zeros((n_nodes, n_states), dtype=int)  
        S[:, 0] = 1
    else:
        A = np.array([[0]])
        S = np.zeros((1, n_states), dtype=int)  
        S[0, 0] = 1
    return Reservoir(A, S, input_nodes=input_nodes, output_nodes=output_nodes)


class Reservoir(GraphDef):

    def __init__(self, A: np.ndarray, 
                 S: np.ndarray, 
                 input_nodes: int=0, 
                 output_nodes: int=0, 
                 input_units: int=1,
                 output_units: int=1,
                 input_gain=0.1, 
                 feedback_gain=0.95, 
                 washout=20):   
        super().__init__(A, S)
        self.input_nodes = input_nodes  # number of fixed I/O nodes
        self.output_nodes = output_nodes 
        self.input_units = input_units
        self.output_units = output_units
        self.input_gain = input_gain
        self.feedback_gain = feedback_gain
        self.washout = washout
        self.reset()

    def _pp(self, 
           g: gt.Graph, 
           pos: gt.VertexPropertyMap = None) -> gt.VertexPropertyMap:
        """
        Pretty prints the input/output nodes.
        Handles positioning of nodes, ensuring consistent spacing for I/O nodes
        and dynamic layout for other nodes. Also adjusts the outlines of I/O nodes.
        """
        # assign colors based on states
        states_1d = self.states_1d()
        cmap = plt.get_cmap('gray', self.n_states + 1)
        state_colors = cmap(states_1d)
        g.vp['plot_color'] = g.new_vertex_property('vector<double>', state_colors)
        
        # determine I/O nodes
        input_nodes = list(range(self.input_nodes)) if self.input_nodes > 0 else []
        output_nodes = list(range(self.input_nodes, self.input_nodes+self.output_nodes)) if self.output_nodes > 0 else []
        other_nodes = [v for v in g.vertices() if int(v) not in input_nodes + output_nodes]

        # use sfdp_layout for the general layout
        other_pos = gt.sfdp_layout(g, pos=pos)

        # initialize vertex measure
        outline_color = g.new_vertex_property("vector<double>")
        pos = g.new_vertex_property("vector<double>")

        # outline colors
        for v in other_nodes:
            outline_color[v] = [0, 0, 0, 0]     # transparent
        for i, v in enumerate(input_nodes):
            outline_color[v] = [1, 0, 0, 0.8]   # red
        for i, v in enumerate(output_nodes):
            outline_color[v] = [0, 0, 1, 0.8]   # blue

        # position
        if input_nodes and output_nodes:

            # if both are present, special layout
            x_min, x_max = float("inf"), float("-inf")
            y_min, y_max = float("inf"), float("-inf")
            for v in other_nodes:
                x, y = other_pos[v]
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # handle case where y_min == y_max, probably only one node
            if y_min == y_max:
                y_min -= 1
                y_max += 1

            # dynamic offsets
            input_x = x_min - 2.0  # left
            output_x = x_max + 2.0  # right
            spacing = max(abs(y_max - y_min) / max(len(input_nodes), len(output_nodes)), 1.0)  # vertical spacing

            # center nodes vertically around graph middle
            center_y = (y_min + y_max) / 2.0
            total_height_input = spacing * (len(input_nodes) - 1)
            total_height_output = spacing * (len(output_nodes) - 1)

            # assign positions and outline colors for input/output nodes
            for i, v in enumerate(input_nodes):
                pos[g.vertex(v)] = (input_x, center_y - total_height_input / 2 + i * spacing)
            for i, v in enumerate(output_nodes):
                pos[g.vertex(v)] = (output_x, center_y - total_height_output / 2 + i * spacing)
            for i, v in enumerate(other_nodes):
                pos[v] = other_pos[v]
        else:
            for v in g.vertices():
                pos[v] = other_pos[v]

        # edge colors
        edge_colors = g.new_edge_property("vector<double>")
        for e in g.edges():
            weight = g.ep.wgt[e]  # Assuming weights are stored as edge property 'wgt'
            if weight > 0:
                edge_colors[e] = [0, 0, 0, 1]  # Black for positive weights
            else:
                edge_colors[e] = [1, 0, 0, 1]  # Red for negative weights

        g.vp['outline_color'] = outline_color
        g.vp['pos'] = pos
        g.ep['edge_color'] = edge_colors
    
    def io_path(self) -> bool:
        """
        Check if there is a path from every input node to at least one output node.
        """
        input_nodes = range(self.input_nodes)
        output_nodes = range(self.input_nodes, self.input_nodes+self.output_nodes)
        
        for input_node in input_nodes:
            visited = set()
            # dfs from input
            dfs_directed(self.A, input_node, visited)
            if not any(output_node in visited for output_node in output_nodes):
                return False
        return True
    
    def bipolar(self) -> 'Reservoir':
        """
        Return a copy of the graph with weights converted to bipolar (-1,1) from one-hot encoding.
        """
        node_states = np.array(self.states_1d())
        # row and column indices of edges
        rows, cols = np.nonzero(self.A)
        # map the states of the source and target nodes 
        states_from = node_states[rows]
        states_to = node_states[cols]
        if len(states_to) == 0 or len(states_from) == 0:
            return Reservoir(self.A, self.S, self.input_nodes, self.output_nodes)
        # vectorize to weights
        new_weights = POLARITY_MATRIX[states_from, states_to]

        A_new = np.zeros_like(self.A)
        A_new[rows, cols] = new_weights  
        return Reservoir(A_new, self.S, self.input_nodes, self.output_nodes)
    
    def no_islands(self) -> 'Reservoir':
        """
        Returns a copy of the graph in which all isolated group of nodes 
        (relative to the input nodes) have been removed
        """
        # no input nodes
        if (self.input_nodes+self.output_nodes == 0) or self.size() == 0:
            return self.copy()
        
        # only one big chunk of cc
        n_cc, _ = connected_components(self.A, directed=False)
        if n_cc == 1:
            return self.copy()
                
        input_nodes = range(self.input_nodes)
        reachable_mask = np.zeros(self.A.shape[0], dtype=bool)
        
        # dfs from each input node
        for input_node in input_nodes:
            visited = set()
            dfs_directed(self.A, input_node, visited)
            reachable_mask[list(visited)] = True

        # I/O nodes are protected
        for node in range(self.input_nodes+self.output_nodes):
            reachable_mask[node] = True
    
        final_A = self.A[reachable_mask][:, reachable_mask]
        final_S = self.S[reachable_mask]
        return Reservoir(final_A, final_S, self.input_nodes, self.output_nodes)

    def train(self, input: np.ndarray, target: np.ndarray):
        """
        Trains a MIMO reservoir computing model using Bayesian Ridge Regression.
        """
        _, input_time_steps = input.shape
        output_dim, target_time_steps = target.shape

        if self.reservoir_state.shape[1] != input_time_steps:
            self.reset(input_time_steps)

        if input_time_steps != target_time_steps:
            raise ValueError("Input and target sequences must have the same length.")

        # N' = (N+1) or (output_nodes+1)
        state = self._run(input, bias=True)  # N' x T

        X = state.T  # T - washout x N'

        w_out = np.zeros((output_dim, X.shape[1]))  # output_dim x N'

        for i in range(output_dim):
            y = target[i, self.washout:]  
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = BayesianRidge(max_iter=3000, tol=1e-6, fit_intercept=False)
                    model.fit(X, y)
                    w_out[i, :] = model.coef_
            except Exception as e:
                print(f"Training failed for output {i}: {e}")
                w_out[i, :] = 0.0

        predictions = w_out @ state  # output_dim x T
        
        # remove the bias node
        self.w_out = w_out[:, :-1]

        return predictions

    def run(self, input):
        time_steps = input.shape[1]
        # check if input sequence length matches previous length
        if self.reservoir_state.shape[1] != time_steps:
            self.reset(time_steps)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            predictions = self.w_out @ self._run(input)

        predictions = np.nan_to_num(predictions, nan=0.0)

        return predictions
    
    def _run(self, input, bias=False):
        """
        Helper function for run. Runs the reservoir without the output layer.
        """
        node_states = self.states_1d()

        # running the reservoir
        for i in range(input.shape[1]):

            # TODO: linear nodes can cause overflows, suppress warnings for now
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                input_contribution = self.input_gain * self.w_in.T @ input[:, i]
                feedback_contribution = self.feedback_gain * self.A.T @ self.reservoir_state[:, i-1] if i > 0 else 0
                raw_state = input_contribution + feedback_contribution

            # activation function for each node
            for node_idx in range(self.size()):
                activation_function = ACTIVATION_TABLE[node_states[node_idx]]
                self.reservoir_state[node_idx, i] = np.nan_to_num(activation_function(raw_state[node_idx]), nan=0.0)
        
        filtered_state = self.reservoir_state
        
        # filtering only for output node states
        if self.output_nodes:
            filtered_state = filtered_state[self.input_nodes:self.input_nodes+self.output_nodes, :]
        
        # add bias node
        if bias:
            filtered_state = np.concatenate((filtered_state, np.ones((1, filtered_state.shape[1]))),axis=0) 

        return filtered_state[:, self.washout:]
 
    def reset(self, state_dim: int=2000):
        self.reservoir_state = np.zeros((self.size(), state_dim))
        self.w_in = np.random.randint(-1, 2, (self.input_units, self.size()))
        # self.w_in = np.random.uniform(-1, 1, (self.input_units, self.size()))
        # masking input nodes
        if self.input_nodes:
            self.w_in[:, self.input_nodes:] = 0 
        self.w_out = np.zeros((self.output_units, self.output_nodes if self.output_nodes else self.size()))
       
    def no_selfloops(self) -> 'Reservoir':
        """
        Returns a copy of the graph in which all self-loops have been removed
        """
        out_A = self.A.copy()
        # set values on the diagonal to zero
        out_A[np.eye(out_A.shape[0], dtype=np.bool_)] = 0 
        return Reservoir(out_A, np.copy(self.S), self.input_nodes, self.output_nodes)
    
    def copy(self):
        return Reservoir(np.copy(self.A), np.copy(self.S), self.input_nodes, self.output_nodes)

    