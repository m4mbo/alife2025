import numpy as np
from grow.reservoir import Reservoir, check_conditions
from measure.tasks import *
from measure.metrics import kernel_rank, generalization_measure, linear_memory_capacity, get_metrics


NRMSE = lambda y,y_fit: np.mean(((y-y_fit)**2)/np.var(y))

class ReservoirFitness:
    """
    Interface for a Callable which takes a graph and returns its fitnesss
    """
    def __init__(self, high_good: bool):
        self.high_good = high_good

    def __call__(self, res: Reservoir):
        raise NotImplementedError
    
class TaskFitness(ReservoirFitness):

    def __init__(self,
                 series: callable,
                 conditions: dict = dict(),
                 self_loops: bool = True,
                 verbose: bool = False,
                 order: int = None,
                 fixed_series: bool = True
                 ) -> None:
        super().__init__(high_good=False)   # error so low is good
        self.self_loops = self_loops
        self.conditions = conditions
        self.verbose = verbose
        self.skip_count = 0
        self.series = series
        self.order = order
        self.memo = {'fitness': [], 'graph': [], 'model': []}
        self.fixed_series = fixed_series
        self.input, self.target = self.series(order=self.order)

    def _generate_series(self):
        """Generate input and target series, depending on fixed_series flag."""
        if not self.fixed_series:
            return self.series(order=self.order) if self.order is not None else self.series()
        return self.input, self.target

    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = check_conditions(res, self.conditions, self.verbose)
        
        res_ = res.copy()
        if not self.self_loops:
            res_ = res.no_selfloops()
        
        if checks_ok:
            errors = []
            for _ in range(5):  # 5 measurements
                self.input, self.target = self._generate_series()
                res_.reset()
                predictions = res_.bipolar().train(self.input, target=self.target)
                err = np.nan if predictions is None else np.min((NRMSE(self.target[:, res.washout:], predictions), 1))
                errors.append(err)
            valid_errors = [e for e in errors if not np.isnan(e)]
            err = np.nanmean(errors) if valid_errors else np.nan
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        
        return err
    
class MetricFitness(ReservoirFitness):

    def __init__(self,
                 metric: str,
                 conditions: dict = None,
                 self_loops: bool = True,
                 verbose: bool = False
                 ) -> None:
        super().__init__(high_good=True)  
        
        self.metric = metric
        self.conditions = conditions or {} 
        self.self_loops = self_loops
        self.verbose = verbose
        self.skip_count = 0
        self.memo = {'fitness': [], 'graph': [], 'model': []}
    
    def _compute_metric(self, res):
        if self.metric == "KR":  
            return (kernel_rank(res) / res.size())
        if self.metric == "GM":
            return generalization_measure(res)
        if self.metric == "LMC":
            return linear_memory_capacity(res)
        if self.metric == "combined":
            kr, gm = get_metrics(res)
            print(kr, gm)
            return (kr / res.size()) + gm

    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = check_conditions(res, self.conditions, self.verbose)
        res_ = res.copy()

        if not self.self_loops:
            res_ = res.no_selfloops()

        if checks_ok:   
            err = self._compute_metric(res_)
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err