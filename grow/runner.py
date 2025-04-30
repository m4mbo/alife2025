# Class from Waldegrave et al. (2024): https://doi.org/10.1162/isal_a_00734
# Copyright (c) 2024, Riversdale Waldegrave


from grow.reservoir import Reservoir
from grow.dgca import DGCA


def rindex(it, li):
    """
    Reverse index() ie.
    index of last occurence of item in list
    """
    return len(li) - 1 - li[::-1].index(it)


class Runner:
    """
    To hold the run params and check if we have entered an attractor
    (so can stop early).
    """

    def __init__(self, max_steps: int, max_size: int) -> None:
        self.max_steps = max_steps
        self.max_size = max_size
        self.graphs: list[Reservoir] = []
        self.hashes: list[int] = []
        self.ids: list[int] = []
        self.status = 'ready'
        self.hash_offset = 1
    
    def record(self, res: Reservoir, duplicate_of: int | None = None) -> None:
        """
        Adds the reservoir and its hash to the records.
        """
        # figure out id
        if len(self.ids)==0:
            self.ids.append(0)
        elif duplicate_of or duplicate_of==0:
            # isomorphic to one we have already seen
            self.ids.append(self.ids[duplicate_of])
        else: 
            self.ids.append(max(self.ids)+1)
        self.graphs.append(res)
        self.hashes.append(res.state_hash())
        
    def reset(self):
        self.graphs: list[Reservoir] = []
        self.hashes: list[int] = []
        self.status = 'ready'

    def already_seen(self, G: Reservoir) -> tuple[bool,bool]:
        """
        Checks if we have already seen this graph.
        """
        this_hash = G.state_hash()
        if this_hash in self.hashes:
            match_idx = [i for i,x in enumerate(self.hashes) if x==this_hash]
            iso_tf = []
            for m in match_idx:
                try:
                    is_iso = G.is_isomorphic(self.graphs[m])
                    if not is_iso:
                        pass
                    iso_tf.append(is_iso)
                except TimeoutError:
                    # isomorphism check timed out. 
                    # assume it IS to be safe
                    iso_tf.append(True)
                    print('Warning: isomorphism check timed out')
            any_iso = any(iso_tf)
            if any_iso:
                idx = match_idx[iso_tf.index(True)]
            else:
                idx = None
            return any_iso, idx
        else:
            return False, None

    def run(self, dgca: DGCA, seed: Reservoir) -> Reservoir:
        """
        Runs for the full number of steps or stops early if the
        graph becomes too big or the system enters an attractor.
        """
        current = seed
        self.record(current)
        for _ in range(self.max_steps):
            next = dgca.step(current)
            if next.size() == 0:
                self.status = 'zero_nodes'
                self.record(next)
                return next
            if next.size() > self.max_size:
                self.status = 'max_size'
                self.record(next)
                return current # because next_graph is too big
            already_seen, match_idx = self.already_seen(next)
            if already_seen:
                # attractor reached, no need to keep going
                # add it to the record anyway so that we can spot the cycle
                self.status = 'attractor'
                self.record(next, duplicate_of=match_idx)
                return next
            self.record(next)
            current = next
        self.status = 'max_steps'
        return current

    def graph_size(self) -> list[int]:
        """
        Returns a list of the size of the graph at each step
        of the run.
        """
        if self.status == 'ready':
            print("Hasn't been run yet!")
        return [g.size() for g in self.graphs]
    
    def graph_connectivity(self) -> list[float]:
        if self.status == 'ready':
            print("Hasn't been run yet!")
        return [g.connectivity() for g in self.graphs]
    
    def attractor_info(self) -> tuple[int, int]:
        """
        Returns the transient and attractor length.
        An attractor length of zero means no attractor
        was found.
        """
        if self.status == 'ready':
            print("Hasn't been run yet!")
        if self.ids[-1] in self.ids[:-1]:
            ind = rindex(self.ids[-1], self.ids[:-1])
            attr_len = len(self.ids) - 1 - ind
        else:
            attr_len = 0
        trans_len = len(self.ids) - attr_len - 1
        return trans_len, attr_len