# hpt.py
import itertools

class GridSampler:
    def __init__(self, space: dict):
        self.space = space
        # normalize to lists
        keys = list(space.keys())
        values = [[v] if not isinstance(v, (list, tuple)) else list(v) for v in space.values()]
        self._items = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    def __iter__(self):
        return iter(self._items)
