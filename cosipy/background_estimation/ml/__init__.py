from importlib.util import find_spec

if find_spec("torch") is None:
    raise RuntimeError("Install cosipy with [ml] optional packages to use these features.")

from .ContinuumEstimationNN import ContinuumEstimationNN
from .ContinuumEstimationNN import GCN
from .ContinuumEstimationNN import ContinuumEstimationInterp