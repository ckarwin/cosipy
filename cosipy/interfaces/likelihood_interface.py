from typing import Protocol, runtime_checkable

__all__ = ['LikelihoodInterface',
           'BinnedLikelihoodInterface',
           'UnbinnedLikelihoodInterface']

from .expectation_interface import ExpectationDensityInterface, BinnedExpectationInterface, ExpectationInterface
from .data_interface import BinnedDataInterface, DataInterface, EventDataInterface
from .background_interface import BackgroundDensityInterface, BinnedBackgroundInterface, BackgroundInterface

@runtime_checkable
class LikelihoodInterface(Protocol):
    def get_log_like(self) -> float:...
    @property
    def nobservations(self) -> int:
        """For BIC and other statistics"""

@runtime_checkable
class BinnedLikelihoodInterface(LikelihoodInterface, Protocol):
    """

    """

@runtime_checkable
class UnbinnedLikelihoodInterface(LikelihoodInterface, Protocol):
    """
    """
