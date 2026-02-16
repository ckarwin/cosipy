import itertools
from typing import Optional, Iterable

import numpy as np
from astromodels import Model, PointSource, ExtendedSource

from cosipy.interfaces import UnbinnedThreeMLModelFoldingInterface, UnbinnedThreeMLSourceResponseInterface
from cosipy.response.threeml_response import ThreeMLModelFoldingCacheSourceResponsesMixin

class UnbinnedThreeMLModelFolding(UnbinnedThreeMLModelFoldingInterface, ThreeMLModelFoldingCacheSourceResponsesMixin):

    def __init__(self,
                 point_source_response = UnbinnedThreeMLSourceResponseInterface,
                 extended_source_response: UnbinnedThreeMLSourceResponseInterface = None):

        # Interface inputs
        self._model = None

        # Implementation inputs
        self._psr = point_source_response
        self._esr = extended_source_response

        if (self._psr is not None) and (self._esr is not None) and self._psr.event_type != self._esr.event_type:
            raise RuntimeError("Point and Extended Source Response must handle the same event type")

        self._event_type = self._psr.event_type

        # Cache
        # Each source has its own cache.
        # We could cache the sum of all sources, but I thought
        # it was not worth it for the typical use case. Usually
        # at least one source changes in between call
        self._cached_model_dict = None
        self._source_responses = {}

    @property
    def event_type(self):
        return self._event_type

    def set_model(self, model: Model):
        """
        The model is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        self._model = model

    def expected_counts(self) -> float:
        """
        Total expected counts
        """

        self._cache_source_responses()

        return sum(s.expected_counts() for s in self._source_responses.values())

    def expectation_density(self) -> Iterable[float]:
        """
        Sum of expectation density
        """

        self._cache_source_responses()

        return [sum(expectations) for expectations in zip(*(s.expectation_density() for s in self._source_responses.values()))]
