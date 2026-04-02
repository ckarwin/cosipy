import logging
logger = logging.getLogger(__name__)

import itertools
from typing import Union, Iterable

import numpy as np
from astropy.time import Time

from cosipy.interfaces import TimeTagEventInterface, EventInterface, TimeTagEventDataInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import itertools_batched


class TimeSelector(EventSelectorInterface):

    def __init__(self, tstart:Time = None, tstop:Time = None, batch_size:int = 10000):
        """
        Assumes events are time-ordered
        
        Selects events that fall within ANY of the time intervals defined by
        corresponding pairs of (tstart, tstop).

        Valid combinations:
        - (None, None): No time constraints
        - (Scalar, None): Single lower bound only
        - (None, Scalar): Single upper bound only  
        - (Scalar, Scalar): Single time interval
        - (List, List): Multiple time intervals (same length required)

        Parameters
        ----------
        tstart: Time, scalar Time, or None
            Start time(s). If list, tstop must also be a list of same length.
        tstop: Time, scalar Time, or None
            Stop time(s). If list, tstart must also be a list of same length.
        batch_size: int, default 10000
            Number of events to process at once
        """
        if tstart is not None and tstop is not None:
            if not tstart.isscalar == tstop.isscalar:
                logger.error("tstart and tstop must both be scalar or both be list.")
                raise ValueError

        elif tstart is None and tstop is not None:
            if tstop.isscalar == False:
                logger.error("When tstart is None, tstop must not be a list.")
                raise ValueError

        elif tstart is not None and tstop is None:
            if tstart.isscalar == False:
                logger.error("When tstop is None, tstart must not be a list.")
                raise ValueError
        
        # tstart is None and tstop is None -> OK.
        
        # Convert scalars to lists for uniform processing
        if tstart is not None and tstart.isscalar == True:
            tstart = Time([tstart])

        if tstop is not None and tstop.isscalar == True:
            tstop = Time([tstop])
        
        # length check
        if tstart is not None and tstop is not None:
            if len(tstart) != len(tstop):
                logger.error(f"tstart and tstop must have same length.")
                raise ValueError

        self._tstart_list = tstart
        self._tstop_list = tstop

        self._batch_size = batch_size
    
    @classmethod
    def from_gti(cls, gti, batch_size:int = 10000):
        """
        Instantiate a multi time selector from good time intervals.

        Parameters
        ----------
        gti: 
            Good time intervals object with tstart_list and tstop_list attributes
        batch_size: int
            Number of events to process at once
        """
        tstart_list = gti.tstart_list
        tstop_list = gti.tstop_list

        selector = cls(tstart_list, tstop_list, batch_size)

        return selector

    def _select(self, events:TimeTagEventDataInterface, early_stop:bool = True) -> Iterable[bool]:

        # Working in chunks/batches.
        # This can optimized based on the system and if events is pre-cached
        # (e.g. events.jd1 and events.jd2 are numpy arrays)

        for chunk in itertools_batched(events, self._batch_size):

            jd1 = []
            jd2 = []

            for event in chunk:
                jd1.append(event.jd1)
                jd2.append(event.jd2)

            time = Time(jd1, jd2, format = 'jd')

            if self._tstart_list is None and self._tstop_list is None:
                result = np.ones(len(time), dtype=bool)

            elif self._tstart_list is None:
                result = time <= self._tstop_list[0]

            elif self._tstop_list is None:
                result = time > self._tstart_list[0]

            else:
                indices = np.searchsorted(self._tstart_list, time, side='right') - 1
                valid = (indices >= 0) & (indices < len(self._tstop_list))
                result = np.zeros(len(time), dtype=bool)
                result[valid] = time[valid] <= self._tstop_list[indices[valid]]

            for sel in result:
                yield sel

            if early_stop and (self._tstop_list is not None and len(time) > 0) and time[-1] > self._tstop_list[-1]:
                # Stop further loading of event
                return
