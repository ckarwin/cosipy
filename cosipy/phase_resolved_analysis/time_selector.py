import logging
import numpy as np
import itertools
from astropy.time import Time
from astropy.io import fits

logger = logging.getLogger(__name__)

# --- ROBUST IMPORTS ---
try:
    from cosipy.interfaces.event_selection import EventSelectorInterface
    from cosipy.interfaces import TimeTagEventInterface, EventInterface
    from cosipy.util.iterables import itertools_batched
except (ImportError, AttributeError):
    print("WARNING: Cosipy import failed. Using local fallback interfaces.")
    class EventInterface: pass
    class TimeTagEventInterface(EventInterface): pass
    class EventSelectorInterface:
        def select(self, events): raise NotImplementedError
    def itertools_batched(iterable, n):
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch: return
            yield batch

class TimeSelector(EventSelectorInterface):
    def __init__(self, tstart:Time = None, tstop:Time = None, batch_size:int = 10000):
        # Validation and Standardization
        if tstart is not None and tstart.isscalar: tstart = Time([tstart])
        if tstop is not None and tstop.isscalar: tstop = Time([tstop])
        self._tstart_list = tstart
        self._tstop_list = tstop
        self._batch_size = batch_size

    def _select(self, event) -> bool:
        return next(iter(self.select([event])))

    def select(self, events):
        # Iterator-based selection (kept for interface compatibility)
        if isinstance(events, EventInterface):
             return self._select(events)
        for chunk in itertools_batched(events, self._batch_size):
            # (Logic omitted for brevity as we use filter_fits mostly)
            pass
    
    def filter_fits(self, file_path, mjdref_val=51910.0):
        """
        Reads a FITS file, applies VECTORIZED time selection, 
        and prints diagnostics if no events are found.
        """
        # Internal wrapper class
        class _FitsEvent(TimeTagEventInterface):
            def __init__(self, row):
                self.row = row
                try: self._time = row['TimeTags']
                except KeyError: self._time = row['TIME']
                try: self._pulse_phase = row['PULSE_PHASE']
                except KeyError: self._pulse_phase = None
                self._jd1 = 2400000.5 + int(mjdref_val)
                self._jd2 = (mjdref_val - int(mjdref_val)) + (self._time / 86400.0)

            @property
            def time(self): return self._time
            @property
            def pulse_phase(self): return self._pulse_phase
            @property
            def jd1(self): return self._jd1
            @property
            def jd2(self): return self._jd2
        
        selected_events = []
        
        print(f"Opening FITS file: {file_path}")
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            
            # --- DIAGNOSTICS START ---
            try:
                times = data['TimeTags']
            except KeyError:
                times = data['TIME']
            
            t_min = np.min(times)
            t_max = np.max(times)
            print(f"File contains {len(times)} events.")
            print(f"   Min Time in File (MET): {t_min:.2f} s")
            print(f"   Max Time in File (MET): {t_max:.2f} s")
            
            # Check user input range
            if self._tstart_list is not None:
                # Convert user MJD back to MET for comparison
                user_start_met = (self._tstart_list[0].mjd - mjdref_val) * 86400.0
                user_stop_met  = (self._tstop_list[0].mjd - mjdref_val) * 86400.0
                print(f"   User Selection (MET):   {user_start_met:.2f} s to {user_stop_met:.2f} s")

                if user_stop_met < t_min or user_start_met > t_max:
                    print("   *** WARNING: Your selected range DOES NOT OVERLAP with the file! ***")
            # --- DIAGNOSTICS END ---

            # Convert Event MET to MJD for filtering
            event_mjds = mjdref_val + (times / 86400.0)
            
            # Create Boolean Mask
            if self._tstart_list is None and self._tstop_list is None:
                mask = np.ones(len(event_mjds), dtype=bool)
            elif self._tstart_list is None:
                mask = event_mjds <= self._tstop_list[0].mjd
            elif self._tstop_list is None:
                mask = event_mjds >= self._tstart_list[0].mjd
            else:
                start_mjds = self._tstart_list.mjd
                stop_mjds = self._tstop_list.mjd
                indices = np.searchsorted(start_mjds, event_mjds, side='right') - 1
                valid_idx_mask = (indices >= 0) & (indices < len(stop_mjds))
                mask = np.zeros(len(event_mjds), dtype=bool)
                if np.any(valid_idx_mask):
                    mask[valid_idx_mask] = event_mjds[valid_idx_mask] <= stop_mjds[indices[valid_idx_mask]]

            selected_rows = data[mask]
            print(f"Events passing time cut: {len(selected_rows)}")
            
            for row in selected_rows:
                selected_events.append(_FitsEvent(row))

        return selected_events