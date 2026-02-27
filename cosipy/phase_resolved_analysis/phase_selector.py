import logging
import numpy as np
import itertools
import os
from astropy.io import fits

logger = logging.getLogger(__name__)

# --- ROBUST IMPORTS (Safety Switch) ---
try:
    from cosipy.interfaces.event_selection import EventSelectorInterface
    from cosipy.interfaces import EventInterface
    from cosipy.util.iterables import itertools_batched
except (ImportError, AttributeError):
    class EventInterface: pass
    class EventSelectorInterface:
        def select(self, events): raise NotImplementedError
    def itertools_batched(iterable, n):
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch: return
            yield batch

class PhaseSelector(EventSelectorInterface):
    """
    Selects events based on pulsar phase. 
    Highly optimized for FITS files via NumPy vectorization.
    """
    def __init__(self, ephemeris_file, pstart, pstop, batch_size=10000):
        self.ephemeris_file = ephemeris_file 
        self.pstart = float(pstart)
        self.pstop = float(pstop)
        self._batch_size = batch_size

    def _get_vectorized_mask(self, phases: np.ndarray) -> np.ndarray:
        """Core logic applied across an entire NumPy array instantly."""
        pstop_norm = self.pstop % 1.0 if self.pstop > 1.0 else self.pstop
        
        if self.pstart <= pstop_norm:
            return (phases >= self.pstart) & (phases <= pstop_norm)
        else:
            return (phases >= self.pstart) | (phases <= pstop_norm)

    def select(self, events):
        """
        Maintains pipeline compatibility. Yields booleans.
        Optimized to use vectorized masks on batches.
        """
        # Fast path for single EventInterface object
        if isinstance(events, EventInterface):
            phase = getattr(events, 'pulse_phase', -1.0)
            if phase is None: return False
            return bool(self._get_vectorized_mask(np.array(phase)))

        # Fast path if events is already a structured NumPy array
        if isinstance(events, np.ndarray) and 'PULSE_PHASE' in events.dtype.names:
            return self._get_vectorized_mask(events['PULSE_PHASE'])

        # Fallback for generic iterables of objects
        for chunk in itertools_batched(events, self._batch_size):
            phases = np.array([getattr(e, 'pulse_phase', -1.0) for e in chunk])
            mask = self._get_vectorized_mask(phases)
            for sel in mask:
                yield bool(sel)

    def filter_events(self, events, output_fits=None, template_fits=None):
        """
        Filters events. Instantaneous execution when passed a FITS file path.
        """
        # --- VECTORIZED FAST PATH FOR FITS FILES ---
        if isinstance(events, str):
            logger.info(f"Auto-loading events from FITS: {events}")
            template_fits = events 
            
            with fits.open(events) as hdul:
                data = hdul[1].data
                # Apply mask to the entire 'PULSE_PHASE' column at once
                mask = self._get_vectorized_mask(data['PULSE_PHASE'])
                # Slice the FITS array instantly
                selected_data = data[mask] 
            
            if output_fits is not None:
                self._save_fits_fast(selected_data, output_fits, template_fits)
                
            return selected_data

        # --- SLOW PATH FOR PYTHON OBJECT LISTS ---
        mask = list(self.select(events))
        selected_events = [e for e, m in zip(events, mask) if m]
        
        if output_fits is not None:
            if template_fits is None:
                logger.warning("'template_fits' missing. Cannot save.")
            else:
                self.save_fits(selected_events, output_fits, template_fits)
                
        return selected_events

    def _save_fits_fast(self, structured_array, output_filename, template_filename):
        """Saves a NumPy structured array directly to FITS (Orders of magnitude faster)."""
        if len(structured_array) == 0:
            logger.warning("Warning: No events to save.")
            return

        logger.info(f"Saving {len(structured_array)} events to {output_filename}...")
        
        try:
            with fits.open(template_filename) as hdul:
                # Plop the filtered array directly into the new HDU
                hdu = fits.BinTableHDU(data=structured_array, header=hdul[1].header)
                hdul_new = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), hdu])
                hdul_new.writeto(output_filename, overwrite=True)
                logger.info(f"Successfully saved: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save FITS: {e}")

    def save_fits(self, events, output_filename, template_filename):
        """Legacy fallback for saving lists of Python objects."""
        # Auto-route to fast save if an array was passed by mistake
        if isinstance(events, np.ndarray):
            return self._save_fits_fast(events, output_filename, template_filename)
            
        if not events:
            logger.warning("Warning: No events to save.")
            return
            
        logger.info(f"Saving {len(events)} events (Legacy Object Mode) to {output_filename}...")
        try:
            with fits.open(template_filename) as hdul:
                columns = hdul[1].columns
                rows = [e.row for e in events if hasattr(e, 'row')]
                
                if not rows:
                    logger.error("Error: Event objects do not contain raw FITS rows.")
                    return

                new_data = fits.FITS_rec.from_columns(columns, nrows=len(rows))
                for i, row in enumerate(rows):
                    new_data[i] = row

                hdu = fits.BinTableHDU(data=new_data, header=hdul[1].header)
                hdul_new = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), hdu])
                hdul_new.writeto(output_filename, overwrite=True)
                logger.info(f"Successfully saved: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save FITS: {e}")