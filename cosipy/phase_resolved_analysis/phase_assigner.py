import numpy as np
import logging
from astropy.io import fits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhaseAssigner:
    """
    Reads a pulsar ephemeris and assigns PULSE_PHASE based on MET directly.
    """
    def __init__(self, par_file):
        self.params = self._parse_par_file(par_file)
        
        # Parse F0 (Frequency) or P0 (Period)
        if 'F0' in self.params:
            val = float(self.params['F0'])
            if val < 1.0: 
                self.f0 = 1.0 / val
                logger.warning(f"F0 < 1.0. Assuming Period. F0={self.f0:.6f} Hz")
            else:
                self.f0 = val
        elif 'P0' in self.params:
            self.f0 = 1.0 / float(self.params['P0'])
        else:
            raise ValueError("PAR file must have F0 or P0")
        
        # We removed T0/Epoch logic as requested.

    def _parse_par_file(self, path):
        p = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and not parts[0].startswith(('#', 'C')):
                    try: p[parts[0].upper()] = parts[1].replace('D','E')
                    except: pass
        return p

    def add_phase_column(self, input_fits, output_fits=None):
        """
        Calculates phase = (MET * F0) % 1.0 and adds column.
        """
        with fits.open(input_fits) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            
            # 1. Get Time (MET)
            try: times = data['TimeTags']
            except KeyError: times = data['TIME']

            # 2. Calculate Phase (Simple Folding)
            # Phase = (Time * Frequency) % 1.0
            phase = (times * self.f0) % 1.0
            
            # 3. Create or Overwrite Column
            cols = data.columns
            if 'PULSE_PHASE' in cols.names:
                logger.info("Overwriting PULSE_PHASE column.")
                data['PULSE_PHASE'] = phase
                new_hdu = fits.BinTableHDU(data=data, header=header)
            else:
                logger.info("Creating PULSE_PHASE column.")
                col = fits.Column(name='PULSE_PHASE', format='D', array=phase)
                new_hdu = fits.BinTableHDU.from_columns(cols + col, header=header)

            if output_fits is None: output_fits = input_fits
            fits.HDUList([hdul[0], new_hdu]).writeto(output_fits, overwrite=True)
            logger.info(f"Saved: {output_fits}")
            return output_fits