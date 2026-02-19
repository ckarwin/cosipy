import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from astropy.coordinates import SkyCoord

from astropy import units as u
from astropy.units import Quantity

from histpy import Histogram
from scoords import SpacecraftFrame

from cosipy.interfaces import EventInterface
from cosipy.interfaces.photon_parameters import PhotonListWithDirectionInSCFrameInterface
from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface, EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldInstrumentResponseFunctionInterface, \
    FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonInterface, PhotonWithDirectionAndEnergyInSCFrameInterface, PhotonListWithDirectionInterface, PhotonListWithDirectionAndEnergyInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.response import FullDetectorResponse
from cosipy.response.NNResponse import NNResponse
from cosipy.util.iterables import itertools_batched, asarray
from operator import attrgetter

class UnpolarizedNNFarFieldInstrumentResponseFunction(FarFieldSpectralInstrumentResponseFunctionInterface):
    
    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface
    
    def __init__(self, response: NNResponse,):
        if response.is_polarized:
            raise ValueError("The provided NNResponse is polarized, but UnpolarizedNNFarFieldInstrumentResponseFunction only supports unpolarized responses.")
        self._response = response
    
    @staticmethod
    def _get_context(photons: PhotonListWithDirectionAndEnergyInSCFrameInterface):
        lon = asarray(photons.direction_lon_rad_sc, dtype=np.float32)
        lat = asarray(photons.direction_lat_rad_sc, dtype=np.float32)
        en  = asarray(photons.energy_keV, dtype=np.float32)

        num_photons = lon.shape[0]
        context = torch.empty((num_photons, 3), dtype=torch.float32)

        context[:, 0] = torch.from_numpy(lon)
        context[:, 1] = torch.from_numpy(lat)
        context[:, 2] = torch.from_numpy(en)
        
        context[:, 1].mul_(-1).add_(np.pi/2)
        
        return context
    
    @staticmethod
    def _get_source(events: EmCDSEventDataInSCFrameInterface):
        lon = asarray(events.scattered_lon_rad_sc, dtype=np.float32)
        lat = asarray(events.scattered_lat_rad_sc, dtype=np.float32)
        phi = asarray(events.scattering_angle_rad, dtype=np.float32)
        en  = asarray(events.energy_keV, dtype=np.float32)
        
        num_events = lon.shape[0]
        source = torch.empty((num_events, 4), dtype=torch.float32)

        source[:, 0] = torch.from_numpy(en)
        source[:, 1] = torch.from_numpy(phi)
        source[:, 2] = torch.from_numpy(lon)        
        source[:, 3] = torch.from_numpy(lat)
        
        source[:, 3].mul_(-1).add_(np.pi/2)
        
        return source
    
    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        context = self._get_context(photons)
        
        return np.asarray(self._response.evaluate_effective_area(context))
    
    def _event_probability(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        source = self._get_source(events)
        context = self._get_context(photons)
        
        return np.asarray(self._response.evaluate_density(context, source))
    
    def _random_events(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> EmCDSEventDataInSCFrameInterface:
        context = self._get_context(photons)
        samples = np.asarray(self._response.sample_density(context))
        samples[:, 3].mul_(-1).add_(np.pi/2)
        
        return EmCDSEventDataInSCFrameFromArrays(
            samples[:, 0], # Energy
            samples[:, 2], # Lon
            samples[:, 3], # Lat
            samples[:, 1]  # Phi
        )

class UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction(FarFieldSpectralInstrumentResponseFunctionInterface):

    event_data_type = EmCDSEventDataInSCFrameInterface

    def __init__(self, response: FullDetectorResponse,
                 batch_size = 100000):

        # Get the differential effective area, which is still integrated on each bin at this point
        # FarFieldInstrumentResponseFunctionInterface uses cm2
        # First convert and then drop the units
        self._diff_area = response.to_dr().project('NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi').to(u.cm * u.cm, copy=False).to(None, copy = False, update = False)

        # Now fix units for the axes
        # PhotonWithDirectionAndEnergyInSCFrameInterface has energy in keV
        # EmCDSEventInSCFrameInterface has energy in keV, phi in rad
        # NuLambda and PsiChi don't have units since these are HealpixAxis. They take SkyCoords
        # Copy the axes the first time since they are shared with the response:FullDetectorResponse input
        self._diff_area.axes['Ei'] = self._diff_area.axes['Ei'].to(u.keV).to(None, copy = False, update = False)
        self._diff_area.axes['Em'] = self._diff_area.axes['Em'].to(u.keV).to(None, copy = False, update = False)
        self._diff_area.axes['Phi'] = self._diff_area.axes['Phi'].to(u.rad).to(None, copy = False, update = False)

        # Integrate to get the total effective area
        self._area = self._diff_area.project('NuLambda', 'Ei')

        # Now make it differential by dividing by the phasespace
        # EmCDSEventInSCFrameInterface energy and phi units have already been taken
        # care off. Only PsiChi remains, which is a direction in the sphere, therefore per steradians
        energy_phase_space =  self._diff_area.axes['Ei'].widths
        phi_phase_space = self._diff_area.axes['Phi'].widths
        psichi_phase_space = self._diff_area.axes['PsiChi'].pixarea().to_value(u.sr)

        self._diff_area /= self._diff_area.axes.expand_dims(energy_phase_space, 'Em')
        self._diff_area /= self._diff_area.axes.expand_dims(phi_phase_space, 'Phi')
        self._diff_area /= psichi_phase_space

        self._batch_size = batch_size

    def effective_area_cm2(self, photons: Iterable[PhotonWithDirectionAndEnergyInSCFrameInterface]) -> Iterable[float]:
        """

        """

        for photon_chunk in itertools_batched(photons, self._batch_size):

            lon, lat, energy_keV = np.asarray([[photon.direction_lon_rad_sc,
                                             photon.direction_lat_rad_sc,
                                             photon.energy_keV] for photon in photon_chunk], dtype=float).transpose()

            direction = SkyCoord(lon, lat, unit = u.rad, frame = SpacecraftFrame())

            for area_eff in self._area.interp(direction, energy_keV):
                yield area_eff

    def differential_effective_area_cm2(self, query: Iterable[Tuple[PhotonWithDirectionAndEnergyInSCFrameInterface, EmCDSEventInSCFrameInterface]]) -> Iterable[float]:
        """
        Return the differential effective area (probability density of measuring a given event given a photon times the effective area)
        """

        for query_chunk in itertools_batched(query, self._batch_size):

            # Psi is colatitude (complementary angle)
            lon_ph, lat_ph, energy_i_keV, energy_m_keV, phi_rad, psi_comp, chi  = \
                np.asarray([[photon.direction_lon_rad_sc,
                             photon.direction_lat_rad_sc,
                             photon.energy_keV,
                             event.energy_keV,
                             event.scattering_angle_rad,
                             event.scattered_lat_rad_sc,
                             event.scattered_lon_rad_sc,
                            ] for photon,event in query_chunk], dtype=float).transpose()

            direction_ph = SkyCoord(lon_ph, lat_ph, unit = u.rad, frame = SpacecraftFrame())
            psichi = SkyCoord(chi, psi_comp, unit=u.rad, frame=SpacecraftFrame())

            for diff_area in self._diff_area.interp(direction_ph, energy_i_keV, energy_m_keV, phi_rad, psichi):
                yield diff_area