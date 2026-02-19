import copy
import os
import json
from typing import Optional, Iterable, Type, Tuple, List

import numpy as np
import h5py
from astromodels import PointSource
from astropy.coordinates import CartesianRepresentation
from executing import Source

from cosipy import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces import UnbinnedThreeMLSourceResponseInterface, EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldInstrumentResponseFunctionInterface
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.util.iterables import asarray

from astropy import units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord
from astropy.time import Time


from importlib.util import find_spec

if find_spec("torch") is None:
    raise RuntimeError("Install cosipy with [ml] optional package to use this feature.")

import torch


class UnbinnedThreeMLPointSourceResponseIRFAdaptive(UnbinnedThreeMLSourceResponseInterface):
    
    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf: FarFieldInstrumentResponseFunctionInterface,
                 sc_history: SpacecraftHistory,):
        
        """
        Will fold the IRF with the point source spectrum by evaluating the IRF at Ei positions adaptively chosen based on characteristic IRF features
        Note that this assumes a smooth flux spectrum
        
        All IRF queries are cached and can be saved to / loaded from a file

        Parameters
        ----------
        data
        irf
        sc_history
        """
        
        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._irf = irf
        self._sc_ori = sc_history
        
        # Default parameters for irf energy node placement
        self._total_energy_nodes = (60, 500)
        self._peak_nodes = (18, 12)
        self._peak_widths = (0.04, 0.1)
        self._energy_range = (100., 10_000.)
        self._batch_size = 1_000_000
        
        # Placeholder for node pool - stored as Tensors
        self._width_tensor: Optional[torch.Tensor] = None
        self._nodes_primary: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._nodes_secondary: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        self._nodes_bkg_1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._nodes_bkg_2: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._nodes_bkg_3: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        
        # Checks to avoid unecessary recomputations
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None
        
        # Cached values
        self._irf_cache: Optional[torch.Tensor] = None # cm^2/rad/sr
        self._irf_energy_node_cache: Optional[np.ndarray] = None # (Optional, only if full batch)
        self._area_cache: Optional[np.ndarray] = None # cm^2*s*keV
        self._area_energy_node_cache: Optional[np.ndarray] = None
        self._exp_events: Optional[float] = None
        self._exp_density: Optional[torch.Tensor] = None
        
        # Precomputed spacecraft history
        self._mid_times = self._sc_ori.obstime[:-1] + (self._sc_ori.obstime[1:] - self._sc_ori.obstime[:-1]) / 2
        self._sc_ori_center = self._sc_ori.interp(self._mid_times)
        
        data_times = self._data.time
        self._n_events = self._data.nevents
        self._unique_mjds, self._inv_idx = np.unique(data_times.mjd, return_inverse=True)
        unique_times_obj = Time(self._unique_mjds, format='mjd')
        
        self._sc_ori_unique = self._sc_ori.interp(unique_times_obj)
        
        interval_ratios = (self._sc_ori.livetime.to_value(u.s) / self._sc_ori.intervals_duration.to_value(u.s))
        bin_indices = np.searchsorted(self._sc_ori.obstime.mjd, self._unique_mjds) - 1

        bin_indices = np.clip(bin_indices, 0, len(self._sc_ori.livetime) - 1)
        unique_ratio = interval_ratios[bin_indices]

        self._livetime_ratio = unique_ratio[self._inv_idx].astype(np.float32)
        
        self._energy_m_keV = torch.as_tensor(asarray(self._data.energy_keV, dtype=np.float32))
        self._phi_rad = torch.as_tensor(asarray(self._data.scattering_angle_rad, dtype=np.float32))
        
        self._lon_scatt = torch.as_tensor(asarray(self._data.scattered_lon_rad_sc, dtype=np.float32))
        self._lat_scatt = torch.as_tensor(asarray(self._data.scattered_lat_rad_sc, dtype=np.float32))
        self._cos_lat_scatt = torch.cos(self._lat_scatt)
        self._sin_lat_scatt = torch.sin(self._lat_scatt)
        self._cos_lon_scatt = torch.cos(self._lon_scatt)
        self._sin_lon_scatt = torch.sin(self._lon_scatt)
        
        #unique_ratio = np.interp(self._unique_mjds, 
        #                         self._mid_times.mjd, 
        #                         self._sc_ori.livetime.to_value(u.s) / self._sc_ori.intervals_duration.to_value(u.s))
        #
        #self._livetime_ratio = unique_ratio[self._inv_idx].astype(np.float32)
        
        #wrong_order = np.where(((data_times[1:] - data_times[:-1]) <= 0))[0]
        #data_times[wrong_order + 1] = data_times[wrong_order + 1] + 1
        #self._sc_ori_data = self._sc_ori.interp(data_times)
        
        #ratio = np.interp(self._data.time.mjd, 
        #                  self._mid_times.mjd, 
        #                  self._sc_ori.livetime.to_value(u.s)/self._sc_ori.intervals_duration.to_value(u.s))
        #self._livetime_ratio = ratio.astype(np.float32)
    
    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface
    
    def set_integration_parameters(self,
                                   total_energy_nodes: Tuple[int, int] = (60, 500),
                                   peak_nodes: Tuple[int, int] = (18, 12),
                                   peak_widths: Tuple[float, float] = (0.04, 0.1),
                                   energy_range: Tuple[float, float] = (100., 10_000.),
                                   batch_size: int = 1_000_000,):
        
        # Reset caches if parameters change
        if (peak_nodes != self._peak_nodes
            or
            peak_widths != self._peak_widths
            or
            total_energy_nodes[0] != self._total_energy_nodes[0]):
            self._irf_cache = None
            self._irf_energy_node_cache = None
            self._width_tensor = None
            self._nodes_primary = None
            self._nodes_secondary = None
            self._nodes_bkg_1 = None
            self._nodes_bkg_2 = None
            self._nodes_bkg_3 = None
        
        if (total_energy_nodes[1] != self._total_energy_nodes[1]):
            self._area_cache = None
            self._area_energy_node_cache = None
        
        if (energy_range != self._energy_range):
            self._irf_cache = None
            self._irf_energy_node_cache = None
            self._area_cache = None
            self._area_energy_node_cache = None
            
        if total_energy_nodes[0] < (peak_nodes[0] + 2 * peak_nodes[1] + 3):
            raise ValueError("To many nodes per peak compared to the total number or peaks!")
            
        if (total_energy_nodes[0] < 1) or (total_energy_nodes[1] < 1):
            raise ValueError("The number of energy nodes must be at least 1.")

        if energy_range[0] >= energy_range[1]:
            raise ValueError("The initial energy interval needs to be increasing!")
        
        if (batch_size < total_energy_nodes[0]) or (batch_size < total_energy_nodes[1]):
            raise ValueError("The batch size cannot be smaller than the number of integration nodes.")
            
        self._total_energy_nodes = total_energy_nodes
        self._peak_nodes = peak_nodes
        self._peak_widths = peak_widths
        self._energy_range = energy_range
        self._batch_size = batch_size
    
    @staticmethod
    def _build_nodes(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, w = np.polynomial.legendre.leggauss(degree)
        return torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), torch.as_tensor(w, dtype=torch.float32).unsqueeze(0)
    
    def _build_split_nodes(self, remaining: int, groups: int):
        q, r = divmod(remaining, groups)
        return [self._build_nodes(q + (1 if i < r else 0)) for i in range(groups)]
    
    def _init_node_pool(self):
        self._width_tensor = torch.tensor([self._peak_widths[0], self._peak_widths[0],
                                           self._peak_widths[1], self._peak_widths[1]], dtype=torch.float32)
        
        self._nodes_primary = self._build_nodes(self._peak_nodes[0])
        self._nodes_secondary = self._build_nodes(self._peak_nodes[1])

        self._nodes_bkg_1 = self._build_nodes(self._total_energy_nodes[0] - self._peak_nodes[0])

        self._nodes_bkg_2 = self._build_split_nodes(
            self._total_energy_nodes[0] - self._peak_nodes[0] - self._peak_nodes[1], 2
        )

        self._nodes_bkg_3 = self._build_split_nodes(
            self._total_energy_nodes[0] - self._peak_nodes[0] - 2 * self._peak_nodes[1], 3
        )

    @staticmethod
    def _scale_nodes_exp(E1: torch.Tensor, E2: torch.Tensor, 
                         nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = E2 - E1

        out_n = (nodes_u + 1).mul(0.5).pow(2).mul(diff).add(E1)
        out_w = (nodes_u + 1).mul(0.5).mul(weights_u).mul(diff)

        return out_n, out_w
    
    @staticmethod
    def _scale_nodes_center(E1: torch.Tensor, E2: torch.Tensor, EC: torch.Tensor,
                            nodes_u: torch.Tensor, weights_u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_left = (nodes_u < 0)
        width_left = (EC - E1)
        width_right = (E2 - EC)

        scale = torch.where(mask_left, width_left, width_right)

        out_n = nodes_u.pow(3).mul(scale).add(EC)
        out_w = nodes_u.pow(2).mul(3).mul(weights_u).mul(scale)

        return out_n, out_w
    
    def _get_escape_peak(self, energy_m_keV: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        E2 = 511.0 / (1.0 + 511.0 / energy_m_keV - torch.cos(phi_rad))
        energy = energy_m_keV + 1022.0 - E2
        
        accept = (energy < self._energy_range[1]) & (energy > self._energy_range[0]) & (energy > 1600.0) & (energy_m_keV < energy)
        return torch.where(accept, energy, torch.tensor(float('nan'), dtype=torch.float32))
    
    def _get_missing_energy_peak(self, phi_geo_rad: torch.Tensor, energy_m_keV: torch.Tensor, 
                                 phi_rad: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        cos_geo = torch.cos(phi_geo_rad)
        cos_phi = torch.cos(phi_rad)
        
        if inverse:
            denom = 2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            root = torch.sqrt(energy_m_keV * (cos_geo - 1) * (-2044.0 - 5 * energy_m_keV + energy_m_keV * cos_geo + 4 * energy_m_keV * cos_phi))
            energy = 511.0 * (energy_m_keV - energy_m_keV * cos_geo + root) / denom
        else:
            denom = 2 * (-1 + cos_geo) * (-511.0 - energy_m_keV + energy_m_keV * cos_phi)
            root = torch.sqrt(energy_m_keV**2 * (cos_geo - 1) * (cos_phi - 1) *
                              ((1022.0 + energy_m_keV)**2 - energy_m_keV * (2044.0 + energy_m_keV) * cos_phi - 2 * energy_m_keV**2 * cos_geo * torch.sin(phi_rad/2)**2))
            energy = (energy_m_keV**2 * (1 - cos_geo - cos_phi + cos_phi * cos_geo) + root) / denom

        accept = (energy < self._energy_range[1]) & (energy > self._energy_range[0]) & (energy > energy_m_keV) & (energy_m_keV/energy - 1 < -0.2)
        return torch.where(accept, energy, torch.tensor(float('nan'), dtype=torch.float32))
    
    def init_cache(self):
        self._update_cache()
    
    def clear_cache(self):
        self._irf_cache = None
        self._irf_energy_node_cache = None
        self._area_cache = None
        self._area_energy_node_cache = None
        self._exp_events = None
        self._exp_density = None
        
        self._last_convolved_source_skycoord = None
        self._last_convolved_source_dict_number = None
        self._last_convolved_source_dict_density = None
        self._sc_coord_sph_cache = None
    
    def set_source(self, source: Source):
        if not isinstance(source, PointSource):
            raise TypeError("Please provide a PointSource!")

        self._source = source
    
    def copy(self) -> UnbinnedThreeMLSourceResponseInterface:
        new_instance = copy.copy(self)
        new_instance.clear_cache()
        new_instance._source = None
        
        return new_instance
    
    @staticmethod
    def _earth_occ(source_coord: SkyCoord, ori: SpacecraftHistory) -> np.ndarray:
        gcrs_cart = ori.location.represent_as(CartesianRepresentation)
        dist_earth_center = gcrs_cart.norm()
        max_angle = np.pi*u.rad - np.arcsin(c.R_earth/dist_earth_center)
        src_angle = source_coord.separation(ori.earth_zenith)
        return (src_angle < max_angle).astype(np.float32)
    
    def _compute_area(self):
        coord = self._source.position.sky_coord
        n_energy = self._total_energy_nodes[1]

        log_E_min = np.log10(self._energy_range[0])
        log_E_max = np.log10(self._energy_range[1])

        x, w = np.polynomial.legendre.leggauss(n_energy)

        scale = 0.5 * (log_E_max - log_E_min)
        y_nodes = scale * x + 0.5 * (log_E_max + log_E_min)
        self._area_energy_node_cache = 10**y_nodes
        
        e_w = (np.log(10) * self._area_energy_node_cache * (w * scale)).astype(np.float32).reshape(1, -1)
        e_n = self._area_energy_node_cache.astype(np.float32)

        sc_coord_sph = self._sc_ori_center.get_target_in_sc_frame(coord)
        earth_occ_index = self._earth_occ(coord, self._sc_ori_center)

        time_weights = (self._sc_ori.livetime.to_value(u.s)).astype(np.float32) * earth_occ_index

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

        n_time = len(lon_ph_rad)
        batch_size_time = self._batch_size // n_energy

        total_area = np.zeros(n_energy, dtype=np.float64)
        
        max_batch_total = n_energy * min(batch_size_time, n_time)
        batch_lons_buffer = np.empty(max_batch_total, dtype=np.float32)
        batch_lats_buffer = np.empty(max_batch_total, dtype=np.float32)
        batch_energies_buffer = np.empty(max_batch_total, dtype=np.float32)

        for i in range(0, n_time, batch_size_time):
            start = i
            end = min(i + batch_size_time, n_time)
            current_n_time = end - start
            current_total = current_n_time * n_energy
            
            #np.repeat(lon_ph_rad[start:end], n_energy, out=batch_lons_buffer[:current_total])
            #np.repeat(lat_ph_rad[start:end], n_energy, out=batch_lats_buffer[:current_total])

            batch_lons_buffer[:current_total].reshape(current_n_time, n_energy)[:] = lon_ph_rad[start:end, np.newaxis]
            batch_lats_buffer[:current_total].reshape(current_n_time, n_energy)[:] = lat_ph_rad[start:end, np.newaxis]
            batch_energies_buffer[:current_total].reshape(current_n_time, n_energy)[:] = e_n
            
            photons = PhotonListWithDirectionAndEnergyInSCFrame(
                batch_lons_buffer[:current_total],
                batch_lats_buffer[:current_total],
                batch_energies_buffer[:current_total]
                )
            
            eff_areas_flat = asarray(self._irf._effective_area_cm2(photons), dtype=np.float32)
            eff_areas_grid = eff_areas_flat.reshape(current_n_time, n_energy)

            total_area += np.einsum('ij,i,j->j', 
                                    eff_areas_grid, 
                                    time_weights[start:end], 
                                    e_w.ravel())

        self._area_cache = total_area
    
    def _fill_nodes(self, nodes_out: torch.Tensor, weights_out: torch.Tensor, 
                    indices: torch.Tensor, mode: int, 
                    sorted_peaks: torch.Tensor, delta: torch.Tensor):
        
        Emin, Emax = self._energy_range
        
        if mode == 1:
            E1 = (sorted_peaks[:, 0] - delta[:, 0]).clamp(min=Emin)
            E2 = (sorted_peaks[:, 0] + delta[:, 0]).clamp(max=Emax)
            
            EC = sorted_peaks[:, 0]
            
            E1, E2, EC = [E.view(-1, 1) for E in (E1, E2, EC)]
            
            c = 0
            w = self._nodes_primary[0].shape[1]
            n_res, w_res = self._scale_nodes_center(E1, E2, EC, *self._nodes_primary)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_center_inplace(E1, E2, EC, *self._nodes_primary, 
            #                                 nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_bkg_1[0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E2, Emax, *self._nodes_bkg_1)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_exp_inplace(E2, Emax, *self._nodes_bkg_1,
            #                              nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
        
        elif mode == 2:
            center_peak = (sorted_peaks[:, 0] + sorted_peaks[:, 1]) / 2
            
            E1 = (sorted_peaks[:, 0] - delta[:, 0]).clamp(min=Emin)
            E3 = (sorted_peaks[:, 1] - delta[:, 1]).clamp(min=center_peak)
            E2 = (sorted_peaks[:, 0] + delta[:, 0]).clamp(max=E3)
            E4 = (sorted_peaks[:, 1] + delta[:, 1]).clamp(max=Emax)
            
            EC1 = sorted_peaks[:, 0]
            EC2 = sorted_peaks[:, 1]
            
            E1, E2, E3, E4, EC1, EC2 = [E.view(-1, 1) for E in (E1, E2, E3, E4, EC1, EC2)]
            
            c = 0
            w = self._nodes_primary[0].shape[1]
            n_res, w_res = self._scale_nodes_center(E1, E2, EC1, *self._nodes_primary)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_center_inplace(E1, E2, EC1, *self._nodes_primary, 
            #                                 nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_bkg_2[0][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E2, E3, *self._nodes_bkg_2[0])
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_exp_inplace(E2, E3, *self._nodes_bkg_2[0],
            #                              nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_secondary[0].shape[1]
            n_res, w_res = self._scale_nodes_center(E3, E4, EC2, *self._nodes_secondary)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_center_inplace(E3, E4, EC2, *self._nodes_secondary,
            #                                 nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_bkg_2[1][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E4, Emax, *self._nodes_bkg_2[1])
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_exp_inplace(E4, Emax, *self._nodes_bkg_2[1],
            #                              nodes_out[indices, c:c+w], weights_out[indices, c:c+w])

        elif mode == 3:
            center_peak_1 = (sorted_peaks[:, 0] + sorted_peaks[:, 1]) / 2
            center_peak_2 = (sorted_peaks[:, 1] + sorted_peaks[:, 2]) / 2
            
            E1 = (sorted_peaks[:, 0] - delta[:, 0]).clamp(min=Emin)
            E3 = (sorted_peaks[:, 1] - delta[:, 1]).clamp(min=center_peak_1)
            E2 = (sorted_peaks[:, 0] + delta[:, 0]).clamp(max=E3)
            E4 = (sorted_peaks[:, 1] + delta[:, 1]).clamp(max=center_peak_2)
            E5 = (sorted_peaks[:, 2] - delta[:, 2]).clamp(min=E4)
            E6 = (sorted_peaks[:, 2] + delta[:, 2]).clamp(max=Emax)
            
            EC1, EC2, EC3 = [sorted_peaks[:, i] for i in range(3)]
            
            E1, E2, E3, E4, E5, E6, EC1, EC2, EC3 = [E.view(-1, 1) for E in (E1, E2, E3, E4, E5, E6, EC1, EC2, EC3)]
            
            c = 0
            w = self._nodes_primary[0].shape[1]
            n_res, w_res = self._scale_nodes_center(E1, E2, EC1, *self._nodes_primary)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_center_inplace(E1, E2, EC1, *self._nodes_primary,
            #                                 nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_bkg_3[0][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E2, E3, *self._nodes_bkg_3[0])
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_exp_inplace(E2, E3, *self._nodes_bkg_3[0],
            #                              nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_secondary[0].shape[1]
            n_res, w_res = self._scale_nodes_center(E3, E4, EC2, *self._nodes_secondary)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_center_inplace(E3, E4, EC2, *self._nodes_secondary,
            #                                 nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_bkg_3[1][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E4, E5, *self._nodes_bkg_3[1])
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_exp_inplace(E4, E5, *self._nodes_bkg_3[1],
            #                              nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_secondary[0].shape[1]
            n_res, w_res = self._scale_nodes_center(E5, E6, EC3, *self._nodes_secondary)
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_center_inplace(E5, E6, EC3, *self._nodes_secondary,
            #                                 nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
            c += w
            w = self._nodes_bkg_3[2][0].shape[1]
            n_res, w_res = self._scale_nodes_exp(E6, Emax, *self._nodes_bkg_3[2])
            nodes_out[indices, c:c+w] = n_res
            weights_out[indices, c:c+w] = w_res
            #self._scale_nodes_exp_inplace(E6, Emax, *self._nodes_bkg_3[2],
            #                              nodes_out[indices, c:c+w], weights_out[indices, c:c+w])
    
    def _get_nodes(self, energy_m_keV: torch.Tensor, phi_rad: torch.Tensor, 
                   phi_geo_rad: torch.Tensor, phi_igeo_rad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        energy_m_keV = energy_m_keV.view(-1, 1)
        phi_rad = phi_rad.view(-1, 1)
        phi_geo_rad = phi_geo_rad.view(-1, 1)
        phi_igeo_rad = phi_igeo_rad.view(-1, 1)
        
        batch_size = energy_m_keV.shape[0]
        
        nodes = torch.zeros((batch_size, self._total_energy_nodes[0]), dtype=torch.float32)
        weights = torch.zeros_like(nodes)
        
        peaks = torch.zeros((batch_size, 4), dtype=torch.float32)
        peaks[:, 0] = energy_m_keV.squeeze()
        peaks[:, 1] = self._get_escape_peak(energy_m_keV, phi_rad).squeeze()
        peaks[:, 2] = self._get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad).squeeze()
        peaks[:, 3] = self._get_missing_energy_peak(phi_igeo_rad, energy_m_keV, phi_rad, inverse=True).squeeze()
        
        diffs = peaks * self._width_tensor[None, ...]
        
        n_peaks = torch.sum(~torch.isnan(peaks), dim=1)
        
        indices_1 = torch.where(n_peaks == 1)[0]
        indices_2 = torch.where(n_peaks == 2)[0]
        indices_3 = torch.where(n_peaks == 3)[0]
        
        if len(indices_1) > 0:
            self._fill_nodes(nodes, weights, indices_1, 1, 
                                      peaks[indices_1, :1], diffs[indices_1, :1])
            
        if len(indices_2) > 0:
            p_sub = peaks[indices_2]
            d_sub = diffs[indices_2]
            mask = ~torch.isnan(p_sub)
            p_comp = p_sub[mask].view(-1, 2)
            d_comp = d_sub[mask].view(-1, 2)
            self._fill_nodes(nodes, weights, indices_2, 2, p_comp, d_comp)
            
        if len(indices_3) > 0:
            p_sub = peaks[indices_3]
            d_sub = diffs[indices_3]
            mask = ~torch.isnan(p_sub)
            p_comp = p_sub[mask].view(-1, 3)
            d_comp = d_sub[mask].view(-1, 3)
            
            p_sorted, idx = torch.sort(p_comp, dim=1)
            d_sorted = torch.gather(d_comp, 1, idx)
            
            self._fill_nodes(nodes, weights, indices_3, 3, p_sorted, d_sorted)
        
        return nodes, weights
    
    def _get_CDS_coordinates(self, lon_src_rad: torch.Tensor, lat_src_rad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_lat_src = torch.cos(lat_src_rad)
        sin_lat_src = torch.sin(lat_src_rad)
        cos_lon_src = torch.cos(lon_src_rad)
        sin_lon_src = torch.sin(lon_src_rad)
        
        cos_geo = (
            cos_lat_src * cos_lon_src * self._cos_lat_scatt * self._cos_lon_scatt +
            cos_lat_src * sin_lon_src * self._cos_lat_scatt * self._sin_lon_scatt +
            sin_lat_src * self._sin_lat_scatt
        )
        
        cos_geo = torch.clip(cos_geo, -1.0, 1.0)
        phi_geo_rad = torch.arccos(cos_geo)
        
        return phi_geo_rad, np.pi - phi_geo_rad
    
    def _compute_density(self):
        coord = self._source.position.sky_coord
        sc_coord_sph = self._sc_coord_sph_cache
        earth_occ_index = self._earth_occ(coord, self._sc_ori_unique)[self._inv_idx]

        lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
        lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)
        
        phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad))

        n_energy = self._total_energy_nodes[0]
        batch_size_events = self._batch_size // n_energy

        self._irf_cache = torch.zeros((self._n_events, n_energy), dtype=torch.float32)
        
        buffer_size = n_energy * min(batch_size_events, self._n_events)
        batch_lon_src_buffer   = np.empty(buffer_size, dtype=np.float32)
        batch_lat_src_buffer   = np.empty(buffer_size, dtype=np.float32)
        batch_energy_buffer    = np.empty(buffer_size, dtype=np.float32)
        batch_phi_buffer       = np.empty(buffer_size, dtype=np.float32)
        batch_lon_scatt_buffer = np.empty(buffer_size, dtype=np.float32)
        batch_lat_scatt_buffer = np.empty(buffer_size, dtype=np.float32)

        for i in range(0, self._n_events, batch_size_events):
            start = i
            end = min(i + batch_size_events, self._n_events)
            current_n = end - start
            current_total = current_n * n_energy

            e_sl = self._energy_m_keV[start:end]
            p_sl = self._phi_rad[start:end]
            pg_sl = phi_geo_rad[start:end]
            pig_sl = phi_igeo_rad[start:end]

            nodes, weights = self._get_nodes(e_sl, p_sl, pg_sl, pig_sl)

            if batch_size_events >= self._n_events:
                 self._irf_energy_node_cache = np.asarray(nodes)

            #np.repeat(lon_ph_rad[start:end], n_energy, out=batch_lon_src_buffer[:current_total])
            #np.repeat(lat_ph_rad[start:end], n_energy, out=batch_lat_src_buffer[:current_total])
            
            batch_lon_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lon_ph_rad[start:end, np.newaxis]
            batch_lat_src_buffer[:current_total].reshape(current_n, n_energy)[:] = lat_ph_rad[start:end, np.newaxis]
            
            batch_energy_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._energy_m_keV[start:end, np.newaxis])
            batch_lon_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lon_scatt[start:end, np.newaxis])
            batch_lat_scatt_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._lat_scatt[start:end, np.newaxis])
            batch_phi_buffer[:current_total].reshape(current_n, n_energy)[:] = np.asarray(self._phi_rad[start:end, np.newaxis])

            #np.repeat(np.asarray(self._energy_m_keV[start:end]), n_energy, out=batch_energy_buffer[:current_total])
            #np.repeat(np.asarray(self._lon_scatt[start:end]), n_energy, out=batch_lon_scatt_buffer[:current_total])
            #np.repeat(np.asarray(self._lat_scatt[start:end]), n_energy, out=batch_lat_scatt_buffer[:current_total])
            #np.repeat(np.asarray(self._phi_rad[start:end]), n_energy, out=batch_phi_buffer[:current_total])

            photons = PhotonListWithDirectionAndEnergyInSCFrame(
                batch_lon_src_buffer[:current_total],
                batch_lat_src_buffer[:current_total],
                np.asarray(nodes).ravel()
                )
            events = EmCDSEventDataInSCFrameFromArrays(
                batch_energy_buffer[:current_total],
                batch_lon_scatt_buffer[:current_total],
                batch_lat_scatt_buffer[:current_total],
                batch_phi_buffer[:current_total],
            )
            
            eff_areas_flat = torch.as_tensor(asarray(self._irf._effective_area_cm2(photons), dtype=np.float32))
            densities_flat = torch.as_tensor(asarray(self._irf._event_probability(photons, events), dtype=np.float32))

            res_block = (densities_flat * eff_areas_flat).view(current_n, n_energy)

            occ = torch.as_tensor(earth_occ_index[start:end]).view(-1, 1)
            live = torch.as_tensor(self._livetime_ratio[start:end]).view(-1, 1)
            
            res_block *= occ * live * weights

            self._irf_cache[start:end] = res_block
    
    def _update_cache(self):
        
        if self._source is None:
            raise RuntimeError("Call set_source() first.")
        
        source_coord = self._source.position.sky_coord
        
        if (self._sc_coord_sph_cache is None) or (source_coord != self._last_convolved_source_skycoord):
            #self._sc_coord_sph_cache = self._sc_ori_data.get_target_in_sc_frame(source_coord)
            self._sc_coord_sph_cache = self._sc_ori_unique.get_target_in_sc_frame(source_coord)[self._inv_idx]
        
        no_recalculation = ((source_coord == self._last_convolved_source_skycoord)
                            and
                            (self._irf_cache is not None)
                            and
                            (self._area_cache is not None))
        
        area_recalculation = ((source_coord != self._last_convolved_source_skycoord)
                              or
                              (self._area_cache is None))
        
        pdf_recalculation = ((source_coord != self._last_convolved_source_skycoord)
                             or
                             (self._irf_cache is None))
        
        if no_recalculation:
            return
        else:
            if area_recalculation:
                self._compute_area()
                
            if pdf_recalculation:
                self._init_node_pool()
                self._compute_density()
            
            self._last_convolved_source_skycoord = source_coord.copy()
    
    def cache_to_file(self, filename: str):
        with h5py.File(filename, 'w') as f:
            f.attrs['total_energy_nodes'] = self._total_energy_nodes
            f.attrs['peak_nodes'] = self._peak_nodes
            f.attrs['peak_widths'] = self._peak_widths
            f.attrs['energy_range'] = self._energy_range
            f.attrs['batch_size'] = self._batch_size
            
            if self._irf_cache is not None:
                f.create_dataset('irf_cache', data=self._irf_cache.numpy(), 
                               compression='gzip', compression_opts=4)
            
            if self._irf_energy_node_cache is not None:
                f.create_dataset('irf_energy_node_cache', data=self._irf_energy_node_cache,
                               compression='gzip')

            if self._area_cache is not None:
                f.create_dataset('area_cache', data=self._area_cache,
                               compression='gzip')
                
            if self._area_energy_node_cache is not None:
                f.create_dataset('area_energy_node_cache', data=self._area_energy_node_cache,
                               compression='gzip')

            if self._exp_events is not None:
                f.create_dataset('exp_events', data=self._exp_events)

            if self._exp_density is not None:
                f.create_dataset('exp_density', data=self._exp_density.numpy(),
                               compression='gzip')

            if self._last_convolved_source_dict_number is not None:
                json_str = json.dumps(self._last_convolved_source_dict_number)
                f.attrs['last_convolved_source_dict_number'] = json_str
            
            if self._last_convolved_source_dict_density is not None:
                json_str = json.dumps(self._last_convolved_source_dict_density)
                f.attrs['last_convolved_source_dict_density'] = json_str

            if self._last_convolved_source_skycoord is not None:
                sc = self._last_convolved_source_skycoord
                f.attrs['last_convolved_lon_deg'] = sc.spherical.lon.deg
                f.attrs['last_convolved_lat_deg'] = sc.spherical.lat.deg
                f.attrs['last_convolved_frame'] = sc.frame.name
    
    def cache_from_file(self, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Cache file {filename} not found.")

        with h5py.File(filename, 'r') as f:
            self._total_energy_nodes = tuple(f.attrs['total_energy_nodes'])
            self._peak_nodes = tuple(f.attrs['peak_nodes'])
            self._peak_widths = tuple(f.attrs['peak_widths'])
            self._energy_range = tuple(f.attrs['energy_range'])
            self._batch_size = int(f.attrs['batch_size'])
            
            if 'irf_cache' in f:
                self._irf_cache = torch.from_numpy(f['irf_cache'][:])
            else:
                self._irf_cache = None

            if 'irf_energy_node_cache' in f:
                self._irf_energy_node_cache = f['irf_energy_node_cache'][:]
            else:
                self._irf_energy_node_cache = None

            if 'area_cache' in f:
                self._area_cache = f['area_cache'][:]
            else:
                self._area_cache = None

            if 'area_energy_node_cache' in f:
                self._area_energy_node_cache = f['area_energy_node_cache'][:]
            else:
                self._area_energy_node_cache = None

            if 'exp_events' in f:
                self._exp_events = float(f['exp_events'][()])
            else:
                self._exp_events = None

            if 'exp_density' in f:
                self._exp_density = torch.from_numpy(f['exp_density'][:])
            else:
                self._exp_density = None

            if 'last_convolved_source_dict_number' in f.attrs:
                self._last_convolved_source_dict_number = json.loads(f.attrs['last_convolved_source_dict_number'])
            else:
                self._last_convolved_source_dict_number = None
            
            if 'last_convolved_source_dict_density' in f.attrs:
                self._last_convolved_source_dict_density = json.loads(f.attrs['last_convolved_source_dict_density'])
            else:
                self._last_convolved_source_dict_density = None

            if 'last_convolved_lon_deg' in f.attrs:
                lon = f.attrs['last_convolved_lon_deg']
                lat = f.attrs['last_convolved_lat_deg']
                frame = f.attrs['last_convolved_frame']
                self._last_convolved_source_skycoord = SkyCoord(lon, lat, unit='deg', frame=frame)
            else:
                self._last_convolved_source_skycoord = None
            
            if self._irf_cache is not None:
                self._init_node_pool()
    
    def expected_counts(self) -> float:
        """
        Return the total expected counts.
        """
        self._update_cache()
        source_dict = self._source.to_dict()
        
        if (source_dict != self._last_convolved_source_dict_number) or (self._exp_events is None):
            area = self._area_cache
            flux = self._source(self._area_energy_node_cache)
            self._exp_events = np.sum(area * flux, dtype=float)
            
        self._last_convolved_source_dict_number = source_dict
        return self._exp_events
    
    def expectation_density(self) -> Iterable[float]:
        """
        Return the expected number of counts density. This equals the event probabiliy times the number of events.
        """
        
        self._update_cache()
        source_dict = self._source.to_dict()
        
        if (source_dict != self._last_convolved_source_dict_density) or (self._exp_density is None):
            self._exp_density = torch.zeros(self._n_events, dtype=self._irf_cache.dtype)

            if self._irf_energy_node_cache is not None:
                flux = torch.as_tensor(self._source(self._irf_energy_node_cache), dtype=self._irf_cache.dtype)

                torch.linalg.vecdot(self._irf_cache, flux, dim=1, out=self._exp_density)

            else:
                n_energy = self._total_energy_nodes[0]
                batch_size = self._batch_size // n_energy
                
                sc_coord_sph = self._sc_coord_sph_cache

                lon_ph_rad = asarray(sc_coord_sph.lon.rad, dtype=np.float32)
                lat_ph_rad = asarray(sc_coord_sph.lat.rad, dtype=np.float32)

                phi_geo_rad, phi_igeo_rad = self._get_CDS_coordinates(torch.as_tensor(lon_ph_rad), torch.as_tensor(lat_ph_rad))

                for i in range(0, self._n_events, batch_size):
                    end = min(i + batch_size, self._n_events)

                    e_sl = self._energy_m_keV[i:end]
                    p_sl = self._phi_rad[i:end]
                    pg_sl = phi_geo_rad[i:end]
                    pig_sl = phi_igeo_rad[i:end]

                    nodes, _ = self._get_nodes(e_sl, p_sl, pg_sl, pig_sl)

                    flux_batch = torch.as_tensor(self._source(np.asarray(nodes)), dtype=self._irf_cache.dtype)
                    
                    torch.linalg.vecdot(self._irf_cache[i:end], flux_batch, dim=1, out=self._exp_density[i:end])
            
        self._last_convolved_source_dict_density = source_dict
        
        #print(self._data.time.unix[self._exp_density <= 0][:100])
        #print(np.sum(self._exp_density <= 0)/self._n_events * 100)
        #print(self.expected_counts() - np.sum(np.log(self._exp_density+1e-12)))
        return np.asarray(self._exp_density, dtype=np.float64)+1e-12