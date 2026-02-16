from pathlib import Path

import numpy as np

import astropy.units as u
import astropy.constants as c

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, GCRS, SphericalRepresentation, CartesianRepresentation, \
    UnitSphericalRepresentation
from astropy.units import Quantity
from mhealpy import HealpixBase
from histpy import Histogram, TimeAxis, HealpixAxis, Axis
from mhealpy import HealpixMap

from scoords import Attitude, SpacecraftFrame

import pandas as pd

from .scatt_map import SpacecraftAttitudeMap
from cosipy.event_selection import GoodTimeInterval

from typing import Union, Optional

import logging
logger = logging.getLogger(__name__)

__all__ = ["SpacecraftHistory"]

class SpacecraftHistory:

    def __init__(self,
                 obstime: Time,
                 attitude: Attitude,
                 location: GCRS,
                 livetime: u.Quantity = None):
        """
        Handles the spacecraft orientation. Calculates the dwell time
        map and point source response over a certain orientation
        period.

        Parameters
        ----------
        obstime:
            The obstime stamps for each pointings. Note this is NOT the obstime duration, see "livetime".
        attitude:
            Spacecraft orientation with respect to an inertial system.
        location:
            Location of the spacecraft at each timestamp in Earth-centered inertial (ECI) coordinates.
        livetime:
            Time the instrument was live for the corresponding
            obstime bin. Should have one less element than the number of
            timestamps. If not provided, it will assume that the instrument
            was fully on without interrruptions.
        """

        time_axis = TimeAxis(obstime, copy = False, label= 'obstime')

        if livetime is None:
            livetime = time_axis.widths.to(u.s)

        self._hist = Histogram(time_axis, livetime, copy_contents = False)

        if not (location.shape == () or location.shape == obstime.shape):
            raise ValueError(f"'location' must be a scalar or have the same length as the timestamps ({obstime.shape}), but it has shape ({location.shape})")

        if not (attitude.shape == () or attitude.shape == obstime.shape):
            raise ValueError(f"'attitude' must be a scalar or have the same length as the timestamps ({obstime.shape}), but it has shape ({attitude.shape})")

        self._attitude = attitude

        self._gcrs = location

    @property
    def nintervals(self):
        return self._hist.nbins

    @property
    def intervals_duration(self):
        return self._hist.axis.widths.to(self._hist.unit)

    @property
    def intervals_tstart(self):
        return self._hist.axis.lower_bounds

    @property
    def intervals_tstop(self):
        return self._hist.axis.upper_bounds

    @property
    def tstart(self):
        return self._hist.axis.lo_lim

    @property
    def tstop(self):
        return self._hist.axis.hi_lim

    @property
    def npoints(self):
        return self._hist.nbins + 1

    @property
    def obstime(self):
        return self._hist.axis.edges

    @property
    def livetime(self):
        return self._hist.contents

    @property
    def attitude(self):
        return self._attitude

    @property
    def location(self)->GCRS:
        return self._gcrs

    @property
    def earth_zenith(self) -> SkyCoord:
        """
        Pointing of the Earth's zenith at the location of the SC
        """
        gcrs_sph = self._gcrs.represent_as(SphericalRepresentation)
        return SkyCoord(ra=gcrs_sph.lon, dec=gcrs_sph.lat, frame='icrs', copy=False)

    @classmethod
    def open(cls, file, tstart:Time = None, tstop:Time = None) -> "SpacecraftHistory":

        """
        Parses timestamps, axis positions from file and returns to __init__.

        Parameters
        ----------
        file : str
            The file path of the pointings.
        tstart:
            Start reading the file from an interval *including* this time. Use select_interval() to
            cut the SC file at exactly this tiem.
        tstop:
            Stop reading the file at an interval *including* this time. Use select_interval() to
            cut the SC file at exactly this tiem.

        Returns
        -------
        cosipy.spacecraftfile.spacecraft_file
            The SpacecraftHistory object.
        """

        file = Path(file)

        if file.suffix == ".ori":
            return cls._parse_from_file(file, tstart, tstop)
        else:
            raise ValueError(f"File format for {file} not supported")

    @classmethod
    def _parse_from_file(cls, file, tstart:Time = None, tstop:Time = None) -> "SpacecraftHistory":
        """
        Parses an .ori txt file with MEGAlib formatting.

        # Columns
        # 0: Always "OG" (orbital geometry)
        # 1: obstime: timestamp in unix seconds
        # 2: lat_x: galactic latitude of SC x-axis (deg)
        # 3: lon_x: galactic longitude of SC x-axis (deg)
        # 4: lat_z galactic latitude of SC z-axis (deg)
        # 5: lon_z: galactic longitude of SC y-axis (deg)
        # 6: altitude: altitude above from Earth's ellipsoid (km)
        # 7: Earth_lat: galactic latitude of the direction the Earth's zenith is pointing to at the SC location (deg)
        # 8: Earth_lon: galactic longitude of the direction the Earth's zenith is pointing to at the SC location (deg)
        # 9: livetime (previously called SAA): accumulated uptime up to the following entry (seconds)

        Parameters
        ----------
        file:
            Path to .ori file

        Returns
        -------
        cosipy.spacecraftfile.spacecraft_file
            The SpacecraftHistory object.
        """

        # First and last line are read only by MEGAlib e.g.
        # Type OrientationsGalactic
        # ...
        # EN
        # Using [:-1] instead of skipfooter=1 because otherwise it's slow and you get
        # ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support skipfooter; you can avoid this warning by specifying engine='python'.

        time, lat_x,lon_x,lat_z,lon_z,altitude,earth_lat,earth_lon,livetime = pd.read_csv(file, sep="\s+", skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), header = None, comment = '#', ).values[:-1].transpose()

        time = Time(time, format="unix")

        if tstart is not None or tstop is not None:
            # Cut early to skip some conversions later on

            start_idx = 0
            stop_idx = time.size

            time_axis = TimeAxis(time, copy=False)

            if tstart is not None:
                start_idx = time_axis.find_bin(tstart)

            if tstop is not None:
                stop_idx = time_axis.find_bin(tstop) + 2

            time = time[start_idx:stop_idx]
            lat_x = lat_x[start_idx:stop_idx]
            lon_x = lon_x[start_idx:stop_idx]
            lat_z = lat_z[start_idx:stop_idx]
            lon_z = lon_z[start_idx:stop_idx]
            altitude = altitude[start_idx:stop_idx]
            earth_lat = earth_lat[start_idx:stop_idx]
            earth_lon = earth_lon[start_idx:stop_idx]
            livetime = livetime[start_idx:stop_idx]

        xpointings = SkyCoord(l=lon_x * u.deg, b=lat_x * u.deg, frame="galactic")
        zpointings = SkyCoord(l=lon_z * u.deg, b=lat_z * u.deg, frame="galactic")

        attitude = Attitude.from_axes(x=xpointings, z=zpointings, frame = 'galactic')

        livetime = livetime[:-1]*u.s # The last element is 0.

        # Currently, the orbit information is in a weird format.
        # The altitude is specified with respect to the Earth's surface, like
        # you would specify it in a geodetic format, while
        # the lon/lat is specified in J2000, like you would in ECI.
        # Eventually everything should be in ECI (GCRS in astropy
        # for all practical purposes), but for now let's do the conversion.
        # 1. Get the direction in galactic
        # 2. Transform to GCRS, which uses RA/Dec (ICRS-like).
        #    This is represented in the unit sphere
        # 3. Add the altitude by transforming to EarthLocation.
        #    Should take care of the non-spherical Earth
        # 4. Go back GCRS, now with the correct distance
        #    (from the Earth's center)
        zenith_gal = SkyCoord(l=earth_lon * u.deg, b=earth_lat * u.deg, frame="galactic", copy = False)
        gcrs = zenith_gal.transform_to('gcrs')
        earth_loc = EarthLocation.from_geodetic(lon=gcrs.ra, lat=gcrs.dec, height=altitude*u.km)
        gcrs2 = GCRS(ra=gcrs.ra, dec=gcrs.dec, distance=earth_loc.itrs.cartesian.norm(), copy=False)

        return cls(time, attitude, gcrs2, livetime)

    @staticmethod
    def _interp_location(t, d1, d2):
        """
        Compute a direction that linearly interpolates between
        directions d1 and d2 using SLERP.

        The two directions are assumed to have the same frame,
        which is also used for the interpolated result.

        Parameters
        ----------
        t : float in [0, 1]
          interpolation fraction
        d1 : GCRS
          1st direction
        d2 : GCRS
          2nd direction

        Returns
        -------
        SkyCoord: interpolated direction

        """

        if np.all(d1 == d2):
            return d1

        v1 = d1.cartesian.xyz.value
        v2 = d2.cartesian.xyz.value
        unit = d1.cartesian.xyz.unit

        # angle between v1, v2
        theta = np.arccos(np.einsum('i...,i...->...',v1, v2)/d1.spherical.distance.value/d2.spherical.distance.value)

        # SLERP interpolated vector
        den = np.sin(theta)
        vi = (np.sin((1 - t) * theta) * v1 + np.sin(t * theta) * v2) / den

        dvi = GCRS(*Quantity(vi, unit = unit, copy = False),  representation_type='cartesian')

        return dvi

    @staticmethod
    def _interp_attitude(t, att1, att2):
        """
        Compute an Attitude that linearly interpolates between
        att1 and att2 using SLERP on their quaternion
        representations.

        The two Attitudes are assumed to have the same frame,
        which is also used for the interpolated result.

        Parameters
        ----------
        t : float in [0, 1]
          interpolation fraction
        att1 : Attitude
        att2 : Attitude

        Returns
        -------
        Attitude : interpolated attitude

        """

        if att1 == att2:
            return att1

        p1 = att1.as_quat()
        p2 = att2.as_quat()

        # angle between quaternions p1, p2 (xyzw order)
        theta = 2 * np.arccos(np.einsum('i...,i...->...',p1.transpose(), p2.transpose()))

        # Makes it work with scalars or any input shape
        t = t[..., np.newaxis]
        theta = theta[..., np.newaxis]

        # SLERP interpolated quaternion
        den = np.sin(theta)
        pi = (np.sin((1 - t) * theta) * p1 + np.sin(t * theta) * p2) / den

        return Attitude.from_quat(pi, frame=att1.frame)

    def interp_attitude(self, time) -> Attitude:
        """

        Returns
        -------

        """

        points, weights = self.interp_weights(time)

        return self.__class__._interp_attitude(weights[1], self._attitude[points[0]], self._attitude[points[1]])

    def interp_location(self, time) -> GCRS:
        """

        Returns
        -------
        """

        points, weights = self.interp_weights(time)

        return self.__class__._interp_location(weights[1], self._gcrs[points[0]], self._gcrs[points[1]])

    def _cumulative_livetime(self, points, weights) -> u.Quantity:

        cum_livetime_discrete = np.append(0 * self._hist.unit, np.cumsum(self.livetime))

        up_to_tstart = cum_livetime_discrete[points[0]]

        within_bin = self.livetime[points[0]] * weights[1]

        cum_livetime = up_to_tstart + within_bin

        return cum_livetime

    def cumulative_livetime(self, time: Optional[Time] = None) -> u.Quantity:
        """
        Get the cumulative live obstime up to this obstime.

        The live obstime in between the internal timestamp is
        assumed constant.

        All by edfault

        Parameters
        ----------
        time:
            Timestamps

        Returns
        -------
        Cummulative live obstime, with units.
        """

        if time is None:
            # All
            return np.sum(self.livetime)

        points, weights = self.interp_weights(time)

        return self._cumulative_livetime(points, weights)

    def interp_weights(self, times: Time):
        return self._hist.axis.interp_weights_edges(times)

    def interp(self, times: Time) -> 'SpacecraftHistory':

        """
        Linearly interpolates attitude and position at a given obstime

        Parameters
        ----------
        times:
            Timestamps to interpolate

        Returns
        -------
        A new SpacecraftHistory object interpolated at these location
        """

        if times.size < 2:
            raise ValueError("We need at least two obstime stamps. See also interp_attitude and inter_location")

        points, weights = self.interp_weights(times)

        interp_attitude = self._interp_attitude(weights[1], self._attitude[points[0]], self._attitude[points[1]])
        interp_location = self._interp_location(weights[1], self._gcrs[points[0]], self._gcrs[points[1]])

        cum_livetime = self._cumulative_livetime(points, weights)
        diff_livetime = cum_livetime[1:] - cum_livetime[:-1]

        return self.__class__(times, interp_attitude, interp_location, diff_livetime)

    def select_interval(self, start:Time = None, stop:Time = None) -> "SpacecraftHistory":
        """
        Returns the SpacecraftHistory file class object for the source interval.

        Parameters
        ----------
        start : astropy.time.Time
            The start obstime of the orientation period. Start of history by default.
        stop : astropy.time.Time
            The end obstime of the orientation period. End of history by default.

        Returns
        -------
        cosipy.spacecraft.SpacecraftHistory
        """

        if start is None:
            start = self.tstart

        if stop is None:
            stop = self.tstop

        if start < self.tstart or stop > self.tstop:
            raise ValueError(f"Input range ({start}-{stop}) is outside the SC history ({self.tstart}-{self.tstop})")

        start_points, start_weights = self.interp_weights(start)
        stop_points, stop_weights = self.interp_weights(stop)

        # Center values
        new_obstime = self.obstime[start_points[1]:stop_points[1]]
        new_attitude = self._attitude.as_matrix()[start_points[1]:stop_points[1]]
        new_location = self._gcrs[start_points[1]:stop_points[1]].cartesian.xyz
        new_livetime = self.livetime[start_points[1]:stop_points[0]]

        # Left edge
        # new_obstime.size can be zero if the requested interval fell completely
        # an existing interval
        if new_obstime.size == 0 or new_obstime[0] != start:
            # Left edge might be included already

            new_obstime = Time(np.append(start.jd1, new_obstime.jd1),
                               np.append(start.jd2, new_obstime.jd2),
                               format = 'jd')

            start_attitude = self._interp_attitude(start_weights[1], self._attitude[start_points[0]], self._attitude[start_points[1]])
            new_attitude = np.append(start_attitude.as_matrix()[None], new_attitude, axis=0)

            start_location = self._interp_location(start_weights[1], self._gcrs[start_points[0]], self._gcrs[start_points[1]])[None].cartesian.xyz
            new_location = np.append(start_location, new_location, axis = 1)

            first_livetime = self.livetime[start_points[0]] * start_weights[0]
            new_livetime = np.append(first_livetime, new_livetime)

        # Right edge
        # It's never included, since stop <= self.obstime[stop_points[1]], and the
        # selection above excludes stop_points[1]
        new_obstime = Time(np.append(new_obstime.jd1, stop.jd1),
                           np.append(new_obstime.jd2, stop.jd2),
                           format='jd')

        stop_attitude = self._interp_attitude(stop_weights[1], self._attitude[stop_points[0]], self._attitude[stop_points[1]])
        new_attitude = np.append(new_attitude, stop_attitude.as_matrix()[None], axis=0)
        new_attitude = Attitude.from_matrix(new_attitude, frame=self._attitude.frame)

        stop_location = self._interp_location(stop_weights[1], self._gcrs[stop_points[0]], self._gcrs[stop_points[1]])[None].cartesian.xyz
        new_location = np.append(new_location, stop_location, axis=1)

        new_location = GCRS(x = new_location[0], y = new_location[1], z = new_location[2],
                            representation_type='cartesian')

        if np.all(start_points == stop_points):
            # This can only happen if the requested interval fell completely
            # an existing interval
            new_livetime[0] -= self.livetime[stop_points[0]]*stop_weights[0]
        else:
            last_livetime = self.livetime[stop_points[0]]*stop_weights[1]
            new_livetime = np.append(new_livetime, last_livetime)

        # We used the internal jd1 and jd2 values, which might have changed the format.
        # Bring it back
        new_obstime.format = self.obstime.format

        return self.__class__(new_obstime, new_attitude, new_location, new_livetime)

    def apply_gti(self, gti: GoodTimeInterval) -> "SpacecraftHistory":
        """
        Returns the SpacecraftHistory file class object by masking livetimes outside the good time interval.

        Parameters
        ----------
        gti: cosipy.event_selection.GoodTimeInterval

        Returns
        -------
        cosipy.spacecraft.SpacecraftHistory
        """
        new_obstime = None
        new_attitude = None
        new_location = None
        new_livetime = None

        for i, (start, stop) in enumerate(zip(gti.tstart_list, gti.tstop_list)):
        # TODO: this line can be replaced with the following line after the PR in the develop branch is merged.
        #for i, (start, stop) in enumerate(gti):
            _sph = self.select_interval(start, stop)

            _obstime = _sph.obstime
            _attitude = _sph._attitude.as_matrix()
            _location = _sph._gcrs.cartesian.xyz
            _livetime = _sph.livetime

            if i == 0:
                new_obstime = _obstime
                new_attitude = _attitude
                new_location = _location
                new_livetime = _livetime
            else:
                new_obstime = Time(np.append(new_obstime.jd1, _obstime.jd1),
                                   np.append(new_obstime.jd2, _obstime.jd2),
                                   format='jd')
                new_attitude = np.append(new_attitude, _attitude, axis = 0)
                new_location = np.append(new_location, _location, axis = 1)
                new_livetime = np.append(new_livetime, 0 * new_livetime.unit) # assign livetime of zero between GTIs
                new_livetime = np.append(new_livetime, _livetime)

        # finalizing
        new_attitude = Attitude.from_matrix(new_attitude, frame=self._attitude.frame)
        new_location = GCRS(x = new_location[0], y = new_location[1], z = new_location[2],
                            representation_type='cartesian')
        new_obstime.format = self.obstime.format

        return self.__class__(new_obstime, new_attitude, new_location, new_livetime)

    @staticmethod
    def _cart_to_polar(v):
        """
        Convert Cartesian 3D unit direction vectors to polar coordinates.

        Parameters
        ----------
        v : np.ndarray(float) [N x 3]
          array of N 3D unit vectors

        Returns
        -------
        lon, colat : np.ndarray(float) [N]
          longitude and co-latitude corresponding to v in radians

        """

        lon   = np.arctan2(v[:,1], v[:,0])
        colat = np.arccos(v[:,2])
        return (lon, colat)

    def get_target_in_sc_frame(self, target_coord):

        """
        Convert a target coordinate in an inertial frame to the path of
        the source in the spacecraft frame.  The target coordinate may
        be provided either as a SkyCoord or as a Cartesian 3-vector,
        which determines the type of the output.

        Parameters
        ----------
        target_coord : astropy.coordinates.SkyCoord or Cartesian 3-vector
            The coordinates of the target object.
        Returns
        -------
        astropy.coordinates.SkyCoord or pair of np.ndarrays
            The target coordinates in the spacecraft frame.  If input
            was a SkyCoord, output is a vector SkyCoord; otherwise, it
            is a pair (longitude, co-latitude) in radians.

        """

        useSkyCoord = isinstance(target_coord, SkyCoord)

        if useSkyCoord:
            target_coord = target_coord.transform_to(self._attitude.frame)
            target_coord = target_coord.cartesian.xyz.value

        src_path_cartesian = np.dot(self._attitude.rot.inv().as_matrix(),
                                    target_coord)

        # convert to spherical lon, colat in radians
        lon, colat = self._cart_to_polar(src_path_cartesian)

        if useSkyCoord:
            # SpacecraftFrame takes lon, lat arguments
            src_path_skycoord = SkyCoord(lon=lon, lat=np.pi / 2 - colat,
                                         unit=u.rad,
                                         frame=SpacecraftFrame())

            return src_path_skycoord
        else:
            # return raw longitude and co-latitude in radians
            return lon, colat

    def get_dwell_map(self, target_coord:SkyCoord, nside:int = None, scheme = 'ring', base:HealpixBase = None) -> HealpixMap:

        """
        Generates the dwell obstime map for the source.

        Parameters
        ----------
        target_coord:
            Source coordinate
        nside:
            Healpix NSIDE
        scheme:
            Healpix pixel ordering scheme
        base:
            HealpixBase defining the grid. Alternative to nside & scheme.

        Returns
        -------
        mhealpy.containers.healpix_map.HealpixMap
            The dwell obstime map.
        """

        # Get source path
        src_path_skycoord = self.get_target_in_sc_frame(target_coord)

        # Empty map
        dwell_map = HealpixMap(nside = nside,
                               scheme = scheme,
                               base = base,
                               coordsys = SpacecraftFrame())

        # Fill
        # Get the unique pixels to weight, and sum all the correspondint weights first, so
        # each pixels needs to be called only once.
        # Based on https://stackoverflow.com/questions/23268605/grouping-indices-of-unique-elements-in-numpy

        # remove the last value. Effectively a 0th order interpolations
        pixels, weights = dwell_map.get_interp_weights(theta=src_path_skycoord[:-1])

        weighted_duration = weights * self.livetime.to_value(u.second)[None]

        pixels = pixels.flatten()
        weighted_duration = weighted_duration.flatten()

        pixels_argsort = np.argsort(pixels)

        pixels = pixels[pixels_argsort]
        weighted_duration = weighted_duration[pixels_argsort]

        first_unique = np.concatenate(([True], pixels[1:] != pixels[:-1]))

        pixel_unique = pixels[first_unique]

        splits = np.nonzero(first_unique)[0][1:]
        pixel_durations = [np.sum(weighted_duration[start:stop]) for start, stop in
                           zip(np.append(0, splits), np.append(splits, pixels.size))]

        for pix, dur in zip(pixel_unique, pixel_durations):
            dwell_map[pix] += dur

        dwell_map.to(u.second, update=False, copy=False)

        return dwell_map

    def get_scatt_map(self,
                      nside,
                      target_coord=None,
                      earth_occ=True,
                      angle_nbins=None) -> SpacecraftAttitudeMap:

        """
        Bin the spacecraft attitude history into a list of discretized
        attitudes with associated time weights.  Discretization is
        performed on the rotation-vector representation of the
        attitude; the supplied nside parameter describes a HEALPix
        grid that discretizes the rotvec's direction, while a multiple
        of nside defines the number of bins to discretize its angle.

        If a target coordinate is provided and earth_occ is True,
        attitudes for which the view of the target is occluded by
        the earth are excluded.

        Parameters
        ----------
        nside : int
            The nside of the scatt map.
        target_coord : astropy.coordinates.SkyCoord, optional
            The coordinates of the target object.
        earth_occ : bool, optional
            Option to include Earth occultation in scatt map calculation.
            Default is True.
        angle_nbins : int (optional)
            Number of bins used for the rotvec's angle. If none
            specified, default is 8*nside

        Returns
        -------
        cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap
            The spacecraft attitude map.

        """

        def _cart_to_polar(v):
            """
            Convert Cartesian 3D unit direction vectors to polar coordinates.

            Parameters
            ----------
            v : np.ndarray(float) [N x 3]
              array of N 3D unit vectors

            Returns
            -------
            lon, colat : np.ndarray(float) [N]
              longitude and co-latitude corresponding to v in radians

            """

            lon = np.arctan2(v[:, 1], v[:, 0])
            colat = np.arccos(v[:, 2])
            return (lon, colat)

        source = target_coord

        if earth_occ:

            # earth radius
            r_earth = 6378.0

            # Need a source location to compute earth occultation
            if source is None:
                raise ValueError("target_coord is needed when earth_occ is True")

            # calculate angle between source direction and Earth zenith
            # for each time stamp
            src_angle = source.separation(self.earth_zenith)

            # get max angle based on altitude
            max_angle = np.pi - np.arcsin(r_earth/(r_earth + self.location.spherical.distance.km))

            # get pointings that are occluded by Earth
            is_occluded = src_angle.rad >= max_angle

            # zero out weights of time bins corresponding to occluded pointings
            time_weights = np.where(is_occluded[:-1], 0, self.livetime.value)

        else:
            source = None # w/o occultation, result is not dependent on source
            time_weights = self.livetime.value

        # Get orientations as rotation vectors (center dir, angle around center)

        rot_vecs   = self._attitude[:-1].as_rotvec()
        rot_angles = np.linalg.norm(rot_vecs, axis=-1)
        rot_dirs   = rot_vecs / rot_angles[:,None]

        # discretize rotvecs for input Attitudes

        dir_axis = HealpixAxis(nside=nside, coordsys=self._attitude.frame)

        if angle_nbins is None:
            angle_nbins = 8*nside

        angle_axis = Axis(np.linspace(0., 2*np.pi, num=angle_nbins+1), unit=u.rad)

        r_lon, r_colat = _cart_to_polar(rot_dirs.value)

        dir_bins = dir_axis.find_bin(theta=r_colat,
                                     phi=r_lon)
        angle_bins = angle_axis.find_bin(rot_angles)

        # compute list of unique rotvec bins occurring in input,
        # along with mapping from time to rotvec bin
        shape = (dir_axis.nbins, angle_axis.nbins)

        att_bins = np.ravel_multi_index((dir_bins, angle_bins),
                                        shape)

        # compute an Attitude for each unique rotvec bin

        unique_atts, time_to_att_map = np.unique(att_bins,
                                                 return_inverse=True)
        (unique_dirs, unique_angles) = np.unravel_index(unique_atts,
                                                        shape)
        v = dir_axis.pix2vec(unique_dirs)

        binned_attitudes = Attitude.from_rotvec(np.column_stack(v) *
                                                angle_axis.centers[unique_angles][:,None],
                                                frame = self._attitude.frame)

        # sum weights for all attitudes mapping to each bin
        binned_weights = np.zeros(len(unique_atts))
        np.add.at(binned_weights, time_to_att_map, time_weights)

        # remove any attitudes with zero weight
        binned_attitudes = binned_attitudes[binned_weights > 0]
        binned_weights   = binned_weights[binned_weights > 0]

        return SpacecraftAttitudeMap(binned_attitudes,
                                     u.Quantity(binned_weights, unit=self.livetime.unit, copy=False),
                                     source = source)

    @staticmethod
    def _sparse_sum_duplicates(indices, weights=None, dtype=None):
        """
        Given an array of indices, possibly with duplicates, and an
        optional array of weights per index (defaults to all ones if
        None), return a sorted array of the unique values in indices
        and, for each, the sum of the weights for each unique index.

        If input weights are provided, only unique indices with
        nonzero weights are returned.

        Parameters
        ----------
        indices : array of int
        weights : array of int or float type
        dtype : data type (optional)
           Type of returned weights.  If None, type is int if weights
           not given, or float64 if they are.

        Returns
        -------
          - unique_indices : array of int
             sorted unique indices in input
          - idx_weights : array of type as described above
             sum of weights for each unique index in input

        """

        if weights is None:
            unique_indices, idx_weights = np.unique(indices,
                                                    return_counts=True)
        else:
            sp_weights = np.bincount(indices, weights)
            unique_indices = np.flatnonzero(sp_weights)
            idx_weights = sp_weights[unique_indices]

        if dtype is not None:
            idx_weights = idx_weights.astype(dtype, copy=False)

        return unique_indices, idx_weights

    def get_exposure(self, base, theta, phi=None,
                     lonlat=False, interp=True):
        """
        Compute the set of exposed HEALPix pixels relative to a
        HealpixBase arising from a sequence of spacecraft-frame
        directions with durations as specified in this SpacecraftFile.

        If theta is a SkyCoord, it specifies the full direction.
        Else, theta and phi specify the direction as angles.  If
        lonlat = True, theta and phi are longitude and latitude in
        degrees; else, theta and phi are co-latitude and longitude in
        radians.

        Parameters
        ----------
        base : HealpixBase
           HEALPix grid used to discretize exposure
        theta : np.ndarray or SkyCoord
           if phi is None, a vector SkyCoord
           if phi is not none, a vector of angles
        colat: np.ndarray, optional
           a vector of angles
        interp : bool, optional
             If True, interpolate the weights onto the HEALPix grid;
             else, just map to nearest bin. (Default: interpolate)

        Returns
        -------
        pixels : np.ndarray (int)
          all HEALPix pixels in the grid with nonzero exposure time
        exposures: np.ndarray (float)
          exposure time for each pixel

        """

        duration = self.livetime.to_value(u.s)

        if len(duration) + 1 != len(theta):
            raise ValueError("Source path must have length equal to # times in SpacecraftFile")

        # remove the last src location. Effectively a 0th-order
        # interpolation
        theta = theta[:-1]
        if phi is not None:
            phi = phi[:-1]

        if interp:
            pixels, weights = base.get_interp_weights(theta=theta,
                                                      phi=phi,
                                                      lonlat=lonlat)
            weighted_duration = weights * duration[None]
        else:
            # do not interpolate
            pixels = base.ang2pix(theta=theta,
                                  phi=phi,
                                  lonlat=lonlat)
            weighted_duration = duration

        unique_pixels, unique_weights = \
            self._sparse_sum_duplicates(pixels.ravel(),
                                        weighted_duration.ravel(),
                                        dtype=np.float32)

        return unique_pixels, unique_weights


