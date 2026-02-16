from cosipy.response import FullDetectorResponse
from cosipy import test_data
from cosipy import SpacecraftHistory
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import os
from pathlib import Path
from astropy.time import Time 

def test_get_time():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    
    ori = SpacecraftHistory.open(ori_path)
    
    assert np.allclose(ori.obstime.unix,
                       [1835478000.0, 1835478001.0, 1835478002.0,
                        1835478003.0, 1835478004.0, 1835478005.0,
                        1835478006.0, 1835478007.0, 1835478008.0,
                        1835478009.0, 1835478010.0])


def test_read_only_selected_range():

    ori_path = test_data.path / "20280301_first_10sec.ori"

    ori = SpacecraftHistory.open(ori_path,
                                 tstart=Time(1835478002.0, format = 'unix'),
                                 tstop = Time(1835478008.0, format='unix')
                                 )

    assert np.allclose(ori.obstime.unix,
                       [1835478002.0,
                        1835478003.0, 1835478004.0, 1835478005.0,
                        1835478006.0, 1835478007.0, 1835478008.0, 1835478009.0])

    ori = SpacecraftHistory.open(ori_path,
                                 tstart=Time(1835478002.5, format = 'unix'),
                                 tstop = Time(1835478007.5, format='unix')
                                 )

    assert np.allclose(ori.obstime.unix,
                       [1835478002.0,
                        1835478003.0, 1835478004.0, 1835478005.0,
                        1835478006.0, 1835478007.0, 1835478008.0])

def test_get_time_delta():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)
    time_delta = ori.intervals_duration.to_value(u.s)

    assert np.allclose(time_delta, np.array([1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                                             1.000000, 1.000000, 1.000000, 1.000000, 1.000000]))

    time_delta = ori.livetime.to_value(u.s)

    assert np.allclose(time_delta, np.array([1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                                             1.000000, 1.000000, 1.000000, 1.000000, 1.000000]))

def test_get_attitude():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)
    
    attitude = ori.attitude

    matrix = np.array([[[0.215904, -0.667290, -0.712818],
                        [0.193436, 0.744798, -0.638638],
                        [0.957062, 0.000000, 0.289883]],
                       
                       [[0.216493, -0.667602, -0.712347],
                        [0.194127, 0.744518, -0.638754],
                        [0.956789, 0.000000, 0.290783]],
                       
                       [[0.217081, -0.667914, -0.711875],
                        [0.194819, 0.744238, -0.638870],
                        [0.956515, -0.000000, 0.291683]],
                       
                       [[0.217669, -0.668227, -0.711402],
                        [0.195511, 0.743958, -0.638985],
                        [0.956240, 0.000000, 0.292582]],
                       
                       [[0.218255, -0.668539, -0.710929],
                        [0.196204, 0.743677, -0.639100],
                        [0.955965, 0.000000, 0.293481]],
                       
                       [[0.218841, -0.668852, -0.710455],
                        [0.196897, 0.743396, -0.639214],
                        [0.955688, -0.000000, 0.294380]],
                       
                       [[0.219426, -0.669165, -0.709980],
                        [0.197590, 0.743114, -0.639327],
                        [0.955411, 0.000000, 0.295279]],
                       
                       [[0.220010, -0.669477, -0.709504],
                        [0.198284, 0.742833, -0.639440],
                        [0.955133, -0.000000, 0.296177]],
                       
                       [[0.220594, -0.669790, -0.709027],
                        [0.198978, 0.742551, -0.639552],
                        [0.954854, 0.000000, 0.297075]],
                       
                       [[0.221176, -0.670103, -0.708550],
                        [0.199673, 0.742268, -0.639663],
                        [0.954574, -0.000000, 0.297973]],
                       
                       [[0.221758, -0.670416, -0.708072],
                        [0.200368, 0.741986, -0.639773],
                        [0.954294, -0.000000, 0.298871]]])
    
    assert np.allclose(attitude.as_matrix(), matrix)


def test_interp_attitude():
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)

    assert np.allclose(ori.interp_attitude(Time(1835478000.5, format = 'unix')).as_quat(), [ 0.21284241, -0.55635581,  0.28699984,  0.75019825])

    # Multiple
    assert np.allclose(ori.interp_attitude(Time([1835478000.5, 1835478000.5], format='unix')).as_quat(),
                       [[0.21284241, -0.55635581, 0.28699984, 0.75019825],[0.21284241, -0.55635581, 0.28699984, 0.75019825]])

    # Test edges
    assert np.allclose(ori.interp_attitude(Time(1835478000.0, format='unix')).as_quat(), ori.attitude[0].as_quat())
    assert np.allclose(ori.interp_attitude(Time(1835478001.0, format='unix')).as_quat(), ori.attitude[1].as_quat())

def test_interp_location():
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)

    assert np.allclose(ori.interp_location(Time(1835478000.5, format = 'unix')).cartesian.xyz.to_value(u.km), [ -378.74248737, -6048.59116724, -3346.84533097])

    # Multiple
    assert np.allclose(ori.interp_location(Time([1835478000.5,1835478000.5], format='unix')).cartesian.xyz.to_value(u.km),
                       np.transpose([[-378.74248737, -6048.59116724, -3346.84533097],[-378.74248737, -6048.59116724, -3346.84533097]]))

    # Test edges
    assert np.allclose(ori.interp_location(Time(1835478000.0, format='unix')).cartesian.xyz.to_value(u.km), ori.location[0].cartesian.xyz.to_value(u.km))
    assert np.allclose(ori.interp_location(Time(1835478001.0, format='unix')).cartesian.xyz.to_value(u.km), ori.location[1].cartesian.xyz.to_value(u.km))


def test_get_target_in_sc_frame():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)

    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_coord)

    assert np.allclose(path_in_sc.lon.deg, 
                       np.array([118.393522, 118.425255, 118.456868, 118.488362, 118.519735, 
                                 118.550989, 118.582124, 118.613139, 118.644035, 118.674813, 118.705471]))

    assert np.allclose(path_in_sc.lat.deg, 
                       np.array([46.733430, 46.687559, 46.641664, 46.595745, 46.549801, 46.503833,
                                 46.457841, 46.411825, 46.365785, 46.319722, 46.273634]))


def test_get_dwell_map():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)
    
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    dwell_map = ori.get_dwell_map(target_coord, nside=1, scheme = 'ring')
    
    assert np.allclose(dwell_map[:].to_value(u.s),
                       np.array([1.895057, 7.615584, 0.244679, 0.244679, 0.000000, 0.000000, 
                                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]))

def test_select_interval():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftHistory.open(ori_path)

    new_ori = ori.select_interval(ori.tstart+0.1*u.s, ori.tstart+2.1*u.s)

    x, y, z = new_ori.attitude.as_axes()

    assert np.allclose(new_ori.obstime.unix,
                       np.array([1.835478e+09, 1.835478e+09, 1.835478e+09, 1.835478e+09]))

    assert np.allclose(np.asarray([x.transform_to('galactic').l.deg, x.transform_to('galactic').b.deg]).transpose().flatten(),
                       np.array([41.86062093, 73.14368765, 41.88225011, 73.09517927, 
                                 41.90629597, 73.0412838 , 41.9087019 , 73.03589454]))

    assert np.allclose(np.asarray([z.transform_to('galactic').l.deg, z.transform_to('galactic').b.deg]).transpose().flatten(),
                       np.array([221.86062093,  16.85631235, 221.88225011,  16.90482073,
                                221.90629597,  16.9587162 , 221.9087019 ,  16.96410546]))

    # Edge cases
    new_ori = ori.select_interval(ori.tstart, ori.tstop)
    assert np.all(new_ori.obstime == ori.obstime)

    new_ori = ori.select_interval(ori.obstime[1], ori.tstop)
    assert np.all(new_ori.obstime == ori.obstime[1:])

    new_ori = ori.select_interval(ori.tstart, ori.obstime[-2])
    assert np.all(new_ori.obstime == ori.obstime[:-1])

    # Fully within single interval
    new_ori = ori.select_interval(ori.tstart + .4*u.s, ori.tstart + .6*u.s)
    assert new_ori.tstart == ori.tstart + .4*u.s
    assert new_ori.tstop == ori.tstart + .6*u.s
    assert new_ori.nintervals == 1
    assert np.isclose(new_ori.livetime[0], 0.2*u.s)

