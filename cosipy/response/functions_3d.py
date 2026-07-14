import logging

import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.coordinates import Galactic

from histpy import Histogram


logger = logging.getLogger(__name__)


def get_integrated_extended_model_3d(
    extendedmodel,
    image_axis,
    energy_axis,
):
    """
    Calculate the integrated flux map for an extended 3D source model.

    Parameters
    ----------
    extendedmodel : astromodels.ExtendedSource
        An astromodels extended source model object.

    image_axis : histpy.HealpixAxis
        Spatial axis for the image.

    energy_axis : histpy.Axis
        Energy axis defining the energy bins.

    Returns
    -------
    flux_map : histpy.Histogram
        Integrated intensity as a function of sky position and energy.
    """

    from cosipy.threeml.custom_functions import GalpropHealpixModel

    if not isinstance(image_axis.coordsys, Galactic):
        raise ValueError(
            "The COSIPy image axis must use Galactic coordinates."
        )

    l, b = image_axis.pix2ang(
        np.arange(image_axis.npix),
        lonlat=True,
    )

    shape = extendedmodel.spatial_shape

    # Make sure the dummy spectral parameter is fixed.
    extendedmodel.spectrum.main.Constant.k.free = False

    # ------------------------------------------------------------------
    # Legacy COSIPy GalpropHealpixModel
    # ------------------------------------------------------------------

    if isinstance(shape, GalpropHealpixModel):

        # The norm is updated internally by 3ML for each likelihood call.
        norm = shape.K.value

        # Calculate the integrated flux map only once.
        if not isinstance(shape._result, np.ndarray):

            intensity = (
                (1 / norm)
                * shape.evaluate(
                    l,
                    b,
                    energy_axis.edges.to(u.MeV),
                    norm,
                )
            )

            shape.intg_flux = np.zeros(
                (
                    intensity.shape[0],
                    intensity.shape[1] - 1,
                )
            )

            energy_edges = energy_axis.edges.to_value(u.MeV)

            logger.info(
                "Integrating GalpropHealpixModel intensity "
                "over energy bins..."
            )

            for j in range(len(intensity)):

                interp_func = interp1d(
                    energy_edges,
                    intensity[j],
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                shape.intg_flux[j] = np.array(
                    [
                        integrate.quad(
                            interp_func,
                            lo_lim,
                            hi_lim,
                        )[0]
                        for lo_lim, hi_lim in zip(
                            energy_edges[:-1],
                            energy_edges[1:],
                        )
                    ]
                )

        flux = norm * shape.intg_flux

    # ------------------------------------------------------------------
    # Astromodels GalpropFitsMapCube
    # ------------------------------------------------------------------

    elif shape.name == "GalpropMap":

        # COSIPy's image axis supplies Galactic l and b, so the model
        # must interpret x and y as Galactic coordinates.
        if isinstance(shape._frame, str):

            frame_name = shape._frame.lower()

        else:

            frame_name = shape._frame.name.lower()


        if frame_name != "galactic":

            raise ValueError(
                "GalpropFitsMapCube must use Galactic coordinates "
                "for COSIPy. Call shape.set_frame(Galactic())."
            )
       
        # Calculate the K = 1 integrated flux map only once.
        if not hasattr(shape, "_cosipy_integrated_flux"):

            logger.info(
                "Evaluating GalpropFitsMapCube and integrating "
                "over COSI energy bins..."
            )

            intensity = shape.evaluate(
                l,
                b,
                energy_axis.edges,
                1.0,
                shape.hash.value,
            )

            # Evaluate() should return:
            #
            #     1 / (energy cm2 s sr)
            #
            # at every energy-bin edge.
            if not isinstance(intensity, u.Quantity):
                raise TypeError(
                    "GalpropFitsMapCube.evaluate() must return "
                    "an astropy Quantity."
                )

            # The model is evaluated at the energy-bin edges.
            # Integrating the linear interpolation between adjacent
            # edges is equivalent to the trapezoidal rule.
            delta_energy = np.diff(energy_axis.edges)

            integrated_flux = (
                0.5
                * (
                    intensity[:, :-1]
                    + intensity[:, 1:]
                )
                * delta_energy[np.newaxis, :]
            )

            integrated_flux = integrated_flux.to(
                (u.s * u.cm**2 * u.sr) ** (-1)
            )

            # Save the K = 1 integrated map.
            shape._cosipy_integrated_flux = (
                integrated_flux.value
            )

        # Apply the current 3ML normalization.
        flux = (
            shape.K.value
            * shape._cosipy_integrated_flux
        )

    else:

        raise NotImplementedError(
            f"Function3D model '{shape.name}' is not currently "
            "supported by COSIPy."
        )

    flux_map = Histogram(
        (image_axis, energy_axis),
        contents=flux,
        unit=(u.s * u.cm**2 * u.sr) ** (-1),
        copy_contents=False,
    )

    return flux_map
