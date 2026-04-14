Interfaces and protocols
========================

The cosipy library has been refactored to make the components required to compute a likelihood modular and interchangeable. These components include the data (binned or event-by-event), the instrument response function (IRF), the convolution of the response with a source model, background models, and the likelihood calculation itself.

The goal is to allow users and developers to experiment with new versions of any component—without needing to modify the library core. Implementations of these components do not need to live within cosipy; they only need to comply with the relevant interface definition. This design also preserves flexibility as analysis ideas evolve over the course of the mission, and it makes it easier to switch between different approaches (for example, binned versus unbinned analyses).

This modularity is achieved by defining protocols for the interfaces between components. In practice, these are class-level contracts that specify a well-defined set of methods, including their expected inputs and outputs, that other parts of the code can rely on. The protocols describe what must be provided, but they do not prescribe how computations are performed; that is left to the specific implementation.

Below we provide an overview of the available interfaces (used interchangeably with “protocols”) and how to use them. We start from the top-most level (the likelihood) and describe all other components needed to compute it. The best place to see practical examples in action is the `Crab spectral fit tutorials  <https://github.com/cositools/cosipy/tree/develop/docs/tutorials/spectral_fits/continuum_fit/crab>`_ (binned and unbinned). Note, however, that not every part of cosipy fully uses these protocols yet; some areas still refer to specific implementations rather than a generic interface.

Likelihood functions
--------------------

The ``LikelihoodInterface`` main method is `get_log_like()`, which a promise for all implementation for result a log-likelihood. The interface, by itself, doesn't define how this will be achieved. AS a concreate example, ``PoissonLikelihood`` needs the observed counts and the expected number of counts, and internally uses the Poisson likelihood function to return the likelihood value.

Expectation interfaces
----------------------

Likelihood functions such as ``PoissonLikelihood`` require the expected  number of events --on average-- under a model that includes both signal and background contributions. This requirement is codified by the ``ExpectationInterface``. Two variants are provided: ``BinnedExpectationInterface``, which returns the expected number of counts in each bin of a binned dataset, and ``ExpectationDensityInterface``, which provides (1) the total expected number of counts and (2) the expectation density, i.e., the expected counts per unit phase space. The expectation density can be understood as the limit of the expected counts in a bin divided by the bin size as the bin size approaches zero. This form is used by the ``UnbinnedLikelihood``.

Expected counts may originate from signal sources or backgrounds.

For signal modeling, the hypothetical source is represented by the `Source` class from `astromodels <https://astromodels.readthedocs.io>`_. This is captured by the `ThreeMLSourceResponseInterface`, which defines the method `set_source(source: Source)` for passing source properties (e.g., position, spectrum, polarization). On its own, ``ThreeMLSourceResponseInterface`` only specifies how a source is provided; it is combined with an ``ExpectationInterface`` to define an class that can accept a `Source` and produce expected counts. An example is ``BinnedThreeMLSourceResponseInterface`` (with a corresponding unbinned counterpart). Finally, ThreeMLModelFoldingInterface generalizes this concept to compute expectations for a collection of sources.

As you can see, protocols such as ``BinnedThreeMLSourceResponseInterface`` are intentionally abstract. This goal of this design is to constrain implementations as little as possible while still providing a reliable contract for the rest of the code. In practice, an implementation of ``BinnedThreeMLSourceResponseInterface`` will typically compute the expected number of counts by convolving ---i.e. integrate over the relevant dimensions such as time, energy, and direction--- an instrument response function (IRF) with a source model (e.g., spectrum, light curve, and/or morphology). However, the IRF itself is defined by a separate protocol (see below).

For backgrounds, we define `BackgroundInterface`. It is intentionally generic and only requires the ability to pass parameter values via ``set_parameters()``. These parameters are completely user-defined: the interface does not assign them any specific meaning. A common concrete example is a single background normalization, but more complex cases are also supported —for example, multiple background components with independent normalizations, or non-linear parameters such as an exponential index, or even the input to a complex simulation. In a fit, these typically act as nuisance parameters. As with the signal-side expectation, ``BackgroundInterface`` is combined with an ``ExpectationInterface`` to define a complete contract for the required inputs and outputs (i.e., an object that can accept background parameters and return the corresponding expected counts).

Instrument Response Functions
-----------------------------

The (far-field) instrument response function (IRF) is responsible for providing:
1. The effective area as a function of a photon energy and incoming direction (in spacecraft coordinates).
2. The probability of obtaining a set of measurement when a photon is detected.

These required outputs are defined in FarFieldInstrumentResponseFunctionInterface by the methods `effective_area_cm()` and `event_probability()`, respectively. Similar, for the binned analysis, there is BinnedInstrumentResponseInterface, which provided `differential_effective_area`, that is, the product effective_area_cm*event_probability integrated on each bin of a binned dataset.

For a Compton telescope, Typical measurements include the measured energyThe set of "measurement" is not defined by the IRF interfaces. This task is delegated to an `EventDataInterface` (see below).











