import numpy as np
import warnings
#import healvis as hv
import yaml, time
import copy
from inspect import getfullargspec

def build_hex_array(hex_spec=(3,4), ants_per_row=None, d=14.6):
    """
    Build an antenna position dict for a hexagonally close-packed array.

    Parameters
    ----------
    hex_spec : tuple, optional
        If `ants_per_row = None`, this is used to specify a hex array as
        `hex_spec = (nmin, nmax)`, where `nmin` is the number of antennas in
        the bottom and top rows, and `nmax` is the number in the middle row.
        The number per row increases by 1 until the middle row is reached.

        Default: (3,4) [a hex with 3,4,3 antennas per row]

    ants_per_row : array_like, optional
        Number of antennas per row. Default: None.

    d : float, optional
        Minimum baseline length between antennas in the hex array, in meters.
        Default: 14.6.

    Returns
    -------
    ants : dict
        Dictionary with antenna IDs as the keys, and tuples with antenna
        (x, y, z) position values (with respect to the array center) as the
        values. Units: meters.
    """
    ants = {}

    # If ants_per_row isn't given, build it from hex_spec
    if ants_per_row is None:
        r = np.arange(hex_spec[0], hex_spec[1]+1).tolist()
        ants_per_row = r[:-1] + r[::-1]

    # Assign antennas
    k = -1
    y = 0.
    dy = d * np.sqrt(3) / 2. # delta y = d sin(60 deg)
    for j, r in enumerate(ants_per_row):

        # Calculate y coord and x offset
        y = -0.5 * dy * (len(ants_per_row)-1) + dy * j
        x = np.linspace(-d*(r-1)/2., d*(r-1)/2., r)
        for i in range(r):
            k += 1
            ants[k] = (x[i], y, 0.)

    return ants

def check_number(a, n, what):
    if len(a) < n:
        raise RuntimeError("Not enough "+what+" supplied. Expected "+str(n))

def build_args(what, yaml_args={}):
    arg_spec = getfullargspec(what)
    required_args = []
    default_args = {}

    len_args = len(arg_spec.args)
    len_defaults = 0 if arg_spec.defaults is None else len(arg_spec.defaults)
    for i, name in enumerate(arg_spec.args):
        if not (name == "self" and i == 0):
            defaults_index = len_defaults-(len_args-i)
            if defaults_index >= 0:
                default_args[name] = arg_spec.defaults[defaults_index]
            else: 
                required_args.append(name)

    # Check required args are present in yaml and extract them
    required = {}
    for name in required_args:
        if name not in yaml_args.keys():
            raise ValueError("Required arg '"+name+"' missing for "+str(what))
        required[name] = yaml_args[name]
        del yaml_args[name]

    # Overwrite default args with values from yaml 
    defaults = { **default_args, **yaml_args }

    return required, defaults

def reduce_dict(d, these_out):
    reduced = copy.copy(d)
    for key in these_out:
        if key in reduced.keys(): del reduced[key]

    return reduced


def define_antennas(cfg_ant):

    if cfg_ant["origin"] == "hex" and "number" in cfg_ant:
        warnings.warn("number of antennas is ignored when origin == hex", file=sys.stderr)
    elif cfg_ant["origin"] == "random" and cfg_ant["number"] == "all":
        raise ValueError("number cannot be \"all\" if origin is \"random\"")

    number = cfg_ant["number"]

    if cfg_ant["origin"] == "random":
        # Random antenna locations
        x = np.random.random(number)*cfg_ant["x_max"]    
        y = np.random.random(number)*cfg_ant["y_max"]
        z = np.random.random(number)*cfg_ant["z_max"]
        ants = {}
        for i in range(number):
            ants[i] = ( x[i], y[i], z[i] )

    elif cfg_ant["origin"] == "hex":
        ants = build_hex_array(**cfg_ant)

    elif cfg_ant["origin"] == "list":
        lants = np.array(cfg_ant["list"])
        if number == "all": number = len(lants)
        check_number(lants, number, "antennas")
        ants = {}
        for i in range(number):
            # Each ant: x, y, z in metres.
            ants[i] = (lants[i, 0], lants[i, 1], lants[i, 2])

    elif cfg_ant["origin"] == "file":
        fants = np.loadtxt(cfg_ant["file_name"])
        if number == "all": number = len(fants)
        check_number(fants, number, "antennas")
        ants = {}
        for i in range(number):
            # Each ant: x, y, z in metres.
            ants[i] = (fants[i, 0], fants[i, 1], fants[i, 2])
    else:
        raise ValueError("Unknown origin specified for antennas")
    
    return ants


def define_sources(cfg_source):
    if cfg_source["origin"] == "random" and cfg_source["number"] == "all":
        raise ValueError("number cannot be \"all\" if origin is \"random\"")


    number = cfg_source["number"]
   
    if cfg_source["origin"] == "random":
        # This gets complicated if we want to be sure there are a fixed number of sources in the sky at every time step.
        # This doesn't do that. 
        ra = np.random.uniform(cfg_source["ra_min"], cfg_source["ra_max"])          # degrees longitude
        dec = np.random.uniform(cfg_source["dec_min"], cfg_source["dec_max"])       # degrees latitude
        flux = np.random.uniform(cfg_source["flux_min"], cfg_source["flux_max"])
        spectral_index = np.random.uniform(cfg_source["spectral_index_min"], cfg_source["spectral_index_max"])
        
    elif cfg_source["origin"] == "file":
        sources = np.loadtxt(cfg_source["file_name"])[:cfg_source["number"]]
        if number == "all": number = len(sources)
        check_number(sources, number, "sources")
        ra = sources[:number, 0].T
        dec = sources[:number, 1].T
        flux = sources[:number, 2].T
        spectral_index = sources[:number, 3].T

    elif cfg_source["origin"] == "list":
        sources = np.array(cfg_source["list"])
        if number == "all": number = len(sources)
        check_number(sources, number, "sources")
        ra = sources[:number, 0]
        dec = sources[:number, 1]
        flux = sources[:number, 2]
        spectral_index = sources[:number, 3]

    else:
        raise ValueError("Unknown origin specified for sources")
    
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    
    return ra, dec, flux, spectral_index

def define_beams(cfg_beam, number):
    
    beam_params = reduce_dict(cfg_beam, ["beam_type"])

    if cfg_beam["beam_type"] == "pyuvsim_analytic":
        from pyuvsim.analyticbeam import AnalyticBeam
        one_beam = AnalyticBeam(**beam_params)
        beam_list = [one_beam for i in range(number)]

    elif cfg_beam["beam_type"] == "polybeam":
        from hera_sim.beams import PolyBeam
        one_beam = PolyBeam(**beam_params)
        beam_list = [one_beam for i in range(number)]

    elif cfg_beam["beam_type"] == "perturbed_polybeam":
        from hera_sim.beams import PerturbedPolyBeam
        one_beam = PerturbedPolyBeam(**beam_params)
        beam_list = [one_beam for i in range(number)]
    else:
        raise ValueError("Unknown beam type")

    return beam_list 

# Not used
def telescope_config(which_package, nant=2, nfreq=2, ntime=1, nsource=1):
    """
    Setup the configuration parameters for pyuvsim/hera_sim/healvis.
    Different packages require different objects for simulation.
    healvis not used here.
    """
    if which_package not in [ "hera_sim", "pyuvsim", "healvis" ]:
        raise ValueError("Unknown package: "+which_package)
        
    np.random.seed(10)          # So we always get the same random values

    # Random antenna locations
    x = np.random.random(nant)*400     # Up to 400 metres
    y = np.random.random(nant)*400
    z = np.random.random(nant)
    ants = {}
    for i in range(nant):
        ants[i] = ( x[i], y[i], z[i] )
        
    # Observing parameters in a UVData object.
    uvdata = io.empty_uvdata(
        nfreq = nfreq,             
        start_freq = cfg_in["start_freq"],
        channel_width = cfg_in["chan_width"],
        start_time = cfg_in["start_time"],
        integration_time = cfg_in["integration_time"],
        ntimes = cfg_in["ntimes"],
        ants = ants,
        polarization_array = np.array([ "XX", "YY", "XY", "YX" ]),
        Npols = 4
    )
    
    # Random sources.
    sources = [
        [ 125.7, -30.72, 2, 0 ],     # Fix a source near zenith, which is at 130.7   -30.72
        ]
    if nsource > 1:                  # Add random others
        ra = 100+np.random.random(nsource-1)*50
        dec = -10.72+np.random.random(nsource-1)*40
        flux = np.random.random(nsource-1)*4
        for i in range(nsource-1): sources.append([ ra[i], dec[i], flux[i], 0])
    sources = np.array(sources)

    # Source locations and frequencies.
    ra_dec = np.deg2rad(sources[:, :2])
    freqs = np.unique(uvdata.freq_array)

    if which_package == "hera_sim":
        # calculate source fluxes for hera_sim. pyuvsim does it a different way.
        flux = (freqs[:,np.newaxis]/freqs[0])**sources[:,3].T*sources[:,2].T      
        beam_ids = list(ants.keys())

    beam_dict = {}
    for i in range(len(beam)): beam_dict[str(i)] = i

    # That's enough for hera_sim, but extra objects are required for pyuvsim and healvis.
    
    if which_package == "pyuvsim":
        # Need a sky model.
        
        # Stokes for the first frequency only. Stokes for other frequencies
        # are calculated later.
        stokes = np.zeros((4, 1, ra_dec.shape[0]))
        stokes[0, 0] = sources[:, 2]
        reference_frequency = np.full(len(ra_dec), freqs[0])
        
        # Setup sky model.
        sky_model = SkyModel(name=[ str(i) for i in range(len(ra_dec)) ],
            ra=Longitude(ra_dec[:, 0], "rad"), dec=Latitude(ra_dec[:, 1], "rad"),
            spectral_type="spectral_index",
            spectral_index=sources[:,3],
            stokes =stokes,
            reference_frequency=Quantity(reference_frequency, "Hz")
            )

        # Calculate stokes at all the frequencies.
        sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)
        
    if which_package == "healvis":
        # Need a GSM model and an Observatory.
        
        baselines = []
        for i in range(len(ants)):
            for j in range(i+1, len(ants)):
                bl = hv.observatory.Baseline(ants[i], ants[j], i, j)
                baselines.append(bl)

        times = np.unique(uvdata.get_times("XX"))

        obs_latitude=-30.7215277777
        obs_longitude = 21.4283055554
        obs_height = 1073

        # create the observatory
        fov = 360  # Deg
        obs = hv.observatory.Observatory(obs_latitude, obs_longitude, obs_height, array=baselines, freqs=freqs)
        obs.set_pointings(times)
        obs.set_fov(fov)
        obs.set_beam(beam)

        gsm = hv.sky_model.construct_skymodel('gsm', freqs=freqs, Nside=64)
    
    
        
    # Return what is relevant for each package, pyuvsim or hera_sim
    if which_package == "hera_sim":
        return uvdata, beam, beam_dict, freqs, ra_dec, flux
    elif which_package == "pyuvsim":
        return uvdata, beam, beam_dict, sky_model
    elif which_package == "healvis":
        return obs, gsm

def run_sim(yaml_file):
    with open(yaml_file) as f:
        cfg_in = yaml.load(f, Loader=yaml.FullLoader)


    if cfg_in["simulator"] in [ "hera_sim", "pyuvsim"]:
        from hera_sim import io

        antennas = define_antennas(cfg_in["antenna_spec"])
        ra, dec, flux, spectral_index = define_sources(cfg_in["source_spec"])
        beam_list = define_beams(cfg_in["beam_spec"], len(antennas.keys()))

        strip_spec = reduce_dict(cfg_in["sim_spec"], ["nfreq", "ntime"])
        
        # Observing parameters in a UVData object.
        uvdata = io.empty_uvdata(
            cfg_in["sim_spec"]["nfreq"],
            cfg_in["sim_spec"]["ntime"],
            antennas,
            **strip_spec
        )
        
        freqs = np.unique(uvdata.freq_array)

        if cfg_in["simulator"] in [ "hera_sim" ]:
            from hera_sim.visibilities import VisCPU
            
            # calculate source fluxes for hera_sim
            flux = (freqs[:, np.newaxis]/freqs[0])**spectral_index*flux

            simulator = VisCPU(
                    uvdata = uvdata,
                    beams = beam_list,
                    beam_ids = list(antennas.keys()),
                    sky_freqs = freqs,
                    point_source_pos = np.column_stack((ra, dec)),
                    point_source_flux = flux,
                    **cfg_in["vis_cpu"]
            )

            start = time.time()
            simulator.simulate()
            hera_sim_time = time.time()-start
            return simulator.uvdata, hera_sim_time 

        elif cfg_in["simulator"] == "pyuvsim":   # pyuvsim
            from astropy.coordinates import Latitude, Longitude
            from astropy.units import Quantity
            from pyuvsim import uvsim
            from pyuvsim.telescope import BeamList
            from pyradiosky import SkyModel
            from pyuvsim import simsetup

            ra_dec = np.column_stack((ra, dec))

            # Stokes for the first frequency only. Stokes for other frequencies
            # are calculated later.
            stokes = np.zeros((4, 1, ra_dec.shape[0]))
            stokes[0, 0] = flux

            reference_frequency = np.full(len(ra_dec), freqs[0])

            # Setup sky model.
            sky_model = SkyModel(name=[ str(i) for i in range(len(ra_dec)) ],
                ra=Longitude(ra_dec[:, 0], "rad"), dec=Latitude(ra_dec[:, 1], "rad"),
                spectral_type="spectral_index",
                spectral_index=spectral_index,
                stokes=stokes,
                reference_frequency=Quantity(reference_frequency, "Hz")
                )

            # Calculate stokes at all the frequencies.
            sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)

            beam_dict = {}
            for i in range(len(beam_list)): beam_dict[str(i)] = i

            start = time.time()
            pyuvsim_uvd = uvsim.run_uvdata_uvsim(uvdata, BeamList(beam_list), beam_dict=beam_dict,
                    catalog=simsetup.SkyModelData(sky_model))
            pyuvsim_time = time.time()-start

            return pyuvsim_uvd, pyuvsim_time

    elif cfg_in["simulator"] == "healvis":
        import healvis as hv

        antennas = define_antennas(cfg_in["antenna_spec"])

        required_args, default_args = build_args(hv.observatory.Baseline, {})
        updated_healvis = "ant1" in default_args.keys()

        baselines = []
        if updated_healvis:
            for i in range(len(antennas)):
                for j in range(i+1, len(antennas)):
                    bl = hv.observatory.Baseline(antennas[i], antennas[j], i, j)
                    baselines.append(bl)
        else:
            for i in range(len(antennas)):
                for j in range(i+1, len(antennas)):
                    bl = hv.observatory.Baseline(antennas[i], antennas[j], i, j)
                    baselines.append(bl)

        freqs = np.arange(cfg_in["sim_spec"]["start_freq"], 
                    cfg_in["sim_spec"]["start_freq"]+cfg_in["sim_spec"]["nfreq"]*cfg_in["sim_spec"]["channel_width"], 
                    cfg_in["sim_spec"]["channel_width"])

        integration_time = cfg_in["sim_spec"]["integration_time"]/float(24*60*60)    # fraction of a day
        times = np.arange(cfg_in["sim_spec"]["start_time"],        
                        cfg_in["sim_spec"]["start_time"]+cfg_in["sim_spec"]["ntime"]*cfg_in["sim_spec"]["integration_time"],
                        cfg_in["sim_spec"]["integration_time"])

        # Set observatory
        if "array" in cfg_in["healvis"]["observatory"].keys() or "freqs" in cfg_in["healvis"]["observatory"].keys():
            raise ValueError("Don't specify array/freqs in yaml file. Use other yaml params to define these.")
        required_args, default_args = build_args(hv.observatory.Observatory, cfg_in["healvis"]["observatory"])
        default_args = reduce_dict(default_args, ["array", "freqs"])
        obs = hv.observatory.Observatory(required_args["latitude"], required_args["longitude"], 
                            array=baselines, freqs=freqs, **default_args)

        obs.set_pointings(times)
        obs.set_fov(cfg_in["healvis"]["fov"])

        # Beams require a bit of finessing. 
        if cfg_in["beam_spec"]["beam_type"] == "healvis_internal":
            required_args, default_args = build_args(hv.observatory.Observatory.set_beam, cfg_in["beam_spec"])
            default_args = reduce_dict(default_args, ["beam_type"])
            obs.setbeam(**default_args)
        else:
            beam_list = define_beams(cfg_in["beam_spec"], len(antennas.keys()))
            obs.set_beam(beam_list)
        
        # Set sky_model
        if "freqs" in cfg_in["healvis"]["sky_model"]:
            raise ValueError("Don't specify freqs in yaml file. Use other yaml params to define these.")
        required_args, default_args = build_args(hv.sky_model.construct_skymodel, cfg_in["healvis"]["sky_model"])
        default_args = reduce_dict(default_args, ["freqs"])
        if required_args["sky_type"] == "flat_spec":
            sky = hv.sky_model.construct_skymodel("flat_spec", freqs=freqs, **default_args)
        elif required_args["sky_type"] == "gsm":
            # create a PyGSM sky
            sky = hv.sky_model.construct_skymodel("gsm", freqs=freqs, **default_args)
        else:
            raise RuntimeError("Unknown healvis sky model")

        start_time = time.time()
        vis, times, bls = obs.make_visibilities(sky, beam_pol='XX')
        healvis_time = time.time()-start_time

        return vis, healvis_time

    else:
        raise RuntimeError("Unknown simulator")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        raise RuntimeError("Error: Expecting one yaml file.")

    run_sim(sys.argv[1])
