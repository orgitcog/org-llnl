import yaml
import logging
import sys
import simpy
from tqdm import tqdm

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "../")

from astropy.table import Table
from astropy.time import Time
from astropy.utils.iers import conf
from slewpy import (
    Hub,
    SpaceSensor,
    Pipe,
    ObsFixedTarget,
    Scheduler,
    Manager,
    LastObserved,
)
from slewpy.utils import CelestialBodySpline
from ssapy import AccelHarmonic, get_body, AccelKepler, SciPyPropagator

from ssapy.orbit import Orbit

#dawn/dust Sun-Syncronus polar obit.
sso_orbit = Orbit.fromTLETuple(("1 14781U 84021B   25036.48746153  .00002879  00000+0  31443-3 0  9993", "2 14781  97.7382   5.4938 0008812  36.8404 323.3417 14.88633284183172"))
print("SSO Polar Orbit, 600km: ",sso_orbit.keplerianElements)

iss_orbit = Orbit.fromTLETuple(("1 25544U 98067A   25044.58106731  .00014347  00000-0  25813-3 0  9997", "2 25544  51.6360 199.2431 0004086 311.6218  48.4420 15.50080818495996"))
print("ISS orbit ",iss_orbit.keplerianElements)


# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "../")

conf.auto_max_age = None

# logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)


def sn_exptime(target, sensor, **kwargs):
    return 1800


def agn_exptime(target, sensor, **kwargs):
    return 1200


def too_exptime(target, sensor, **kwargs):
    return 1800


# priority function that will count how many observations were taken in the
# defined period and adjust the priority higher or lower to match the desired
# target
def target_priority(target, sensor, **kwargs):
    ctime = sensor.env.t0.gps + sensor.env.now

    if len(target.obs_times) == 0:
        return 1

    obs_period = np.sum((ctime - target.obs_times) < target.obs_period)
    prior = target.obs_per_period / (obs_period + 1)

    return prior


# priority function that will linearly increase the
# priority after the most recent observation relative to desired
# average observation period
def target_priority2(target, sensor, **kwargs):
    ctime = sensor.env.t0.gps + sensor.env.now
    if len(target.obs_times) > 0:
        last_time = target.obs_times[-1]
    else:
        return sensor.env.t0.gps

    avg_period = target.obs_period / target.obs_per_period
    priority = (ctime - last_time) / avg_period

    return priority

def target_priority3(target, sensor, **kwargs):
    '''
    The average rate at which observations of this target should be made is 
        r = (number of observerations of the target that still need to be made) / (time remaining in the window for this target)
    
    For events that happen at rate r, the time until the next event is an exponential random variable.
    
    So sample this exponential "time until the next observation" for each target
     and the target with the lowest of these times should get the highest priority.
    '''
    num_obs_total = round(target.obs_per_period * (target.tfinal - target.tstart) / target.obs_period)
    num_obs_left = num_obs_total - len(target.obs_times)
    time_left = target.tfinal - sensor.env.now

    # Don't observe if we got required number of observations
    if num_obs_left <= 0:
        return -1e300

    target_rate = num_obs_left / time_left
    if sensor.rng:
        random = sensor.rng.uniform()
    else:
        random = np.random.rand()
    minus_time_til_next = np.log(random) / target_rate
    return minus_time_til_next

def target_priority4(target, sensor, **kwargs):
    '''
    Try to get regularly spaced observations.
    Time to the next observation should be (time of most recent obs) + (ideal interval between obs) - (current time)
    Target with the lowest time til next observation is highest priority.
    '''
    type = target.type

    obs_so_far = len(target.obs_times)

    now = sensor.env.now
    
    # first observation should be as soon after tstart as possible
    if obs_so_far == 0:
        time_til_next = target.tstart - now
        return -time_til_next

    last_obs = target.obs_times[-1] - sensor.env.t0.gps

    if type == 'AGN_low':
        interval = 3.*3600 # 8x per day
    
    elif type == 'AGN_med':
        interval = 12.*3600 # 2x per day
    
    elif type == 'AGN_high':
        interval = 2.*86400 # every other day
    
    elif type == 'SNe':
        interval = 2.5*86400 # every 2.5 days
   
    elif type == 'SNe_CC':
        #20 times in 3 days, then 10 times over rest of the 2 weeks
        interval = 3*86400/20 if (now - target.tstart < 3*86400) else 11*86400/10

    elif type == 'SNe_SED':
        # 2 observations as soon as possible, but don't take more than 2 obs
        interval = 0.0 if obs_so_far < 2 else 1e300

    elif type == 'TOO':
        #every day until you get 3 observations, then 2 more times over rest of the two weeks
        interval = 86400.0 if (obs_so_far < 3) else 12*86400/2
    
    else:
        raise Exception(f"target is unknown type: {target.type=}")

    time_til_next = last_obs + interval - now
    
    return -time_til_next


def target_priority5(target, sensor, **kwargs):
    '''
    Try to get regularly spaced observations.
    Adapts time-til-next-observation to try to achieve a given total number of observations.
    Time to the next observation is (time of most recent obs) + (ideal interval between obs) - (current time)
    Target with the lowest time til next observation is highest priority.

    priority = -(time til next obs)
    so if priority is negative it means target was observed ahead of ideal time,
     positive means it was observed after ideal time.
    '''
    type = target.type

    obs_so_far = len(target.obs_times)

    now = sensor.env.now
    
    # first observation should be as soon after tstart as possible
    if obs_so_far == 0:
        time_til_next = target.tstart - now
        return -time_til_next

    last_obs = target.obs_times[-1] - sensor.env.t0.gps

    if type == 'AGN_low':
        base_interval = 3.*3600 # 8x per day
        tstop = target.tstart + 90*86400. # 90 day duration
        adaptive_interval = (tstop - last_obs)/(720 - obs_so_far) # try to get 720 total obs
        interval = min(base_interval, adaptive_interval)
    
    elif type == 'AGN_med':
        base_interval = 12.*3600 # 2x per day
        tstop = target.tstart + 180*86400. # 180 day duration
        adaptive_interval = (tstop - last_obs)/(360 - obs_so_far) #try to get 360 total obs
        interval = min(base_interval, adaptive_interval)
    
    elif type == 'AGN_high':
        base_interval = 2.*86400 # every other day
        tstop = target.tstart + 730*86400. #730 day duration
        adaptive_interval = (tstop - last_obs)/(180 - obs_so_far) #try to get 180 total obs
        interval = min(base_interval, adaptive_interval)
    
    elif type == 'SNe':
        base_interval = 2.5*86400 # every 2.5 days
        tstop = target.tstart + 60*86400. # 60 day duration
        adaptive_interval = (tstop - last_obs)/(24 - obs_so_far) #try to get 24 total obs
        interval = min(base_interval, adaptive_interval)
   
    elif type == 'SNe_CC':
        #20 times in 3 days, then 10 times over rest of the 2 weeks
        phase1stop = target.tstart + 3*86400.
        if now < phase1stop:
            base_interval = 3*86400/20
            adaptive_interval = (phase1stop - last_obs)/(20 - obs_so_far)
            interval = min(base_interval, adaptive_interval)
        else:
            base_interval = 11*86400/10
            tstop = target.tstart + 14*86400. # 14 day duration
            adaptive_interval = (tstop - last_obs)/(30 - obs_so_far)
            interval = min(base_interval, adaptive_interval)

    elif type == 'SNe_SED':
        # 2 observations as soon as possible, but don't take more than 2 obs
        interval = 0.0 if obs_so_far < 2 else 1e300

    elif type == 'TOO':
        # 3 obs in first 3 days, then 2 more within the rest of the two weeks
        phase1stop = target.tstart + 3*86400.
        if obs_so_far < 3:
            base_interval = 86400.
            adaptive_interval = (phase1stop - last_obs)/(3 - obs_so_far)
            interval = min(base_interval, adaptive_interval)
        else:
            base_interval = 11*86400/2
            tstop = target.tstart + 14*86400. # 14 day duration
            adaptive_interval = (tstop - last_obs)/(5 - obs_so_far)
            interval = min(base_interval, adaptive_interval)
    
    else:
        raise Exception(f"target is unknown type: {target.type=}")

    time_til_next = last_obs + interval - now
    
    return -time_til_next


sensor_props = """
# Filter
n_exp: 1
exptime: 100
readout_time: 0
solar_exclusion: 90
lunar_exclusion: 20
limb_angle: 25
slew_speed: 1 # deg/sec
processing_time: 0
orbit: [7164172.36450565, 0.00, 0.00, 0.0, 0.0, 0.0, 630768548.816]
"""

nday = int(sys.argv[1])
num_sat = int(sys.argv[2])
orbit_type = str(sys.argv[3])
slurm_job_id = str(sys.argv[4])

seed = 53332
start = "J2000"
min_wait = 100  # how long to wait if nothing is seen
outdir = "./"
# Number of satellites in the constelation
# more than one number will run multiple sims
nsat = [num_sat]

#Load fixed targets
file_name = 'randomtargets_sunmoongal_720days.npz'# randomtargets_sunmoongal_days90to730.npz' 
fixed_target_input = pd.DataFrame(np.load(f'{file_name}',allow_pickle=True)['targetlist'])

agn_fixed_target = pd.read_csv("agn_schedule.csv") #pd.read_csv("../data/agn_schedule_manual.csv")

outname = f"slurm_{slurm_job_id}_nsats_{nsat[0]}_orbit_{orbit_type}_"
run_time = nday * 86400
sensor_dead_fraction_per_day = 0.1
start_time = Time(start)

# Setup and start the simulation
constellations = [[i * 2 * np.pi / n for i in range(n)] for n in nsat]
scheduler = Scheduler()

earth = get_body("earth", model='egm2008')
aEarth = AccelKepler() + AccelHarmonic(earth, 20, 20)
accel = aEarth # other terms we could include are: aMoon, aSun, aSolRad, aEarthRad, aDrag
# Build propagator
prop = SciPyPropagator(accel)

for num_sats, constellation in enumerate(constellations):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    request_pipe = Pipe(env)
    data_pipe = Pipe(env)
    target_pipe = Pipe(env)

    # Initialize celestial body splines for the entire simulation with a 2,000 second buffer
    # to account for predictions at the end of the simulation
    logging.info(f"Initializing celestial body splines for {run_time/86400:.1f} days")
    sun_spline = CelestialBodySpline.get_instance('sun', start_time.gps, start_time.gps + run_time + 2000, 10000)
    moon_spline = CelestialBodySpline.get_instance('moon', start_time.gps, start_time.gps + run_time + 2000, 10000)

    params1 = yaml.safe_load(sensor_props)
    sensors = []

    for n, satelite in enumerate(constellation):
        sso_orbit = f"orbit: [6986675.5882278755, 0.0019784439203286443, 1.7059452900052454, 0.9322367464746133, 0.08994138328427939, {satelite}, 630768548.816]"
        iss_orbit = f"orbit: [6801124.969361802, 0.001248390329540996, 0.9008123730631086, 0.4661182676371163, -2.8131957376595804, {satelite}, 630768548.816]"

        if orbit_type == "sso":
            orbit_str = sso_orbit
        elif orbit_type =="iss":
            orbit_str = iss_orbit
        else:
            raise ValueError("Orbit type can be sso or iss")
            
        file2 = sensor_props.replace(
            "orbit: [7164172.36450565, 0.00, 0.00, 0.0, 0.0, 0.0, 630768548.816]", orbit_str,
            )

        params2 = yaml.safe_load(file2)

        sensors.append(
            SpaceSensor(
                env=env,
                params=params2,
                request_pipe=request_pipe,
                data_pipe=data_pipe,
                name=f"sensor_{n+1}",
                min_wait=min_wait,
                dead_fraction_per_day=sensor_dead_fraction_per_day,
                propagator=prop,
                rng=rng,
            )
        )

    targets = []

    for _, agn_transient in agn_fixed_target.iterrows():
        
        target = ObsFixedTarget(
                    np.deg2rad(agn_transient['ra']),
                    np.deg2rad(agn_transient['dec']),
                    _id=_,
                    tstart=agn_transient['tstart'],
                    tfinal=agn_transient['tstart'] + agn_transient['duration'],
                    priority=target_priority4,
                    calc_exptime=agn_exptime,
                    type=agn_transient['type'],
                    obs_per_period=0, #this doesn't matter anymore with new priority functions
                    obs_period=86400,
                    )

        targets.append(target)

    for _, transient in fixed_target_input.iterrows():

        if transient['type'] != 'TOO' and transient['type'] != 'SNe':
            target = ObsFixedTarget(
                        np.deg2rad(transient['ra']),
                        np.deg2rad(transient['dec']),
                        _id=_,
                        tstart=transient['tstart'],
                        tfinal=transient['tstart'] + transient['duration'],
                        priority=target_priority4,
                        calc_exptime=sn_exptime,
                       type=transient['type'],
                       obs_per_period=0, #this
                      obs_period=86400,
                    )
            targets.append(target)

    t0 = Time(start_time, format="gps")
    hub = Hub(
        env=env,
        request_pipe=request_pipe,
        data_pipe=data_pipe,
        sensor_list=sensors,
        target_pipe=target_pipe,
        scheduler=scheduler,
        obs_targets=targets,
        t0=t0,
        runtime=run_time,
    )

    for i in tqdm(range(1,run_time)):
        env.run(until=i)

    # Record data outputs
    time = []
    tid = []
    types = []
    sid = []
    types = []
    exptimes = []
    tstarts = []
    tfinals = []
    tpriors = []
    slews = []
    sr = []
    sv = []

    for t in hub.obs_targets:
        for i in range(len(t.obs_times)):
            time.append(t.obs_times[i])
            tid.append(t.id)
            types.append(t.type)
            sid.append(t.sensor_ids[i])
            exptimes.append(t.exptimes[i])
            tpriors.append(t.priorities[i])
            slews.append(t.slew_times[i])
            sr.append(t.sensor_r[i])
            sv.append(t.sensor_v[i])

    sr = np.array(sr)
    sv = np.array(sv)

    

    data = {}
    data["time"] = time
    data["tid"] = tid
    data['type'] = types
    data["sid"] = sid
    data["exptime"] = exptimes
    data["priority"] = tpriors
    data['slew_time'] = slews
    data['sensor_x'] = sr[:,:,0].flatten()
    data['sensor_y'] = sr[:,:,1].flatten()
    data['sensor_z'] = sr[:,:,2].flatten()
    data['sensor_vx'] = sv[:,:,0].flatten()
    data['sensor_vy'] = sv[:,:,1].flatten()
    data['sensor_vz'] = sv[:,:,2].flatten()

    table = Table(data)
    table.write(f"{outdir}/{outname}_obs.csv", overwrite=True)

    tid = []
    types = []
    tras = []
    tdecs = []
    tstarts = []
    tends = []
    for t in hub.obs_targets:
        tid.append(t.id)
        types.append(t.type)
        tras.append(t.ra)
        tdecs.append(t.dec)
        tstarts.append(t.tstart)
        tends.append(t.tfinal)

    data_t = {}
    data_t["id"] = tid
    data_t["ra"] = tras
    data_t["dec"] = tdecs
    data_t["type"] = types
    data_t["start"] = tstarts
    data_t["end"] = tends
    table_t = Table(data_t)
    table_t.write(f"{outdir}/{outname}_target.csv", overwrite=True)

