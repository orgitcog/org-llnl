import ssapy
import logging
import simpy
import numpy as np
import astropy.coordinates as ac
import astropy.units as u

from abc import ABC, abstractmethod
from astropy.time import Time
from ssapy.constants import WGS84_EARTH_RADIUS
from astropy.coordinates import SkyCoord

from .transmission import Pipe
from .utils import (
    limb_angle,
    celestial_separation_angle,
)

from collections.abc import Iterable


def iterable(obj):
    return isinstance(obj, Iterable)


class Sensor(ABC):
    """Observer base class.

    Takes a dict with defined parameters used to compute S/N of observations.


    Parameters
    ----------
    env : simpy Environment
        Environment for discrete event simulation
    params: dict
        telescope properties
    request_pipe: simpy Pipe
        data connection to request new targets from hub
    data_pipe: simpy Pipe
        data connection to send data to hub

    """

    def __init__(
        self,
        env=None,
        params={},
        request_pipe=None,
        data_pipe=None,
        name="sensor",
        random_start=100,
        min_wait=10,
        rng=None,
        record_obs_at_start=True,
        dead_fraction_per_day=None,
    ):

        self.processing_time = params.get("processing_time", 0)
        self.n_exp = params.get("n_exp")
        self.exptime = params.get("exptime")  # units seconds
        self.record_obs_at_start = record_obs_at_start
        self.no_obs = []
        self.dead_fraction_per_day = dead_fraction_per_day
        self.dead_times = None
        self.rng = rng

        # TODO Not sure if these delays are doing the correct thing
        self.request_delay = params.get("request_delay", 0)  # units seconds
        self.data_delay = params.get("data_delay", 0)  # units seconds

        self.name = name
        self.t0 = None
        self.targets = None
        self.random_start = random_start
        self.min_wait = min_wait
        self.env = env
        self.data_pipe = data_pipe

        if self.env:
            self.recieve_pipe = Pipe(env)
        else:
            self.recieve_pipe = None
        self.request_pipe = request_pipe

        if env:
            self.request_target = self.env.process(self.request_fixed_target())
        else:
            self.request_target = None

    def __getstate__(self):
        """Remove references to simpy for pickling"""
        self_dict = self.__dict__.copy()
        del self_dict["env"]
        del self_dict["recieve_pipe"]
        del self_dict["data_pipe"]
        del self_dict["request_pipe"]
        del self_dict["request_target"]

        return self_dict

    def calculate_dead_times(self):

        """Set dead time for sensor in simulation"""

        self.dead_times = []

        if self.dead_fraction_per_day:
            ndays = int(np.ceil(self.env.runtime // 86400))
            for i in range(ndays):
                if self.rng:
                    times = self.rng.uniform(i * 86400, (i + 1) * 86400)
                else:
                    times = np.random.uniform(i * 86400, (i + 1) * 86400)
                self.dead_times.append(times)
        self.active_dead_times = self.dead_times

    @abstractmethod
    def is_visible(self, target, time):
        return ValueError("Not implemented for base class")

    def request_fixed_target(self):
        """Request a new target from hub and return data"""
        if self.random_start > 0:
            if self.rng:
                wait = self.rng.integers(self.random_start)
            else:
                wait = np.random.randint(self.random_start)

            yield self.env.timeout(wait)
            logging.debug(f"Waited {wait} to start requesting targets")

        self.calculate_dead_times()

        while True:
            # Check if we need to stop for dead time
            if len(self.active_dead_times) > 0:
                if self.env.now > self.active_dead_times[0]:
                    logging.debug(
                        f"Dead time for {self.dead_fraction_per_day * 86400} at {self.env.now}"
                    )
                    yield self.env.timeout(self.dead_fraction_per_day * 86400)
                    self.active_dead_times.pop(0)

            # Request new observation
            self.request_pipe.put(
                (self.name, f"data request", None), self.request_delay
            )

            # wait to receive observations
            (msg, pointings, obs_targets) = yield self.recieve_pipe.get()

            if len(pointings) == 0:
                logging.info(f"no objects visible for {self.name}")
                self.no_obs.append(self.env.now)
                yield self.env.timeout(self.min_wait)
                continue

            # assume the scheduler will return orbits
            mode = "fixed_target"
            tracks = []
            try:
                out_data = []
                current_obs_ids = np.array([target.id for target in obs_targets])
                for pointing, obs_target in zip(pointings, obs_targets):
                    ra, dec, times = pointing
                    duration = times[1] - times[0]
                    logging.info(
                        f"{self.name} got target {obs_target.id} at {self.env.now} for {duration} sec"
                    )

                    # if space sensor compute distance between current pointing and assigned pointing
                    if type(self) is SpaceSensor:
                        c1 = SkyCoord(self.current_ra, self.current_dec, unit="rad")
                        c2 = SkyCoord(ra, dec, unit="rad")
                        ang_sep = c1.separation(c2).value
                        slew_time = ang_sep / self.slew_speed
                        logging.debug(
                            f"Need to slew {ang_sep} deg, for {slew_time} seconds"
                        )
                        yield self.env.timeout(slew_time)

                    ctime = self.env.t0 + self.env.now * u.s
                    r, v = self.rv(ctime.gps)

                    if self.record_obs_at_start:
                        data = self.getFixedPointing(
                            obs_target,
                            ra,
                            dec,
                            times,
                            duration * u.s,
                            r,
                            v,
                            slew_time,
                        )
                        yield self.env.timeout(duration)
                    else:
                        yield self.env.timeout(duration)
                        data = self.getFixedPointing(
                            obs_target,
                            ra,
                            dec,
                            times,
                            duration * u.s,
                            r,
                            v,
                            slew_time,
                        )

                    self.current_ra = ra
                    self.current_dec = dec
                    obs_target.assigned = False
                    logging.debug(f"Processing time {self.processing_time}")
                    yield self.env.timeout(self.processing_time)

                    self.data_pipe.put(
                        (self.name, f"Sending update at {self.env.now}", data),
                        self.data_delay,
                    )

            except simpy.Interrupt:
                logging.info(f"Observations interupted at {self.env.now}")

    def getFixedPointing(
        self, target, ra, dec, t0, exptime, r, v, slew_time
    ):

    """Adds target for simulated observation.

    Args:
      target: target from hub
      ra: right ascension of target
      dec: declination of target 
      t0: current time
      exptime: exposure time in seconds
      r: observer position vector
      v: observer velocity vector
      slew_time: time delay for slew time 

    Returns:
      None, updated target 

    """

        target.add_data(
            t0,
            self.name,
            exptime.value,
            target.current_priority,
            r,
            v,
            slew_time,
        )

        return None


class SpaceSensor(Sensor):
    """Observer on an orbit orbit"""

    def __init__(
        self,
        params={},
        orbit=None,
        solar_exclusion=None,
        lunar_exclusion=None,
        limb_angle=None,
        slew_speed=None,
        propagator=ssapy.KeplerianPropagator(),
        **kwargs,
    ):
        Sensor.__init__(self, params=params, **kwargs)

        if orbit is None:
            # This is keplerian
            if isinstance(params["orbit"], list):
                vals = params["orbit"]

                try:
                    time = Time(float(vals[6]), format="gps")
                except ValueError:
                    time = Time(vals[6])

                self.orbit = ssapy.Orbit.fromKeplerianElements(
                    float(vals[0]),
                    float(vals[1]),
                    float(vals[2]),
                    float(vals[3]),
                    float(vals[4]),
                    float(vals[5]),
                    time,
                )

            elif isinstance(params["orbit"], str):
                # TLE case
                if len(params["orbit"].split("\n")) == 3:
                    self.orbit = ssapy.Orbit.fromTLETuple(
                        params["orbit"].split("\n")[:2]
                    )
                else:
                    raise ValueError(f"Invalid orbit: {orbit}")

            else:
                raise ValueError(f"Invalid orbit: {orbit}")
        else:
            self.orbit = orbit
        self.propagator = propagator
        self.observer = ssapy.OrbitalObserver(self.orbit, self.propagator)

        if solar_exclusion is None:
            self.solar_exclusion = params.get("solar_exclusion")
        else:
            self.solar_exclusion = solar_exclusion

        # Add moon exclusion angle
        if lunar_exclusion is None:
            self.lunar_exclusion = params.get("lunar_exclusion")
        else:
            self.lunar_exclusion = lunar_exclusion

        if limb_angle is None:
            self.limb_angle = params.get("limb_angle")
        else:
            self.limb_angle = limb_angle
        if slew_speed is None:
            self.slew_speed = params.get("slew_speed")
        else:
            self.slew_speed = slew_speed

        self.current_ra = 0
        self.current_dec = 0
        self.cache_rv = {}
        self.max_cache_size = 1000

    def rv(self, time):

        """Gets position and velocity of observer at a given time.

        Args:
          time: gps seconds  

        Returns:
          r, observer position vector
          v, observer velocity vector

        """
        r = []
        v = []
        if iterable(time) is False:
            time = [time]

        for t in time:
            if t in self.cache_rv:
                rr, vv = self.cache_rv[t]
                r.append(rr)
                v.append(vv)
            else:
                rr, vv = self.observer.getRV(t)
                if len(self.cache_rv) >= self.max_cache_size:
                    self.cache_rv.pop(
                        next(iter(self.cache_rv))
                    )  # Remove the oldest entry
                self.cache_rv[t] = (rr, vv)
                r.append(rr)
                v.append(vv)
        r = np.array(r)
        v = np.array(v)
        return r, v

    def is_visible(self, target, time):
        """Checks if target is visible from observer position given earth, moon, and solar 
           observing constraints.

        Args:
          target: target orbit object
          time: gps seconds  

        Returns:
          visible bool

        """
        r_target, _ = target.rv(time)
        r, _ = self.rv(time)

        # Calculate solar angle
        solar_angle = celestial_separation_angle("sun", r, r_target, time, env=self.env)

        # Calculate moon angle
        moon_angle = celestial_separation_angle("moon", r, r_target, time, env=self.env)

        # Calculate limb angle
        l_angle = limb_angle(r, r_target)

        # Check all visibility conditions
        visible = (
            (solar_angle > self.solar_exclusion)
            & (moon_angle > self.lunar_exclusion)
            & (l_angle > self.limb_angle)
        )

        logging.debug(
            f"visible: {visible} -- solar: {(solar_angle > self.solar_exclusion)}, "
            f"moon: {(moon_angle > self.lunar_exclusion)}, limb: {l_angle > self.limb_angle}"
        )

        return visible
