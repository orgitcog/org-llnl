import ssapy
import numpy as np
import logging
from ssapy.utils import lb_to_unit
from astropy.time import Time


class ObsFixedTarget:
    """Observed fixed location

    Parameters
    ----------
    _id : int
        orbit id
    name : str
        name of target
    """

    def __init__(
        self,
        ra,
        dec,
        _id=None,
        name="",
        true_target=None,
        priority=None,
        dist=14959787070000,  # 100AU in m
        type=None,
        calc_exptime=None,
        max_n_obs=None,
        tstart=-np.inf,
        tfinal=np.inf,
        block=True,
        obs_per_period=None,
        obs_period=None,
    ):
        self.ra = ra
        self.dec = dec
        self.id = _id
        self.name = name
        self.type = type
        self.calc_exptime = calc_exptime
        self.max_n_obs = max_n_obs
        self.tstart = tstart
        self.tfinal = tfinal
        self.block = block
        self.obs_per_period = obs_per_period
        self.obs_period = obs_period
        self.current_priority = 0

        # Set of times target was observed
        self.obs_times = []
        self.mag_lims = []
        self.sensor_ids = []
        self.exptimes = []
        self.priorities = []
        self.slew_times = []
        self.sensor_r = []
        self.sensor_v = []

        # id of link to true target
        self.true_target = true_target
        self.priority = priority
        self.dist = dist

        self.assigned = False
        self.lost = False

    def rv(self, times, mode="equinoctial"):
        """Get position and velocity at specified times

        Parameters
        ----------
        times: list of Astropy.Time or float
            set of times

        Returns
        -------
        r: array of float
            Position in meters
        v: array of float
            Velocity in meters per second
        """
        r_hat = lb_to_unit(self.ra, self.dec)
        return np.array([r_hat * self.dist] * len(times)), np.array(
            [0, 0, 0] * len(times)
        )

    def add_data(self, time, sensor_id, exptime, priority, r, v, slew_time):
        self.obs_times.append(time[0])
        self.sensor_ids.append(sensor_id)
        self.exptimes.append(exptime)
        self.priorities.append(priority)
        self.slew_times.append(slew_time)
        self.sensor_r.append(r)
        self.sensor_v.append(v)