import logging
import numpy as np
import astropy.units as u

from astropy.time import Time
from abc import ABC, abstractmethod


class Manager(ABC):
    """ABC class for how to assign pointings to a list of targets"""

    def __init__(self):
        self.sensor_history = []
        self.target_history = []
        self.time_history = []
        self.pointing_history = []
        self.no_pointing_time = []
        self.no_pointing_sensor = []


    @abstractmethod
    def assign_pointings(self, obs, sensor, time, max_pointings, mode):
        """Assign pointing(s) to sensor based on a set of times

        Parameters
        ----------
        obs : list of Targets
            set of known observations
        sensor : Sensor
            sensor to assign pointings
        times : astropy.Time
            current time in the simulation
        max_pointings : int
            maximum number of pointings to assign
        mode : str either "orbit" or "tracking"
            defines how to return the pointing. If mode=orbit then return an orbit object
            otherwise return an ra,dec location
        """
        return ValueError("Not implemented for base class")


def target_comp(target):
    """Custom comparison to sort by weight and last time observed or number of observations"""
    last_time = None

    if len(target.obs_times) == 0:
        last_time = 0
    else:
        last_time = len(target.obs_times)

    return target.current_priority, last_time


class LastObserved(Manager):
    """Assign new pointings based on the targets with the most distant observation in time"""

    def assign_pointings(self, obs, sensor, time, max_pointings=1):

        times = {}
        min_exptime = np.inf
        active_obs = []
        for ob in obs:
            if sensor.env.now < ob.tstart:
                continue

            if sensor.env.now > ob.tfinal:
                continue

            if ob.calc_exptime:
                single_exptime = ob.calc_exptime(ob, sensor)
            else:
                single_exptime = sensor.exptime
            total_exptime = sensor.n_exp * single_exptime
            if total_exptime < min_exptime:
                min_exptime = total_exptime

            times[ob] = time.gps + np.arange(
                0, total_exptime + single_exptime, single_exptime
            )
            if isinstance(ob.priority, (float, int)):
                ob.current_priority = ob.priority
            else:
                ob.current_priority = ob.priority(ob, sensor)
            active_obs.append(ob)

        sorted_obs = sorted(active_obs, key=target_comp, reverse=True)

        pointings = []
        targets = []

        for target in sorted_obs:
            if target.assigned and target.block:
                continue
            if target.lost:
                continue
            if target.max_n_obs:
                if len(target.obs_times) >= target.max_n_obs:
                    continue

            if sensor.env.now < target.tstart:
                continue

            if sensor.env.now > target.tfinal:
                continue

            vis = sensor.is_visible(target, times[target])
            if np.all(vis):
                logging.info(f"target {target.id} visible by {sensor.name}")
                obsPos, obsVel = sensor.rv(times[target])
                r, v = target.rv(times)
                ra, dec = target.ra, target.dec

                pointings.append((ra, dec, times[target]))
                target.assigned = True
                targets.append(target)

            if len(pointings) >= max_pointings:
                break

        if len(targets) > 0:
            self.sensor_history.append(sensor.name)
            self.pointing_history.append([p for p in pointings])
            self.time_history.append(time.gps)
            self.target_history.append([target.id for target in targets])
        else:
            self.no_pointing_time.append(time.gps)
            self.no_pointing_sensor.append(sensor.name)

        return pointings, targets


class Scheduler:
    """Class to schedule observations

    Parameters
    ----------
    obs_targets : list ObsTarget
        set of known target objects
    manager : Manager
        deines how jobs are scheduled

    """

    def __init__(
        self,
        manager=LastObserved(),
    ):

        self.manager = manager