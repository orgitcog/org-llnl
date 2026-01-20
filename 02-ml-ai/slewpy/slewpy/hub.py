import simpy
from astropy.time import Time
import astropy.units as u
import logging


class Hub:
    """Central hub for satellite communications

    Parameters
    ----------
    env : simpy Environment
        Discrete event environment
    request_pipe : simpy Pipe
        channel for sensors to request new targets
    data_pipe : simpy Pipe
        channel to receive observations from sensors
    sensor_list : list of Sensor objects
        set of sensors to communicate with
    targets : list of target objects
        this contains the true orbit information for all targets
    obs_targets : list of ObsTarget
        list of currently known Targets
    scheduler : Scheduler
        object that determines how targets are assigned to sensors
    t0 : astropy Time
        time at which the simulation starts
    """

    def __init__(
        self,
        env,
        request_pipe,
        data_pipe,
        sensor_list,
        obs_targets,
        target_pipe=None,
        scheduler=None,
        t0=Time("J2000"),
        runtime=None,
    ):
        self.env = env
        self.request_pipe = request_pipe
        self.data_pipe = data_pipe
        self.target_pipe = target_pipe
        self.targets = []
        self.obs_targets = obs_targets
        self.t0 = t0
        self.env.t0 = t0
        if runtime:
            self.env.runtime = runtime
        self.scheduler = scheduler

        self.env.process(self.listen_request())
        self.env.process(self.recieve_target_info())
        if self.target_pipe:
            self.env.process(self.listen_new_target())

        self.sensor_list = {}
        for sensor in sensor_list:
            self.sensor_list[sensor.name] = sensor
            self.sensor_list[sensor.name].t0 = t0
            self.sensor_list[sensor.name].targets = self.targets

    def send_target(self, sensor, pointings, obs_targets):
        """Send sensor a set of pointings and current set of observed targets"""
        delay = 0
        sensor.recieve_pipe.put((f"Sending target ", pointings, obs_targets), delay)

    def listen_new_target(self):
        """Listen for sensor to request a new set of pointings"""
        while True:
            name, msg, target = yield self.target_pipe.get()

            logging.info(f"New target added at {self.env.now}")
            self.obs_targets.append(target)

    def listen_request(self):
        """Listen for sensor to request a new set of pointings"""
        while True:
            name, msg, data = yield self.request_pipe.get()

            logging.info(
                f"Received request at {self.env.now} from sensor {name}:  {msg}"
            )

            pointings, obs_targets = self.scheduler.manager.assign_pointings(
                self.obs_targets, self.sensor_list[name], self.t0 + self.env.now * u.s
            )

            self.send_target(self.sensor_list[name], pointings, obs_targets)

    def recieve_target_info(self):
        """Listen for sensor to send updated track information"""
        while True:
            name, msg, data = yield self.data_pipe.get()

