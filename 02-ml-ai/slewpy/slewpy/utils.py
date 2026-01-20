import numpy as np
from ssapy import rv
from ssapy.utils import norm, normed, unitAngle3
from astropy.time import Time
from ssapy.constants import WGS84_EARTH_RADIUS
import astropy.coordinates as ac
import astropy.units as u
from astropy.coordinates import get_sun, get_moon
import logging
from scipy.interpolate import CubicSpline
from collections.abc import Iterable


class CelestialBodySpline:
    """Efficient position interpolation for celestial bodies using cubic splines.

    This class creates and manages cubic spline interpolations for celestial body
    positions over a time range, providing fast position lookups with minimal
    computational overhead.
    """

    # Registry of position calculators by body name
    _position_calculators = {
        "sun": lambda times: get_sun(times).cartesian.get_xyz().to(u.m).value.T,
        "moon": lambda times: get_moon(times).cartesian.get_xyz().to(u.m).value.T,
    }

    # Store singleton instances (one per body type)
    _instances = {}

    @classmethod
    def get_instance(cls, body_name, start_time=None, end_time=None, num_points=10000):
        """Get or create the singleton instance for a specific celestial body.

        Parameters
        ----------
        body_name : str
            Name of the celestial body ('sun' or 'moon')
        start_time : Time or float, optional
            Start time as astropy Time or GPS seconds
        end_time : Time or float, optional
            End time as astropy Time or GPS seconds
        num_points : int, optional
            Number of interpolation points

        Returns
        -------
        CelestialBodySpline
            Instance for the requested body

        Raises
        ------
        ValueError
            If body_name is not a supported celestial body
        """
        body_name = body_name.lower()

        if body_name not in cls._position_calculators:
            valid_bodies = ", ".join(cls._position_calculators.keys())
            raise ValueError(
                f"Unknown celestial body '{body_name}'. Available bodies: {valid_bodies}"
            )

        # Create new instance if it doesn't exist
        if body_name not in cls._instances:
            cls._instances[body_name] = cls(body_name)

        instance = cls._instances[body_name]

        # Initialize/reinitialize if time range provided
        if start_time is not None and end_time is not None:
            instance.initialize(start_time, end_time, num_points)

        return instance

    def __init__(self, body_name):
        """Initialize a celestial body spline interpolator.

        Parameters
        ----------
        body_name : str
            Name of the celestial body ('sun' or 'moon')

        Raises
        ------
        ValueError
            If body_name is not a supported celestial body
        """
        self.body_name = body_name.lower()

        if self.body_name not in self._position_calculators:
            valid_bodies = ", ".join(self._position_calculators.keys())
            raise ValueError(
                f"Unknown celestial body '{self.body_name}'. Available bodies: {valid_bodies}"
            )

        self.position_calculator = self._position_calculators[self.body_name]

        # Initialize state
        self.start_time = None
        self.end_time = None
        self.num_points = None
        self.x_spline = None
        self.y_spline = None
        self.z_spline = None

    def initialize(self, start_time, end_time, num_points):
        """Initialize or update the position spline for a time interval.

        Parameters
        ----------
        start_time : Time or float
            Start time as astropy Time or GPS seconds
        end_time : Time or float
            End time as astropy Time or GPS seconds
        num_points : int, optional
            Number of interpolation points

        Raises
        ------
        ValueError
            If end_time <= start_time
        """
        # Convert times to GPS seconds if needed
        if isinstance(start_time, Time):
            start_time_gps = start_time.gps
        else:
            start_time_gps = float(start_time)

        if isinstance(end_time, Time):
            end_time_gps = end_time.gps
        else:
            end_time_gps = float(end_time)

        # Validate time range
        if end_time_gps <= start_time_gps:
            raise ValueError(
                f"End time ({end_time_gps}) must be greater than start time ({start_time_gps})"
            )

        # Store parameters
        self.start_time = start_time_gps
        self.end_time = end_time_gps
        self.num_points = num_points

        logging.debug(
            f"Initializing {self.body_name.capitalize()} spline with {num_points} points"
        )

        # Generate evenly spaced times
        times_gps = np.linspace(start_time_gps, end_time_gps, num_points)

        # Convert to astropy Time
        times = Time(times_gps, format="gps")

        try:
            # Calculate positions
            positions = self.position_calculator(times)

            # Extract coordinates
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]

            # Create time parameter for spline (relative to start)
            t = times_gps - start_time_gps

            # Create splines
            self.x_spline = CubicSpline(t, x)
            self.y_spline = CubicSpline(t, y)
            self.z_spline = CubicSpline(t, z)

            logging.debug(
                f"{self.body_name.capitalize()} spline initialized: {start_time_gps} to {end_time_gps}"
            )

        except Exception as e:
            logging.error(f"Failed to initialize {self.body_name} spline: {str(e)}")
            raise

    def get_position(self, query_time):
        """Get interpolated position(s) at the requested time(s).

        Parameters
        ----------
        query_time : Time, float, or array
            Query time(s) as astropy Time, GPS seconds, or array of times

        Returns
        -------
        np.ndarray
            Position array with shape (3,) for scalar input or (n, 3) for array input

        Raises
        ------
        RuntimeError
            If spline is not initialized
        ValueError
            If query_time is outside the initialized time range
        """
        if not self.is_initialized():
            raise RuntimeError(
                f"{self.body_name.capitalize()} spline not initialized. Call initialize() first."
            )

        # Convert to GPS seconds
        if isinstance(query_time, Time):
            query_time_gps = query_time.gps
        else:
            query_time_gps = query_time

        # Handle scalar input
        scalar_input = np.isscalar(query_time_gps) or (
            isinstance(query_time_gps, Time) and query_time_gps.isscalar
        )
        if scalar_input:
            query_time_gps = np.array([query_time_gps])
        else:
            query_time_gps = np.asarray(query_time_gps)

        # Check range
        if np.any(query_time_gps < self.start_time) or np.any(
            query_time_gps > self.end_time
        ):
            raise ValueError(
                f"Query time outside spline range [{self.start_time}, {self.end_time}]"
            )

        # Calculate relative times
        t = query_time_gps - self.start_time

        # Get interpolated coordinates
        x = self.x_spline(t)
        y = self.y_spline(t)
        z = self.z_spline(t)

        # Combine coordinates
        positions = np.column_stack((x, y, z))

        # Return appropriate shape based on input
        return positions[0] if scalar_input else positions

    def is_initialized(self):
        """Check if the spline is initialized.

        Returns
        -------
        bool
            True if spline is initialized, False otherwise
        """
        return (
            self.x_spline is not None
            and self.y_spline is not None
            and self.z_spline is not None
        )


def celestial_separation_angle(body_name, obs_pos, target_pos, time, env=None):
    """Calculate the angular separation between a target and a celestial body as seen from an observer.

    Parameters
    ----------
    body_name : str
        Name of the celestial body ('sun' or 'moon')
    obs_pos : ndarray
        Observer position in meters
    target_pos : ndarray
        Target position in meters
    time : Time or float
        Time of observation
    env : Environment, optional
        Simulation environment

    Returns
    -------
    float
        Angular separation in degrees
    """
    # Get celestial body position
    body = CelestialBodySpline.get_instance(body_name)

    try:
        # Get position from spline
        body_pos = body.get_position(time)
    except (ValueError, RuntimeError):
        # Fall back to direct calculation
        if isinstance(time, Time):
            t = time
        else:
            t = Time(time, format="gps")

        if body_name.lower() == "sun":
            body_pos = get_sun(t).cartesian.get_xyz().to(u.m).value.T
        elif body_name.lower() == "moon":
            body_pos = get_moon(t).cartesian.get_xyz().to(u.m).value.T
        else:
            raise ValueError(f"Unknown celestial body: {body_name}")

    # Calculate vectors from observer
    body_vector = body_pos - obs_pos
    target_vector = target_pos - obs_pos

    # Handle different array shapes
    if len(body_vector.shape) > 1:
        # For arrays, calculate norms along the right axis
        body_norm = np.linalg.norm(body_vector, axis=1)
        target_norm = np.linalg.norm(target_vector, axis=1)

        # Reshape for broadcasting
        body_unit = body_vector / body_norm[:, np.newaxis]
        target_unit = target_vector / target_norm[:, np.newaxis]

        # Calculate dot product for each pair of vectors
        dot_product = np.sum(body_unit * target_unit, axis=1)
    else:
        # For single vectors
        body_norm = np.linalg.norm(body_vector)
        target_norm = np.linalg.norm(target_vector)

        body_unit = body_vector / body_norm
        target_unit = target_vector / target_norm

        dot_product = np.sum(body_unit * target_unit)

    # Clip to prevent numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle
    angle_rad = np.arccos(dot_product)
    return np.rad2deg(angle_rad)


def dot(x, y):
    return np.einsum("...i, ...i", x, y)


def limb_angle(obs, target, r=WGS84_EARTH_RADIUS):

    alpha = np.rad2deg(np.arcsin(r / norm(obs)))
    beta = np.rad2deg(np.arccos(dot(normed(-obs), normed(target - obs))))
    limb = beta - alpha
    return limb