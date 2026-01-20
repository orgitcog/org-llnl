"""
The priority function for each target is called at each decision point.
A good priority rule is that a target's priority value is

    priority = -(desired time left until target's next observation),

"Desired" means that this would be the ideal time if sensor availability
and observing geometry weren't a concern.

A priority function should have the signature
    priorityfunc(target, sensor, **kwargs)
and return a single numerical value.
"""


def priority_fixed(target, sensor, **kwargs):
    """Calculate priority for fixed time interval between observations.

    Aims for a fixed cadence (e.g. 6 hours) between successive observations of a target.

    Args:
        target (ObsFixedTarget): The observation target.
        sensor (Sensor): The sensor object providing environment context.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        float: Target's priority, -(desired time until the next observation)
    """
    interval = 6.0 * 3600  # Ideal cadence in seconds

    now = sensor.env.now  # Current time in seconds since simulation start
    obs_so_far = len(target.obs_times)
    
    # First observation should be as soon after beginning of target's tstart as possible
    if obs_so_far == 0:
        time_til_next = target.tstart - now
        return -time_til_next
    
    t_last = target.obs_times[-1] - sensor.env.t0.gps  # Most recent observation, relative to simulation start
    t_next = t_last + interval  # Desired time of next observation
    time_til_next = t_next - now
    return -time_til_next
    

def priority_twophase(target, sensor, **kwargs):
    """Calculate priority for a cadence that changes after a given time.

    E.g. observe every hour for first 3 days, then every 12 hours.
    
    Args:
        target (ObsFixedTarget): The observation target.
        sensor (Sensor): The sensor object providing environment context.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        float: Target's priority, -(desired time until the next observation)
    """
    now = sensor.env.now  # Current time in seconds since simulation start
    obs_so_far = len(target.obs_times)
    
    # First observation should be as soon after beginning of target's tstart as possible
    if obs_so_far == 0:
        time_til_next = target.tstart - now
        return -time_til_next
        
    t_last = target.obs_times[-1] - sensor.env.t0.gps  # Most recent observation, relative to simulation start
    if now - target.tstart < 3*86400: 
        interval = 1.0 * 3600 # Within the first 3 days of window observe every hour
    else:
        interval = 12.0 * 3600 # After 3 days observe every 12 hrs
    time_til_next = t_last + interval - now
    return -time_til_next
    

def priority_adaptive(target, sensor, **kwargs):
    """Calculate priority that can adapt cadence to achieve a given number
    of observations in the target's observing window.

    E.g. aim to observe every 6 hours but try to ensure at least 
     200 observations in 50 days.
    
    Args:
        target (ObsFixedTarget): The observation target.
        sensor (Sensor): The sensor object providing environment context.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        float: Target's priority, -(desired time until the next observation)
    """
    now = sensor.env.now  # Current time in seconds since simulation start
    obs_so_far = len(target.obs_times)
    
    # First observation should be as soon after beginning of target's tstart as possible
    if obs_so_far == 0:
        time_til_next = target.tstart - now
        return -time_til_next
        
    t_last = target.obs_times[-1] - sensor.env.t0.gps  # Most recent observation, relative to simulation start
    t_stop = target.tstart + 50.0 * 86400 # End of 50 days
    
    base_interval = 6.0 * 3600 # Basic cadence is every 6 hrs
    adaptive_interval = (t_stop - t_last)/(200 - obs_so_far) # Cadence needed to get 200 total observations
    interval = min(base_interval, adaptive_interval)
    
    time_til_next = t_last + interval - now
    return -time_til_next


