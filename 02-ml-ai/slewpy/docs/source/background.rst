==========
Background
==========


Descrete Event Simulation
-------------------------

slewpy is built on top of a discrete event simulation (DES) framework implemented in `Simpy <https://simpy.readthedocs.io/en/latest/contents.html>`_ . In DES, the system state changes only at discrete points in time, triggered by events. In slewpy, examples of events are: a transient appearing in the night sky, a satellite observing a transient, or a satellite becoming available after observing a target. In DES, processes are sequences of events or actions that entities in the system perform over time. The simulation environment manages the simulation clock and event scheduling. It coordinates the execution of all processes and advances time to the next scheduled event. Resources represent limited, shared entities such as satellites. Processes request and release resources, and may have to wait if the resource is unavailable.

One of the key advantages of using DES is that it only simulates changes at event times, skipping periods where nothing happens. This makes it much more efficient than time-step simulations for many systems. DES allows the study of system bottlenecks, resource utilization, and the impact of different system policies or configurations.


Priority Functions
------------------

Optimal scheduling of observations is a complex planning problem that requires a
joint analysis of all targets and the future time series of sensor positions.

Slewpy finds a good approximate solution using an independent priority function for each target.  At each decision point, 
each target's priority function returns a numerical priority value for the target, with no regard for the other 
targets or sensor positions. The scheduler then selects the highest priority target that has a visible viewing geometry.

The priority function can return a value on an arbitrary scale. However, we have found that a rational way to
assign priority is according to a target's *ideal time to its next observation*. For example, if a target's
desired cadence is :math:`T` (say :math:`T = 6~\mathrm{hr}`) and it was last observed at time
:math:`t_\mathrm{last}`, its ideal time of next observation would be :math:`t = t_\mathrm{last} + T` and 
its priority should be

.. math::

   \mathrm{priority} = -\left[ (t_\mathrm{last} + T) - t_\mathrm{cur} \right],

where :math:`t_\mathrm{cur}` is the current time. The overall minus sign means that a target that should be observed sooner
gets a higher priority. A negative priority is the amount of slack before the ideal next observation, while a
positive priority indicates that the target is "overdue" for an observation.

We recommend sticking to a physical definition of priority (e.g. time until next observation). It is tempting to boost 
the priority of an important target by hand (e.g. by adding 1e10 to its priority if you *really* want it to get
observed now) but this can quickly lead to a tower of ad hoc fixes that are hard to manage.
