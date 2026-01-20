import simpy


class Pipe(object):
    """This class represents the propagation of data through a pipe."""

    def __init__(self, env, capacity=simpy.core.Infinity):
        self.env = env
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=self.capacity)

    def latency(self, data, delay):
        yield self.env.timeout(delay)
        self.store.put(data)

    def put(self, data, delay):
        return self.env.process(self.latency(data, delay))

    def get(self):
        return self.store.get()


class BroadcastPipe(object):
    """A Broadcast pipe that allows one process to send messages to many.

    This construct is useful when message consumers are running at
    different rates than message generators and provides an event
    buffering to the consuming processes.

    The parameters are used to create a new
    :class:`~simpy.resources.store.Store` instance each time
    :meth:`get_output_conn()` is called.

    """

    def __init__(self, env, capacity=simpy.core.Infinity):
        self.env = env
        self.capacity = capacity
        self.pipes = []

    def put(self, data, delay):
        """Broadcast a *value* to all receivers, with delay"""
        if not self.pipes:
            raise RuntimeError("There are no output pipes.")

        if isinstance(delay, list) is False:
            delay = [delay for pipe in self.pipes]

        events = [pipe.put(data, d) for pipe, d in zip(self.pipes, delay)]
        return self.env.all_of(events)  # Condition event for all "events"

    def get_output_conn(self):
        """Get a new output connection for this broadcast pipe.

        The return value is a :class:`~simpy.resources.store.Store`.

        """
        pipe = Pipe(self.env, capacity=self.capacity)
        self.pipes.append(pipe)
        return pipe
