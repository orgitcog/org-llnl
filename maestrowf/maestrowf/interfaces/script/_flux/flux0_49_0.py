import logging
from math import ceil
import os

from maestrowf.abstracts.enums import (
    CancelCode,
    JobStatusCode,
    State,
    StepPriority,
    SubmissionCode,
)
from maestrowf.abstracts.interfaces.flux import FluxInterface
from maestrowf.utils import (
    dict_to_dot_strings,
    iter_dotpath_items,
    unflatten_dotpath_dict,
    coerce_dict_values,
    update_recursive
)

from rich.pretty import pprint
LOGGER = logging.getLogger(__name__)

try:
    import flux
except ImportError:
    LOGGER.info("Failed to import Flux. Continuing.")


class FluxInterface_0490(FluxInterface):
    # This utility class is for Flux 0.49.0
    key = "0.49.0"

    flux_handle = None
    _urgencies = {
        StepPriority.HELD: 0,
        StepPriority.MINIMAL: 1,
        StepPriority.LOW: 9,
        StepPriority.MEDIUM: 16,
        StepPriority.HIGH: 24,
        StepPriority.EXPEDITE: 31,
    }

    # Config for the groups of alloc args with their own jobspec methods
    known_alloc_arg_types = ["attributes", "shell_options", "conf"]
    _addtl_alloc_arg_type_map = {
        "setopt": "shell_options",
        "o": "shell_options",
        "setattr": "attributes",
        "S": "attributes",
        "conf": "conf",
    }
    _addtl_arg_cli_key = {
        "attributes": "setattr",
        "shell_options": "setopt",
        "conf": "conf"
    }

    @classmethod
    def addtl_alloc_arg_types(cls):
        """
        Return set of additional allocation args that this adapter knows how
        to wire up to the jobspec python apis, e.g. 'attributes',
        'shell_options', ... This is aimed specifically at the repeated types,
        which collect many flags/key=value pairs which go through a specific
        jobspec call.  Everything not here gets dumped into a 'misc' group
        for individual handling.

        :return: List of string

        .. note::

           Should we have an enum for these or something vs random strings?
        """
        return cls.known_alloc_arg_types

    @classmethod
    def addtl_alloc_arg_type_map(cls, option):
        """
        Map verbose/brief cli arg option name (o from -o, setopt from --setopt)
        onto known alloc arg types this interface implements

        :param option: option string corresponding to flux cli input
        :return: string, one of known_alloc_arg_types
        """
        return cls._addtl_alloc_arg_type_map.get(option, None)

    @classmethod
    def get_addtl_arg_cli_key(cls, arg_type):
        """
        Return expected cli key associated with each normalized arg type.
        `arg_type` not in known_arg_types are assumed to be the key already
        to facilitate flexible pass through to launcher

        :param arg_type: string noting arg group or cli key
        :returns: cli key used for this arg

        .. note::

           Can we find a reasonable default prefix (where are things put
           by default in flux, attributes.system?)
        """
        if arg_type in cls.known_alloc_arg_types:
            return cls._addtl_arg_cli_key.get(arg_type)

        elif arg_type in cls._addtl_alloc_arg_type_map:
            return cls._addtl_arg_cli_key.get(cls._addtl_alloc_arg_type_map[arg_type])

        else:
            # NOTE: can't log unknonws here -> this func doesn't know if which
            # group of args it's processing (alloc or launcher)
            # Defer to alloc normalization to pre-clean the alloc group
            return arg_type

    @classmethod
    def get_flux_urgency(cls, urgency) -> int:
        if isinstance(urgency, str):
            LOGGER.debug("Found string urgency: %s", urgency)
            urgency = StepPriority.from_str(urgency)

        if isinstance(urgency, StepPriority):
            LOGGER.debug("StepUrgency urgency of '%s' given..", urgency)
            return cls._urgencies[urgency]
        else:
            LOGGER.debug("Float urgency of '%s' given..", urgency)
            return ceil(float(urgency) * 31)

    @classmethod
    def render_additional_args(cls, args_dict):
        """
        Helper to render additional argument sets to flux cli format for
        use in constructing $(LAUNCHER) line and flux batch directives.
        This includes coercing bools to json's lowercase str forms and
        omitting values for flag types (null/none via maestro's yaml)

        :param args_dict: Dictionary of flux arg keys and name: value pairs
        :yield: formatted strings of cli options/values

        .. note::

           Promote this to the general/base adapters to handle non-normalizable
           scheduler/machine specific options
        """
        # Normalize to string form that json expects for bools:
        # flux expects json's lowercase 'true' not 'True' from python bool
        coerced_vals = coerce_dict_values(
            args_dict,
            lambda x: str(x).lower() if isinstance(x, bool) else x
        )

        base_render_tmpl = "{prefix}{cli_key}{sep}{dotpath}"
        value_render_tmpl = "={value}"
        for arg_key, arg_value in coerced_vals.items():
            # Get the cli key and associated rendering info
            cli_key = cls.get_addtl_arg_cli_key(arg_key)
            cli_info = cls.get_cli_arg_prefix_sep(cli_key)

            # Note: dotpath encoding comes after the group/prefix (setattr, ...)
            for dotpath, value in iter_dotpath_items(arg_value):
                rendered_opt = base_render_tmpl.format(
                    prefix=cli_info['prefix'],
                    cli_key=cli_key,
                    sep=cli_info['sep'],
                    dotpath=dotpath
                )

                # Flag types have None, and we want to exclude it from launcher
                # and headers to better match interactive use omitting '=None'
                if value:
                    rendered_opt += value_render_tmpl.format(value=value)

                yield rendered_opt

    @classmethod
    def normalize_additional_args(cls, args_dict, group_name=None, filter_unknown=False):
        """
        Helper to normalize additional arguments to known types and an
        unflattened nested dictionary structure.  This unflattens any
        dotpath encoded nested dictionary keys.

        :param args_dict: Dictionary of flux arg keys and name: value pairs
        :type args_dict: dict
        :param group_name: Optional name of group/tag to use in log messages
                           when filtering_unknown is on
        :type group_name: str
        :param filter_unknown: flag to block pass through of unknown args, e.g.
                               for allocation where we can't handle arbitrary
        :type filter_unknown: bool
        :return: dict of packed args with top level keys being the adapter
                 version specific addtl_alloc_arg_types
        """
        # First, normalize and unflatten everything into dicts
        unflattened_batch_args = {
            arg_type: {}
            for arg_type in cls.addtl_alloc_arg_types()
        }

        if filter_unknown:
            known_arg_prefixes = ', '.join([f"'{prefix}'" for prefix in cls._addtl_alloc_arg_type_map.keys()])
        else:
            known_arg_prefixes = ''

        for arg_key, arg_values in args_dict.items():
            arg_type = cls.addtl_alloc_arg_type_map(arg_key)
            if arg_type is None:
                if filter_unknown:
                    LOGGER.warn(
                        "Filtering '%s' in unhandled type '%s' from '%s' args."
                        "  Known types are %s",
                        arg_values,
                        arg_key,
                        str(group_name) if group_name else "UNKNOWN",
                        known_arg_prefixes
                    )
                    continue
                
                arg_type = arg_key

            if isinstance(arg_values, dict):
                unflattened_batch_arg = {arg_type: unflatten_dotpath_dict(arg_values)}
            else:
                unflattened_batch_arg = {arg_type: arg_values}

            # Update to ensure we don't clobber prior values
            unflattened_batch_args = update_recursive(unflattened_batch_args,
                                                      unflattened_batch_arg)

        return unflattened_batch_args

    @classmethod
    def pack_addtl_batch_args(cls, args_dict):
        """
        Normalize the allocation args and pack up into the interface specific
        groups that have assocated jobspec methods, e.g. conf, setattr, setopt.
        Ensure arg formats match their end point's types.  For current flux
        versions as of 0.78.0: setattr and setopt require dicts be flattened
        into lists of dotpath encoded strings and values, while conf requires
        a completely unflattened dict.

        :param args_dict: Dict with keys normalized to cls.addtl_alloc_arg_types
        :return: dictionary of allocation arg groups to attach to jobspecs
        """
        # Normalize None values common in flag inputs for use in python api
        # see https://github.com/flux-framework/flux-core/blob/a3860d4dea5b5a17c473cff4385276e882275252/src/bindings/python/flux/cli/base.py#L734
        # NOTE: only doing this in alloc; LAUNCHER cli passes through
        #       to flux cli (None values are omittied, e.g.
        #       {o: fastload: None} renders to -o fastload
        #       Python api doesn't appear to have default value handling?
        packed_batch_args = {
            arg_type: {}
            for arg_type in cls.addtl_alloc_arg_types()
        }

        dotpath_format = ["attributes", "shell_options"]

        for arg_key, arg_values in args_dict.items():
            coerced_vals = coerce_dict_values(arg_values,
                                              lambda x: 1 if x is None else x)

            if arg_key in dotpath_format:
                group_values = list(iter_dotpath_items(coerced_vals))
            else:
                group_values = coerced_vals

            packed_batch_args[arg_key] = group_values

        return packed_batch_args

    @classmethod
    def submit(
        cls,
        nodes,
        procs,
        cores_per_task,
        path,
        cwd,
        walltime,
        ngpus=0,
        job_name=None,
        force_broker=True,
        urgency=StepPriority.MEDIUM,
        waitable=False,
        queue=None,
        bank=None,
        addtl_batch_args=None,
        exclusive=False,
        **kwargs,
    ):
        # Sanitize/initialize the extra batch args
        if addtl_batch_args is None:
            addtl_batch_args = {}

            # May want to also support setattr_shell_option at some point?
            for batch_arg_type in cls.addtl_alloc_arg_types(): #["attributes", "shell_options", "conf"]:

                if batch_arg_type not in addtl_batch_args:
                    addtl_batch_args[batch_arg_type] = {}

        try:
            # TODO: add better error handling/throwing in the class func
            # to enable more uniform detection/messaging when connection fails
            # to deal with both missing uri in allocations on non-flux machines
            cls.connect_to_flux()

            # NOTE: This previously placed everything under a broker. However,
            # if there's a job that schedules items to Flux, it will schedule
            # all new jobs to the sub-broker. Sometimes this is desired, but
            # it's incorrect to make that the general case. If we are asking
            # for a single node, don't use a broker -- but introduce a flag
            # that can force a single node to run in a broker.

            # Attach any conf inputs to the jobspec
            conf_dict = addtl_batch_args.get('conf', None)

            if force_broker:
                LOGGER.debug(
                    "Launch under Flux sub-broker. [force_broker=%s, "
                    "nodes=%d]",
                    force_broker,
                    nodes,
                )
                # Need to attach broker opts to the constructor?
                # TODO: Add in extra broker options if not null
                ngpus_per_slot = int(ceil(ngpus / nodes))
                jobspec = flux.job.JobspecV1.from_nest_command(
                    [path],
                    num_nodes=nodes,
                    cores_per_slot=cores_per_task,
                    num_slots=procs,
                    gpus_per_slot=ngpus_per_slot,
                    conf=conf_dict,
                    exclusive=exclusive,
                    queue=queue,
                    bank=bank
                )
            else:
                if conf_dict:
                    LOGGER.warn("'conf' options not currently supported with "
                                " nested=False.  Ignoring.")
                LOGGER.debug(
                    "Launch under root Flux broker. [force_broker=%s, "
                    "nodes=%d]",
                    force_broker,
                    nodes,
                )
                jobspec = flux.job.JobspecV1.from_command(
                    [path],
                    num_tasks=procs,
                    num_nodes=nodes,
                    cores_per_task=cores_per_task,
                    gpus_per_task=ngpus,
                    exclusive=exclusive,
                    queue=queue,
                    bank=bank
                )

            LOGGER.debug("Handle address -- %s", hex(id(cls.flux_handle)))
            if job_name:
                jobspec.setattr("system.job.name", job_name)
            else:
                job_name = "maestro_flux_job"  # Make safe for .out/.err
            jobspec.cwd = cwd
            jobspec.environment = dict(os.environ)

            # Slurp in extra attributes if not null
            # (-S/--setattr)
            # NOTE: these are sanitized upstream to be (dotpath, value) tuples, flags
            # set to have value of '1' to mach flux cli
            for batch_attr_dotpath, batch_attr_value in addtl_batch_args["attributes"]:
                jobspec.setattr(batch_attr_dotpath, batch_attr_value)

            # Add in job shell options if not null (-o/--setopt)
            # NOTE: these are sanitized upstream to be dotpath, value tuples
            for batch_opt_dotpath, batch_opt_value in addtl_batch_args["shell_options"]:
                jobspec.setattr_shell_option(batch_opt_dotpath, batch_opt_value)
            
            if walltime > 0:
                jobspec.duration = walltime

            jobspec.stdout = f"{job_name}.{{{{id}}}}.out"
            jobspec.stderr = f"{job_name}.{{{{id}}}}.err"

            # Submit our job spec.
            jobid = flux.job.submit(
                cls.flux_handle, jobspec, waitable=waitable, urgency=urgency
            )
            submit_status = SubmissionCode.OK
            retcode = 0

            LOGGER.info(
                "Submission returned status OK. -- "
                "Assigned identifier (%s)",
                jobid,
            )

            # NOTE: cannot pickle JobID instances, so must store jobid's as
            # strings and reconstruct for use later. Also ensure we get the
            # Base58 form instead of integer for better user facing logging
            jobid = str(jobid.f58)

        except ConnectionResetError as exception:
            LOGGER.error("Submission failed -- Message (%s).",
                         exception,
                         exc_info=True)
            jobid = -1
            retcode = -2
            submit_status = SubmissionCode.ERROR
        except Exception as exception:
            LOGGER.error("Submission failed -- Message (%s).",
                         exception,
                         exc_info=True)
            jobid = -1
            retcode = -1
            submit_status = SubmissionCode.ERROR

        return jobid, retcode, submit_status

    @classmethod
    def parallelize(cls, procs, nodes=None, launcher_args=None, **kwargs):

        args = ["flux", "run", "-n", str(procs)]

        # if we've specified nodes, add that to wreckrun
        ntasks = nodes if nodes else 1
        args.append("-N")
        args.append(str(ntasks))

        if "cores per task" in kwargs:
            args.append("-c")
            # Error checking -> more comprehensive handling in base schedulerscriptadapter?
            if not kwargs["cores per task"]:
                cores_per_task = 1
            else:
                cores_per_task = kwargs["cores per task"]

            # args.append(str(kwargs["cores per task"]))
            args.append(str(cores_per_task))

            LOGGER.info("Adding 'cores per task' %s to flux args",
                        str(kwargs["cores per task"]))

        ngpus = kwargs.get("gpus", 0)
        if ngpus:
            gpus = str(ngpus)
            args.append("-g")
            args.append(gpus)

        # flux has additional arguments that can be passed via flags such as
        # '-o', '-S', ...
        if launcher_args is None:
            launcher_args = {}

        # Look for optional exclusive flag
        exclusive = kwargs.pop('exclusive', False)

        addtl = []
        LOGGER.info("Processing 'exclusive': %s", exclusive)
        if exclusive:
            addtl.append("--exclusive")

        addtl_args = kwargs.get("addtl_args", {})
        if addtl_args and launcher_args:
            # TODO: better way to mark things deprecated that's not buried?
            LOGGER.warn("'args' input is deprecated in v1.1.12.  Use the more "
                        "flexible 'launcher_args' going forward. Combining.")
        if 'o' in launcher_args:
            launcher_args['o'].update(**addtl_args)
        else:
            launcher_args['o'] = addtl_args

        addtl += [arg for arg in cls.render_additional_args(launcher_args)]
        args.extend(addtl)

        return " ".join(args)

    @classmethod
    def get_statuses(cls, joblist):
        # We need to import flux here, as it may not be installed on
        # all systems.
        cls.connect_to_flux()

        LOGGER.debug("Flux handle address -- %s", hex(id(cls.flux_handle)))

        # Reconstruct JobID instances from the str form of the Base58 id:
        # NOTE: cannot pickle JobID instances, so must store as strings and
        # reconstruct for use
        jobs_rpc = flux.job.list.JobList(
            cls.flux_handle,
            ids=[flux.job.JobID(jid) for jid in joblist])

        statuses = {}
        for jobinfo in jobs_rpc.jobs():
            LOGGER.debug(f"Checking status of job with id {str(jobinfo.id.f58)}")
            statuses[str(jobinfo.id.f58)] = cls.state(jobinfo.status_abbrev)

        chk_status = JobStatusCode.OK
        #  Print all errors accumulated in JobList RPC:
        try:
            for err in jobs_rpc.errors:
                chk_status = JobStatusCode.ERROR
                LOGGER.error("Error in JobList RPC %s", err)
        except EnvironmentError:
            pass

        return chk_status, statuses

    @classmethod
    def cancel(cls, joblist):
        """
        Cancel a job using Flux 0.17.0 cancellation API.

        :param joblist: A list of job identifiers to cancel.
        :return: CancelCode enumeration that reflects result of cancellation.
        "return: A cancel return code indicating how cancellation call exited.
        """
        # We need to import flux here, as it may not be installed on
        # all systems.
        cls.connect_to_flux()

        LOGGER.debug("Handle address -- %s", hex(id(cls.flux_handle)))
        LOGGER.debug(
            "Attempting to cancel jobs.\nJoblist:\n%s",
            "\n".join(str(j) for j in joblist),
        )

        # NOTE: cannot pickle JobID instances, so must store as strings and
        # reconstruct for use
        jobs_rpc = flux.job.list.JobList(
            cls.flux_handle,
            ids=[flux.job.JobID(jid) for jid in joblist])

        cancel_code = CancelCode.OK
        cancel_rcode = 0
        for job in jobs_rpc.jobs():
            try:
                LOGGER.debug("Cancelling Job %s...", str(job.id.f58))
                flux.job.cancel(cls.flux_handle, int(job.id))
            except Exception as exception:
                LOGGER.error("Job %s: %s", str(job.id.f58), str(exception))
                cancel_code = CancelCode.ERROR
                cancel_rcode = 1

        return cancel_code, cancel_rcode

    @staticmethod
    def state(state):
        if state == "D":        # Note this is actually short for DEPEND and is part of flux' pending virtual state
            return State.PENDING
        elif state == "S":      # Note this is short for SCHED and is also part of flux's pending virtual state
            return State.QUEUED
        elif state == "R":
            return State.RUNNING
        elif state == "C":
            return State.FINISHING
        elif state == "CD":
            return State.FINISHED
        elif state == "F":
            return State.FAILED
        elif state == "CA":
            return State.CANCELLED
        elif state == "TO":
            return State.TIMEDOUT
        else:
            LOGGER.error(f"Unhandled state: {state}")
            return State.UNKNOWN
