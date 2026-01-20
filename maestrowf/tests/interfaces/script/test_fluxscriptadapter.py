###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Francesco Di Natale, dinatale3@llnl.gov.
#
# LLNL-CODE-734340
# All rights reserved.
# This file is part of MaestroWF, Version: 1.0.0.
#
# For details, see https://github.com/LLNL/maestrowf.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
"""
This module is intended to test maestrowf.interfaces.script.fluxscriptadapter
module. It is setup to use nose or pytest to do that testing.

This was created to verify existing functionality of the ScriptAdapterFactory
as it was converted to dynamically load all ScriptAdapters using a namespace
plugin methodology.
"""
import re

from maestrowf.interfaces.script.fluxscriptadapter import FluxScriptAdapter
from maestrowf.interfaces import ScriptAdapterFactory

from maestrowf.datastructures.core import Study
from maestrowf.datastructures.core.executiongraph import ExecutionGraph
from maestrowf.specification import YAMLSpecification

from rich.pretty import pprint
import pytest


def test_flux_adapter():
    """
    Tests to verify that FluxScriptAdapter has the key property set to 'flux'
    this is validate that existing specifications do not break.
    :return:
    """
    assert FluxScriptAdapter.key == 'flux'


def test_flux_adapter_in_factory():
    """
    Testing to makes sure that the FluxScriptAdapter has been registered
    correctly in the ScriptAdapterFactory.
    :return:
    """
    saf = ScriptAdapterFactory
    # Make sure FluxScriptAdapter is in the facotries object
    assert saf.factories[FluxScriptAdapter.key] == FluxScriptAdapter
    # Make sure the FluxScriptAdapter key is in the valid adapters
    assert FluxScriptAdapter.key in ScriptAdapterFactory.get_valid_adapters()
    # Make sure that get_adapter returns the FluxScriptAdapter when asking
    # for it by key
    assert (ScriptAdapterFactory.get_adapter(FluxScriptAdapter.key) ==
           FluxScriptAdapter)


@pytest.mark.sched_flux         # This one needs to at least import flux to work
@pytest.mark.parametrize(
    "spec_file, variant_id, expected_batch_files",
    [
        (
            "hello_bye_parameterized_flux",
            1,
            {   # NOTE: should we contsruct this, and just use study + step_id + sched.sh.variant_id?
                "hello_world_GREETING.Hello.NAME.Pam": "hello_bye_parameterized_flux.hello_world_GREETING.Hello.NAME.Pam.flux.sh.1",
                "bye_world_GREETING.Hello.NAME.Pam": "hello_bye_parameterized_flux.bye_world_GREETING.Hello.NAME.Pam.flux.sh.1"
            }
        ),
        (
            "hello_bye_parameterized_flux",
            3,
            {   # NOTE: should we contsruct this, and just use study + step_id + sched.sh.variant_id?
                "hello_world_GREETING.Hello.NAME.Pam": "hello_bye_parameterized_flux.hello_world_GREETING.Hello.NAME.Pam.flux.sh.3",
                "bye_world_GREETING.Hello.NAME.Pam": "hello_bye_parameterized_flux.bye_world_GREETING.Hello.NAME.Pam.flux.sh.3"
            }
        ),
        (                       # Group to test unknown alloc arg filtering
            "hello_bye_parameterized_flux",
            4,
            {   # NOTE: should we contsruct this, and just use study + step_id + sched.sh.variant_id?
                "hello_world_GREETING.Hello.NAME.Pam": "hello_bye_parameterized_flux.hello_world_GREETING.Hello.NAME.Pam.flux.sh.4",
                "bye_world_GREETING.Hello.NAME.Pam": "hello_bye_parameterized_flux.bye_world_GREETING.Hello.NAME.Pam.flux.sh.4"
            }
        ),
    ]
)
def test_flux_script_serialization(
        spec_file,
        variant_id,
        expected_batch_files,
        variant_spec_path,
        load_study,
        variant_expected_output,
        text_diff,
        tmp_path                # Pytest tmp dir fixture: Path()):
):
    spec_path = variant_spec_path(spec_file + f"_{variant_id}.yaml")
    # pprint(spec_path)
    yaml_spec = YAMLSpecification.load_specification(spec_path)  # Load this up to get the batch info

    study: Study = load_study(spec_path, tmp_path, dry_run=True)

    pkl_path: str
    ex_graph: ExecutionGraph
    pkl_path, ex_graph = study.stage()
    ex_graph.set_adapter(yaml_spec.batch)

    adapter = ScriptAdapterFactory.get_adapter(yaml_spec.batch["type"])
    pprint(f"Adapter args: {ex_graph._adapter}")
    adapter = adapter(**ex_graph._adapter)

    # Setup a diff ignore pattern to filter out the #INFO (flux version ...
    ignore_patterns = [
        re.compile(r'#MAESTRO-INFO \(flux version\)')
    ]

    # Loop over the steps and execute them
    for step_name, step_record in ex_graph.values.items():
        if step_name == "_source":
            continue
        pprint(f"Step name: {step_name}")
        ex_graph._execute_record(step_record, adapter)

        # pprint(step_name)
        # pprint("Step record:")
        # pprint(step_record)
        # # pprint(_record)
        # pprint("Written script:")
        with open(step_record.script, "r") as written_script_file:
            written_script = written_script_file.read()
            # pprint(written_script.splitlines())

        assert step_name in expected_batch_files

        with open(variant_expected_output(expected_batch_files[step_name]), 'r') as ebf:
            expected_script = ebf.read()

        # assert written_script == expected_script
        assert text_diff(written_script, expected_script, ignore_patterns=ignore_patterns)



@pytest.mark.sched_flux         # This one needs to at least import flux to work
@pytest.mark.parametrize(
    "spec_file, variant_id, expected_jobspec_keys",
    [
        (
            "hello_bye_parameterized_flux",
            2,                  # Use the invalid queue one to avoid waiting for jobs to run
            {   # NOTE: should we contsruct this, and just use study + step_id + sched.sh.variant_id?
                "hello_world_GREETING.Hello.NAME.Pam": {
                    'resources': [{'exclusive': True}],
                    'attributes': {
                        'system': {
                            'shell': {
                                'options': {'bar': 42, 'foo': 'bar'},
                            },
                            'queue': 'pdebug',
                            'bank': 'guests',
                            'files': {
                                'conf.json': {'data': {'resource': {'rediscover': True, "noverify": True}}}
                            },
                            'gpumode': 'CPX'
                        }
                    }
                },
                "bye_world_GREETING.Hello.NAME.Pam": {
                    'attributes': {
                        'system': {
                            'shell': {
                                'options': {'bar': 42, 'foo': 'bar'},
                            },
                            'queue': 'pdebug',
                            'bank': 'guests',
                            'files': {
                                'conf.json': {'data': {'resource': {'rediscover': True, "noverify": True}}}
                            },
                            'gpumode': 'CPX'
                        }
                    }
                },
            }
        ),
    ]
)
def test_flux_job_opts(
        spec_file,
        variant_id,
        expected_jobspec_keys,
        variant_spec_path,
        load_study,
        flux_jobspec_check,
        tmp_path,               # Pytest tmp dir fixture: Path()):
        generate_jobspec_from_script,
):
    spec_path = variant_spec_path(spec_file + f"_{variant_id}.yaml")
    pprint(spec_path)
    yaml_spec = YAMLSpecification.load_specification(spec_path)  # Load this up to get the batch info

    print(f"{tmp_path=}")
    study: Study = load_study(spec_path, tmp_path, dry_run=False)
    print(f"{study.output_path=}")
    pkl_path: str
    ex_graph: ExecutionGraph
    pkl_path, ex_graph = study.stage()
    ex_graph.set_adapter(yaml_spec.batch)

    adapter = ScriptAdapterFactory.get_adapter(yaml_spec.batch["type"])
    adapter = adapter(**ex_graph._adapter)

    # Setup a diff ignore pattern to filter out the #INFO (flux version ...
    ignore_patterns = [
        re.compile(r'#MAESTRO-INFO \(flux version\)')
    ]

    # Loop over the steps and execute them
    for step_name, step_record in ex_graph.values.items():
        if step_name == "_source":
            continue
        
        ex_graph._execute_record(step_record, adapter)
        pprint("Step name:")
        pprint(step_name)
        pprint("Step record:")
        pprint(step_record)

        retcode, job_status = ex_graph.check_study_status()

        pprint(f"{retcode=}, {job_status=}")
        for record_name, status in job_status.items():
            if step_name == record_name:
                jobid = ex_graph.values[record_name].jobid[-1]

                # Cancel the job; we just need jobspecs, not completion
                c_record = adapter.cancel_jobs([jobid])

                flux_jobspec_check(jobid,
                                   expected_jobspec_keys[step_name],
                                   source_label='mark.parametrized jobspec',
                                   ignore_keys={"tasks.0.command.3"},
                                   debug=False)

                # Add a second test to ensure it matches written scripts' jobspec
                script_jobspec = generate_jobspec_from_script(step_record.script)
                
                flux_jobspec_check(jobid,
                                   script_jobspec,
                                   source_label='script jobspec',
                                   ignore_keys={
                                       "tasks.0.command.3",  # script/cmd obtained via different methods between two modes
                                       "attributes.system.environment",  # Not populated in 'base' jobspec from python, can't filter from batch --dry-run
                                       # TURN OFF TEMPORARILY TO PATCH UP ASSERTION MESSAGE TRUNCATION
                                       "attributes.system.cwd",  # Irrelevant difference
                                       "attributes.system.shell.options.rlimit",  # Not populated via python?
                                       "attributes.system.files.script",  # Attached to cmd in python?
                                   },
                                   debug=False)
