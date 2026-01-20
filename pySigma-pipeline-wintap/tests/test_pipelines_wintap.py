import pytest
from sigma.backends.duckdb import DuckDBBackend
from sigma.collection import SigmaCollection
from sigma.exceptions import SigmaTransformationError
from sigma.processing.resolver import ProcessingPipelineResolver

from sigma.pipelines.wintap import wintap_pipeline
from sigma.pipelines.wintap.constants import *


# Define a fixture for the DuckDB backend
@pytest.fixture(scope="class")
def duckdb_backend():
    pipe_resolver = ProcessingPipelineResolver()
    pipe_resolver.add_pipeline_class(wintap_pipeline())
    backend = DuckDBBackend(pipe_resolver.resolve(pipe_resolver.pipelines))
    return backend


# Define individual fixtures for each Sigma rule
@pytest.fixture
def windows_image_load_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Image Load
        status: test
        logsource:
            category: image_load
            product: windows
        detection:
            selection:
                ImageLoaded|endswith: '\amsi.dll'
                Image|endswith:
                    - '\ExtExport.exe'
                    - '\odbcconf.exe'
                    - '\regsvr32.exe'
                    - '\rundll32.exe'
            condition: selection
        """
    )


@pytest.fixture
def windows_network_connection_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Network Connection
        status: test
        logsource:
            category: network_connection
            product: windows
        detection:
            selection:
                LogonId: 'George'
                Image|endswith: '\addinutil.exe'
            condition: selection
        """
    )


@pytest.fixture
def windows_registry_set_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Registry Set
        status: test
        logsource:
            category: registry_set
            product: windows
        detection:
            selection:
                EventType: SetValue
                TargetObject|endswith: '\Software\Winternals\BGInfo\Database'
            condition: selection
        """
    )


@pytest.fixture
def windows_registry_event_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Registry Event
        status: test
        logsource:
            product: windows
            category: registry_event
        detection:
            selection:
                TargetObject|contains: '\SAM\SAM\Domains\Account\Users\Names\'
                TargetObject|endswith: '$'
                Image|endswith: '\lsass.exe'
            condition: selection
        """
    )


@pytest.fixture
def windows_registry_delete_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Registry Delete
        status: test
        logsource:
            category: registry_delete
            product: windows
        detection:
            selection:
                EventType: DeleteValue
                TargetObject|endswith: '\Microsoft\Windows\WindowsAI\DisableAIDataAnalysis'
            condition: selection
        """
    )


@pytest.fixture
def windows_registry_add_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Registry Add
        status: test
        logsource:
            product: windows
            category: registry_add
        detection:
            selection:
                EventType: CreateKey
                TargetObject|contains: '\software\NetWire'
            condition: selection
        """
    )


@pytest.fixture
def windows_process_creation_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Process Creation
        status: test
        logsource:
            category: process_creation
            product: windows
        detection:
            selection_img:
                - Image|endswith:
                    - '\7z.exe'
                    - '\7zr.exe'
                    - '\7za.exe'
            selection_extension:
                CommandLine|contains:
                    - '.dmp'
                    - '.dump'
                    - '.hdmp'
            condition: all of selection_*
        """
    )


@pytest.fixture
def windows_missing_fields_sigma_rule():
    return SigmaCollection.from_yaml(
        r"""
        title: Windows Network Connection
        status: test
        logsource:
            category: network_connection
            product: windows
        detection:
            selection:
                Initiated: 'true'
                Image|endswith: '\addinutil.exe'
            condition: selection
        """
    )


def convert_wintap(duckdb_backend, rule: SigmaCollection):
    return duckdb_backend.convert(rule)


## Happy Paths


def test_wintap_pipeline_image_load(duckdb_backend, windows_image_load_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_image_load_sigma_rule)
    assert result is not None
    assert IMAGE_LOAD_TABLE in result


def test_wintap_pipeline_network_connection(duckdb_backend, windows_network_connection_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_network_connection_sigma_rule)
    assert result is not None
    assert NETWORK_TABLE in result


def test_wintap_pipeline_registry_set(duckdb_backend, windows_registry_set_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_registry_set_sigma_rule)
    assert result is not None
    assert REGISTRY_TABLE in result


def test_wintap_pipeline_registry_event(duckdb_backend, windows_registry_event_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_registry_event_sigma_rule)
    assert result is not None
    assert REGISTRY_TABLE in result


def test_wintap_pipeline_registry_delete(duckdb_backend, windows_registry_delete_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_registry_delete_sigma_rule)
    assert result is not None
    assert REGISTRY_TABLE in result


def test_wintap_pipeline_registry_add(duckdb_backend, windows_registry_add_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_registry_add_sigma_rule)
    assert result is not None
    assert REGISTRY_TABLE in result


def test_wintap_pipeline_process_create(duckdb_backend, windows_process_creation_sigma_rule):
    result = convert_wintap(duckdb_backend, windows_process_creation_sigma_rule)
    assert result is not None
    assert JOINED_PROCESS_TABLE in result


## Unhappy Paths


def test_wintap_pipeline_missing_wintap_field(duckdb_backend, windows_missing_fields_sigma_rule):
    with pytest.raises(
        SigmaTransformationError,
        match="wintap does not contain all necessary fields for detection",
    ):
        convert_wintap(duckdb_backend, windows_missing_fields_sigma_rule)
