from sigma.backends.duckdb.duckdb import TABLE_NAME_PATTERN, DuckDBBackend
from sigma.pipelines.base import Pipeline
from sigma.pipelines.common import (
    LogsourceCondition,
    logsource_windows_image_load,
    logsource_windows_network_connection,
    logsource_windows_process_access,
    logsource_windows_process_creation,
    logsource_windows_registry_add,
    logsource_windows_registry_delete,
    logsource_windows_registry_event,
    logsource_windows_registry_set,
)
from sigma.processing.conditions import ExcludeFieldCondition
from sigma.processing.finalization import ConcatenateQueriesFinalizer
from sigma.processing.pipeline import ProcessingItem, ProcessingPipeline, QueryPostprocessingItem
from sigma.processing.postprocessing import ReplaceQueryTransformation
from sigma.processing.transformations import (
    DetectionItemFailureTransformation,
    FieldMappingTransformation,
    RuleFailureTransformation,
)

from .constants import *

wintap_field_mappings = {
    "CommandLine": "args",
    "Computer": "hostname",
    "EventType": "activity_type",
    "Hashes": "hashes",
    "Image": "process_path",
    "ImageLoaded": "filename",
    "LogonId": "user_name",
    "ParentProcessId": "parent_pid_hash",
    "ProcessId": "pid_hash",
    "TargetObject": "reg_path",
    "User": "user_name",
}

wintap_joined_field_mappings = {
    "OriginalFileName": "process_name",
    "ParentCommandLine": "parent_command_line",
    "ParentImage": "parent_process_name",
}

supported_rule_types = [
    logsource_windows_image_load(),
    logsource_windows_network_connection(),
    logsource_windows_registry_set(),
    logsource_windows_registry_event(),
    logsource_windows_registry_delete(),
    logsource_windows_registry_add(),
    logsource_windows_process_creation(),
    # logsource_windows_process_access(),
]


@Pipeline
def wintap_pipeline() -> (
    ProcessingPipeline
):  # Processing pipelines should be defined as functions that return a ProcessingPipeline object.
    return ProcessingPipeline(
        name="Wintap - Sigma rule to wintap duckdb format pipeline",
        allowed_backends=frozenset([DuckDBBackend]),
        priority=20,
        items=[
            # # do not translate rules who have fields we don't track yet
            ProcessingItem(
                identifier="wintap_missing_field",
                transformation=DetectionItemFailureTransformation(
                    "wintap does not contain all necessary fields for detection"
                ),
                field_name_conditions=[
                    ExcludeFieldCondition(
                        fields=list(wintap_field_mappings.keys())
                        + list(wintap_joined_field_mappings.keys())
                    )
                ],
                rule_condition_linking=any,
                rule_conditions=supported_rule_types,
            ),
            ProcessingItem(
                identifier="wintap_duckdb_fieldmapping",
                transformation=FieldMappingTransformation(
                    wintap_field_mappings | wintap_joined_field_mappings
                ),
                rule_conditions=[
                    LogsourceCondition(
                        product="windows",
                    ),
                ],
            ),
            # do not translate rules who have events we don't track yet
            ProcessingItem(
                identifier="wintap_unsupported_rule_type",
                transformation=RuleFailureTransformation(
                    "Rule type is not supported by duckdb wintap backend yet."
                ),
                rule_condition_negation=True,
                rule_condition_linking=any,
                rule_conditions=supported_rule_types,
            ),
        ],
        # put table name in based on the category of event
        postprocessing_items=[
            QueryPostprocessingItem(
                transformation=ReplaceQueryTransformation(
                    pattern=TABLE_NAME_PATTERN, replacement=IMAGE_LOAD_TABLE
                ),
                rule_condition_linking=any,
                rule_conditions=[
                    logsource_windows_image_load(),
                ],
                identifier="image_load_table_mapping",
            ),
            QueryPostprocessingItem(
                transformation=ReplaceQueryTransformation(
                    pattern=TABLE_NAME_PATTERN, replacement=NETWORK_TABLE
                ),
                rule_condition_linking=any,
                rule_conditions=[
                    logsource_windows_network_connection(),
                ],
                identifier="network_connection_table_mapping",
            ),
            QueryPostprocessingItem(
                transformation=ReplaceQueryTransformation(
                    pattern=TABLE_NAME_PATTERN, replacement=REGISTRY_TABLE
                ),
                rule_condition_linking=any,
                rule_conditions=[
                    logsource_windows_registry_set(),
                    logsource_windows_registry_event(),
                    logsource_windows_registry_delete(),
                    logsource_windows_registry_add(),
                ],
                identifier="registry_table_mapping",
            ),
            QueryPostprocessingItem(
                transformation=ReplaceQueryTransformation(
                    pattern=TABLE_NAME_PATTERN, replacement=JOINED_PROCESS_TABLE
                ),
                rule_condition_linking=any,
                rule_conditions=[
                    LogsourceCondition(
                        category="process_tampering",
                        product="windows",
                    ),
                    logsource_windows_process_creation(),
                    # logsource_windows_process_access(),
                ],
                identifier="process_table_mapping",
            ),
        ],
        finalizers=[ConcatenateQueriesFinalizer()],
    )
