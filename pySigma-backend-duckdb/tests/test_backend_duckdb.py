import pytest
from sigma.collection import SigmaCollection

from sigma.backends.duckdb import DuckDBBackend

WINTAP_FORMAT = "wintap"


@pytest.fixture
def duckdb_backend():
    return DuckDBBackend()


# TODO: implement tests for some basic queries and their expected results.
def test_duckdb_and_expression(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    fieldA: valueA
                    fieldB: valueB
                condition: selection
        """
            )
        )
        == [r"SELECT * FROM TABLE_NAME WHERE (fieldA ILIKE 'valueA' AND fieldB ILIKE 'valueB')"]
    )


def test_duckdb_or_expression(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection1:
                    fieldA: valueA
                selection2:
                    fieldB: valueB
                condition: 1 of selection*
        """
            )
        )
        == [r"SELECT * FROM TABLE_NAME WHERE (fieldA ILIKE 'valueA' OR fieldB ILIKE 'valueB')"]
    )


def test_duckdb_and_or_expression(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    fieldA:
                        - valueA1
                        - valueA2
                    fieldB:
                        - valueB1
                        - valueB2
                condition: selection
        """
            )
        )
        == [
            r"SELECT * FROM TABLE_NAME WHERE ((fieldA ILIKE 'valueA1' OR fieldA ILIKE 'valueA2') AND (fieldB ILIKE 'valueB1' OR fieldB ILIKE 'valueB2'))"
        ]
    )


def test_duckdb_or_and_expression(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection1:
                    fieldA: valueA1
                    fieldB: valueB1
                selection2:
                    fieldA: valueA2
                    fieldB: valueB2
                condition: 1 of selection*
        """
            )
        )
        == [
            r"SELECT * FROM TABLE_NAME WHERE ((fieldA ILIKE 'valueA1' AND fieldB ILIKE 'valueB1') OR (fieldA ILIKE 'valueA2' AND fieldB ILIKE 'valueB2'))"
        ]
    )


def test_duckdb_in_expression(
    duckdb_backend: DuckDBBackend,
):
    print(
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    fieldA:
                        - valueA
                        - valueB
                        - valueC*
                condition: selection
        """
            )
        )
    )
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    fieldA:
                        - valueA
                        - valueB
                        - valueC*
                condition: selection
        """
            )
        )
        == [
            r"SELECT * FROM TABLE_NAME WHERE (fieldA ILIKE 'valueA' OR fieldA ILIKE 'valueB' OR fieldA ILIKE 'valueC%' ESCAPE '\')"
        ]
    )


def test_duckdb_regex_query(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    fieldA|re: foo.*bar
                    fieldB: foo
                condition: selection
        """
            )
        )
        == [r"SELECT * FROM TABLE_NAME WHERE (fieldA SIMILAR TO 'foo.*bar' AND fieldB ILIKE 'foo')"]
    )


def test_duckdb_cidr_query(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    field|cidr: 192.168.0.0/16
                condition: selection
        """
            )
        )
        == [r"SELECT * FROM TABLE_NAME WHERE (field ILIKE '192.168.%' ESCAPE '\')"]
    )


def test_duckdb_field_name_with_whitespace(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    field name: value
                condition: selection
        """
            )
        )
        == [r"SELECT * FROM TABLE_NAME WHERE ('field name' ILIKE 'value')"]
    )


def test_duckdb_list_contains(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    - Hashes|contains: '09D278F9DE118EF09163C6140255C690'
                condition: selection
        """
            )
        )
        == [
            " ".join(
                r"""SELECT * FROM TABLE_NAME WHERE
            ((CASE
                WHEN typeof(Hashes) = 'VARCHAR[]' THEN list_contains(CAST(Hashes AS VARCHAR[]), '09D278F9DE118EF09163C6140255C690')
                WHEN CAST(Hashes AS VARCHAR) ILIKE '%09D278F9DE118EF09163C6140255C690%' ESCAPE '\' THEN TRUE
                ELSE FALSE
            END))
        """.split()
            )
        ]
    )


# TODO: implement tests for all backend features that don't belong to the base class defaults, e.g. features that were
# implemented with custom code, deferred expressions etc.


def test_duckdb_and_expression(
    duckdb_backend: DuckDBBackend,
):
    assert (
        duckdb_backend.convert(
            SigmaCollection.from_yaml(
                """
            title: Test
            status: test
            logsource:
                category: test_category
                product: test_product
            detection:
                selection:
                    fieldA: valueA
                    fieldB: valueB
                condition: selection
        """
            ),
            WINTAP_FORMAT,
        )
        == [
            r"SELECT * FROM TABLE_NAME WHERE (fieldA ILIKE 'valueA' AND fieldB ILIKE 'valueB') AND daypk = {{ search_day_pk }}"
        ]
    )
