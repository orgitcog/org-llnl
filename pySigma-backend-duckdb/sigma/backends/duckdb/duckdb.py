import re
from typing import Any, ClassVar, Dict, List, Optional, Pattern, Tuple, Union

from sigma.conditions import (
    ConditionAND,
    ConditionFieldEqualsValueExpression,
    ConditionItem,
    ConditionNOT,
    ConditionOR,
    ConditionValueExpression,
)
from sigma.conversion.base import TextQueryBackend
from sigma.conversion.deferred import DeferredQueryExpression
from sigma.conversion.state import ConversionState
from sigma.exceptions import SigmaFeatureNotSupportedByBackendError
from sigma.processing.pipeline import ProcessingPipeline
from sigma.rule import SigmaRule
from sigma.types import SigmaCIDRExpression, SigmaCompareExpression, SigmaString, SpecialChars

TABLE_NAME_PATTERN = "TABLE_NAME"


class DuckDBBackend(TextQueryBackend):
    """DuckDB - in-memory SQL Database backend."""

    # Table Name Regular Expression to be used by tansformation to replace the string
    table_name = "TABLE_NAME"

    # Operator precedence: tuple of Condition{AND,OR,NOT} in order of precedence.
    # The backend generates grouping if required
    name: ClassVar[str] = "DuckDB - in-memory SQL Database backend"
    formats: Dict[str, str] = {
        "default": "Plain DuckDB - in-memory SQL Database queries",
        "wintap": "DuckDB formatted with Wintap partitioning",
    }
    requires_pipeline: bool = False

    precedence: ClassVar[Tuple[ConditionItem, ConditionItem, ConditionItem]] = (
        ConditionNOT,
        ConditionAND,
        ConditionOR,
    )
    parenthesize: bool = True
    group_expression: ClassVar[str] = (
        "({expr})"  # Expression for precedence override grouping as format string with {expr} placeholder
    )

    # Generated query tokens
    token_separator: str = " "  # separator inserted between all boolean operators
    or_token: ClassVar[str] = "OR"
    and_token: ClassVar[str] = "AND"
    not_token: ClassVar[str] = "NOT"
    eq_token: ClassVar[str] = (
        " ILIKE "  # Token inserted between field and value (without separator)
    )

    # String output
    ## Fields
    ### Quoting
    field_quote: ClassVar[str] = (
        "'"  # Character used to quote field characters if field_quote_pattern matches (or not, depending on field_quote_pattern_negation). No field name quoting is done if not set.
    )
    field_quote_pattern: ClassVar[Pattern] = re.compile(
        "^[a-zA-Z0-9_]*$"
    )  # Quote field names if this pattern (doesn't) matches, depending on field_quote_pattern_negation. Field name is always quoted if pattern is not set.
    field_quote_pattern_negation: ClassVar[bool] = (
        True  # Negate field_quote_pattern result. Field name is quoted if pattern doesn't matches if set to True (default).
    )

    ### Escaping
    # field_escape: ClassVar[
    #    str
    # ] = "\\"  # Character to escape particular parts defined in field_escape_pattern.
    # field_escape_quote: ClassVar[
    #    bool
    # ] = True  # Escape quote string defined in field_quote
    # field_escape_pattern: ClassVar[Pattern] = re.compile(
    #    "\\s"
    # )  # All matches of this pattern are prepended with the string contained in field_escape.

    ## Values
    str_quote: ClassVar[str] = "'"  # string quoting character (added as escaping character)
    escape_char: ClassVar[str] = "\\"  # Escaping character for special characrers inside string
    wildcard_multi: ClassVar[str] = "%"  # Character used as multi-character wildcard
    wildcard_single: ClassVar[str] = "_"  # Character used as single-character wildcard
    add_escaped: ClassVar[str] = ""  # Characters quoted in addition to wildcards and string quote
    filter_chars: ClassVar[str] = ""  # Characters filtered
    bool_values: ClassVar[Dict[bool, str]] = {  # Values to which boolean values are mapped.
        True: "true",
        False: "false",
    }

    # String matching operators. if none is appropriate eq_token is used.
    startswith_expression: ClassVar[str] = "{field} ILIKE '{value}%' ESCAPE '\\'"
    endswith_expression: ClassVar[str] = "{field} ILIKE '%{value}' ESCAPE '\\'"
    contains_expression: ClassVar[str] = " ".join(
        """
            (CASE
                WHEN typeof({field}) = 'VARCHAR[]' THEN list_contains(CAST({field} AS VARCHAR[]), '{value}')
                WHEN CAST({field} AS VARCHAR) ILIKE '%{value}%' ESCAPE '\\' THEN TRUE
                ELSE FALSE
            END)
        """.split()
    )
    wildcard_match_expression: ClassVar[str] = (
        "{field} ILIKE '{value}' ESCAPE '\\'"  # Special expression if wildcards can't be matched with the eq_token operator
    )

    # Regular expressions
    # Regular expression query as format string with placeholders {field}, {regex}, {flag_x} where x
    # is one of the flags shortcuts supported by Sigma (currently i, m and s) and refers to the
    # token stored in the class variable re_flags.
    re_expression: ClassVar[str] = "{field} SIMILAR TO '{regex}'"
    re_escape_char: ClassVar[str] = ""  # Character used for escaping in regular expressions
    re_escape: ClassVar[Tuple[str]] = ()  # List of strings that are escaped
    re_escape_escape_char: bool = True  # If True, the escape character is also escaped
    re_flag_prefix: bool = (
        True  # If True, the flags are prepended as (?x) group at the beginning of the regular expression, e.g. (?i). If this is not supported by the target, it should be set to False.
    )
    # Mapping from SigmaRegularExpressionFlag values to static string templates that are used in
    # flag_x placeholders in re_expression template.
    # By default, i, m and s are defined. If a flag is not supported by the target query language,
    # remove it from re_flags or don't define it to ensure proper error handling in case of appearance.
    # re_flags: Dict[SigmaRegularExpressionFlag, str] = {
    #    SigmaRegularExpressionFlag.IGNORECASE: "i",
    #    SigmaRegularExpressionFlag.MULTILINE : "m",
    #    SigmaRegularExpressionFlag.DOTALL    : "s",
    # }

    # Case sensitive string matching expression. String is quoted/escaped like a normal string.
    # Placeholders {field} and {value} are replaced with field name and quoted/escaped string.
    case_sensitive_match_expression: ClassVar[str] = "{field} ILIKE {value} ESCAPE '\\'"
    # Case sensitive string matching operators similar to standard string matching. If not provided,
    # case_sensitive_match_expression is used.
    case_sensitive_startswith_expression: ClassVar[str] = "{field} ILIKE '{value}%' ESCAPE '\\'"
    case_sensitive_endswith_expression: ClassVar[str] = "{field} ILIKE '%{value}' ESCAPE '\\'"
    case_sensitive_contains_expression: ClassVar[str] = "{field} ILIKE '%{value}%' ESCAPE '\\'"

    # CIDR expressions: define CIDR matching if backend has native support. Else pySigma expands
    # CIDR values into string wildcard matches.
    cidr_expression: ClassVar[Optional[str]] = (
        None  # CIDR expression query as format string with placeholders {field}, {value} (the whole CIDR value), {network} (network part only), {prefixlen} (length of network mask prefix) and {netmask} (CIDR network mask only).
    )

    # Numeric comparison operators
    compare_op_expression: ClassVar[str] = (
        "{field} {operator} {value}"  # Compare operation query as format string with placeholders {field}, {operator} and {value}
    )
    # Mapping between CompareOperators elements and strings used as replacement for {operator} in compare_op_expression
    compare_operators: ClassVar[Dict[SigmaCompareExpression.CompareOperators, str]] = {
        SigmaCompareExpression.CompareOperators.LT: "<",
        SigmaCompareExpression.CompareOperators.LTE: "<=",
        SigmaCompareExpression.CompareOperators.GT: ">",
        SigmaCompareExpression.CompareOperators.GTE: ">=",
    }

    # Expression for comparing two event fields
    field_equals_field_expression: ClassVar[Optional[str]] = (
        None  # Field comparison expression with the placeholders {field1} and {field2} corresponding to left field and right value side of Sigma detection item
    )
    field_equals_field_escaping_quoting: Tuple[bool, bool] = (
        True,
        True,
    )  # If regular field-escaping/quoting is applied to field1 and field2. A custom escaping/quoting can be implemented in the convert_condition_field_eq_field_escape_and_quote method.

    # Null/None expressions
    field_null_expression: ClassVar[str] = (
        "{field} IS NULL"  # Expression for field has null value as format string with {field} placeholder for field name
    )

    # Field existence condition expressions.
    field_exists_expression: ClassVar[str] = (
        "{field} IS NOT NULL"  # Expression for field existence as format string with {field} placeholder for field name
    )
    field_not_exists_expression: ClassVar[str] = (
        "{field} IS NULL"  # Expression for field non-existence as format string with {field} placeholder for field name. If not set, field_exists_expression is negated with boolean NOT.
    )

    # Field value in list, e.g. "field in (value list)" or "field containsall (value list)"
    convert_or_as_in: ClassVar[bool] = False  # Convert OR as in-expression
    convert_and_as_in: ClassVar[bool] = False  # Convert AND as in-expression
    in_expressions_allow_wildcards: ClassVar[bool] = (
        False  # Values in list can contain wildcards. If set to False (default) only plain values are converted into in-expressions.
    )
    field_in_list_expression: ClassVar[str] = (
        "{field} {op} ({list})"  # Expression for field in list of values as format string with placeholders {field}, {op} and {list}
    )
    # or_in_operator: ClassVar[str] = (
    #     ""  # Operator used to convert OR into in-expressions. Must be set if convert_or_as_in is set
    # )
    # and_in_operator : ClassVar[str] = "contains-all"    # Operator used to convert AND into in-expressions. Must be set if convert_and_as_in is set
    list_separator: ClassVar[str] = ", "  # List element separator

    # Value not bound to a field
    unbound_value_str_expression: ClassVar[str] = (
        '"{value}"'  # Expression for string value not bound to a field as format string with placeholder {value}
    )
    unbound_value_num_expression: ClassVar[str] = (
        "{value}"  # Expression for number value not bound to a field as format string with placeholder {value}
    )
    # unbound_value_re_expression : ClassVar[str] = '_=~{value}'   # Expression for regular expression not bound to a field as format string with placeholder {value} and {flag_x} as described for re_expression

    # Query finalization: appending and concatenating deferred query part
    deferred_start: ClassVar[str] = (
        " "  # String used as separator between main query and deferred parts
    )
    deferred_separator: ClassVar[str] = ""  # String used to join multiple deferred query parts
    deferred_only_query: ClassVar[str] = (
        ""  # String used as query if final query only contains deferred expression
    )

    def __init__(
        self,
        processing_pipeline: Optional[ProcessingPipeline] = None,
        collect_errors: bool = False,
        table_name: str = "",
        **kwargs,
    ):
        super().__init__(processing_pipeline, collect_errors, **kwargs)
        self.table_name = table_name

    def convert_value_str(
        self, s: SigmaString, state: ConversionState, no_quote: bool = False
    ) -> str:
        converted = s.convert(
            self.escape_char,
            self.wildcard_multi,
            self.wildcard_single,
            self.add_escaped,
            self.filter_chars,
        )
        converted = converted.replace("'", "''")
        if self.decide_string_quoting(s) and not no_quote:
            return self.quote_string(converted)
        else:
            return converted

    def convert_condition_field_eq_val_str(
        self, cond: ConditionFieldEqualsValueExpression, state: ConversionState
    ) -> Union[str, DeferredQueryExpression]:
        try:
            remove_quote = True
            if self._is_startswith(cond):
                expr = self.startswith_expression
                value = cond.value[:-1]
            elif self._is_endswith(cond):
                expr = self.endswith_expression
                value = cond.value[1:]
            elif self._is_contains(cond):
                expr = self.contains_expression
                value = cond.value[1:-1]
            elif self._is_wildcard_match(cond):
                expr = self.wildcard_match_expression
                value = cond.value
            else:
                expr = f"{self.escape_and_quote_field(cond.field)}{self.eq_token}{self.convert_value_str(cond.value, state)}"
                value = cond.value
                remove_quote = False

            if remove_quote:
                return expr.format(
                    field=self.escape_and_quote_field(cond.field),
                    value=self.convert_value_str(value, state, remove_quote),
                )
            else:
                return expr.format(
                    field=self.escape_and_quote_field(cond.field),
                    value=self.convert_value_str(value, state),
                )
        except TypeError:
            raise NotImplementedError()

    def _is_startswith(self, cond: ConditionFieldEqualsValueExpression) -> bool:
        return (
            self.startswith_expression is not None
            and cond.value.endswith(SpecialChars.WILDCARD_MULTI)
            and not cond.value[:-1].contains_special()
        )

    def _is_endswith(self, cond: ConditionFieldEqualsValueExpression) -> bool:
        return (
            self.endswith_expression is not None
            and cond.value.startswith(SpecialChars.WILDCARD_MULTI)
            and not cond.value[1:].contains_special()
        )

    def _is_contains(self, cond: ConditionFieldEqualsValueExpression) -> bool:
        return (
            self.contains_expression is not None
            and cond.value.startswith(SpecialChars.WILDCARD_MULTI)
            and cond.value.endswith(SpecialChars.WILDCARD_MULTI)
            and not cond.value[1:-1].contains_special()
        )

    def _is_wildcard_match(self, cond: ConditionFieldEqualsValueExpression) -> bool:
        return self.wildcard_match_expression is not None and (
            cond.value.contains_special()
            or self.wildcard_multi in cond.value
            or self.wildcard_single in cond.value
            or self.escape_char in cond.value
        )

    def convert_condition_field_eq_val_cidr(
        self, cond: ConditionFieldEqualsValueExpression, state: ConversionState
    ) -> Union[str, DeferredQueryExpression]:
        cidr: SigmaCIDRExpression = cond.value
        expanded = cidr.expand()
        expanded_cond = ConditionOR(
            [
                ConditionFieldEqualsValueExpression(cond.field, SigmaString(network))
                for network in expanded
            ],
            cond.source,
        )
        return self.convert_condition(expanded_cond, state)

    def finalize_query_default(
        self, rule: SigmaRule, query: str, index: int, state: ConversionState
    ) -> Any:
        if self.table_name:
            return f"SELECT * FROM {self.table_name} WHERE ({query})"
        return f"SELECT * FROM {TABLE_NAME_PATTERN} WHERE ({query})"

    def finalize_query_wintap(
        self, rule: SigmaRule, query: str, index: int, state: ConversionState
    ) -> Any:
        if self.table_name:
            return f"SELECT * FROM {self.table_name} WHERE ({query}) AND daypk = {{{{ search_day_pk }}}}"
        return f"SELECT * FROM {TABLE_NAME_PATTERN} WHERE ({query}) AND daypk = {{{{ search_day_pk }}}}"

    # TODO : match_bm25
    def convert_condition_val_str(
        self, cond: ConditionValueExpression, state: ConversionState
    ) -> Union[str, DeferredQueryExpression]:
        """Conversion of value-only strings."""
        raise SigmaFeatureNotSupportedByBackendError(
            "Value-only string expressions (i.e Full Text Search or 'keywords' search) are not supported by the backend."
        )

    # TODO : match_bm25
    def convert_condition_val_num(
        self, cond: ConditionValueExpression, state: ConversionState
    ) -> Union[str, DeferredQueryExpression]:
        """Conversion of value-only numbers."""
        raise SigmaFeatureNotSupportedByBackendError(
            "Value-only number expressions (i.e Full Text Search or 'keywords' search) are not supported by the backend."
        )

    def finalize_output_wintap(self, queries: List[Dict]) -> str:
        return queries
