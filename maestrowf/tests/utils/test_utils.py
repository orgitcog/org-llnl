from contextlib import nullcontext as does_not_raise
from pathlib import Path
import string
import pytest
from pytest import raises
from rich.pretty import pprint

from maestrowf.datastructures.core import ParameterGenerator
from maestrowf.datastructures.core.parameters import Combination
from maestrowf.utils import (
    make_safe_path,
    parse_version,
    dict_to_dot_strings,
    iter_dotpath_items,
    unflatten_dotpath_tuples,
    unflatten_dotpath_dict,
    coerce_dict_values,
    update_recursive
)
from packaging.version import Version, InvalidVersion

from hypothesis import given, HealthCheck, settings, strategies

@pytest.mark.parametrize(
    "version_string, expected_version, error",
    [
        ("0.49.0", Version("0.49.0"), does_not_raise()),
        ("0.50.0rc2", Version("0.50.0rc2"), does_not_raise()),
        ("0.49.0-225-g53e087510", Version("0.49.0"), does_not_raise()),
        ("2.0.0-rc.1+build.123", Version("2.0.0-rc.1+build.123"), does_not_raise()),
        ("2.0.0+build.1848", Version("2.0.0+build.1848"), does_not_raise()),
        ("1.2.3-0123", Version("1.2.3-0123"), does_not_raise()),
        ("1.2.3-0123.0123", None, raises(InvalidVersion)),
        ("1.0.0-alpha..1", None, raises(InvalidVersion)),
        ("01.1.1", Version("01.1.1"), does_not_raise()),
        ("9.8.7-whatever+meta+meta", None, raises(InvalidVersion)),
    ],
)
def test_parse_version(version_string, expected_version, error):
    """
    Test version parser that first applies pep440 style with a fallback
    to semantic version parser against subset of semver's tests
    and a few variants of flux core's version strings.
    """
    with error:
        version_parts = parse_version(version_string)
        assert version_parts == expected_version


@pytest.mark.parametrize(
    "test_version_string, ref_version, expected, base_expected",
    [
        ("0.49.0", Version("0.49.0"), True, True),
        ("0.50.0rc2", Version("0.49.0"), True, True),
        ("0.49.0-225-g53e087510", Version("0.49.0"), True, True),
        ("0.48.0", Version("0.49.0"), False, False),
        ("0.49.0rc1", Version("0.49.0"), False, True),
    ],
)
def test_version_greater(test_version_string, ref_version, expected, base_expected):
    """
    Test version comparison between variants of flux core's version strings
    and Maestro's flux verison adapters to ensure correct adapter version
    selection and error handling.  Tests raw comparisons as well as fallback
    to base_version for ignoring dev/pre-release variants
    """
    test_version = parse_version(test_version_string)
    ver_cmp = test_version >= ref_version
    print(f"Version '{test_version}': base = '{test_version.base_version}', is prerelease = '{test_version.is_prerelease}'")

    assert ver_cmp == expected

    ver_cmp_base = test_version.base_version >= ref_version.base_version
    assert ver_cmp_base == base_expected


@pytest.mark.parametrize(
    "test_base_path, test_param_combos, expected_paths",
    [
        (
            "hello_world",
            {
                "NAME": {
                    "values": ["Pam", "Jim", "Michael", "Dwight"],
                    "labels": "NAME.%%"
                },
                "GREETING": {
                    "values": ["Hello", "Ciao", "Hey", "Hi"],
                    "labels": "GREETING.%%"
                },
                "FAREWELL": {
                    "values": ["Goodbye", "Farewell", "So long", "See you later"],
                    "labels": "FAREWELL.%%"
                }
            },
            ['hello_world/FAREWELL.Goodbye.GREETING.Hello.NAME.Pam',
             'hello_world/FAREWELL.Farewell.GREETING.Ciao.NAME.Jim',
             'hello_world/FAREWELL.So_long.GREETING.Hey.NAME.Michael',
             'hello_world/FAREWELL.See_you_later.GREETING.Hi.NAME.Dwight']
        ),
        (
            "foo_bar_kwargs",
            {
                "FOOBAR": {
                    "values": ["--foobar 9 2 5 1", "--foobar 1 6 4 3"],
                    "labels": "FOOBAR.%%"
                },
            },
            ['foo_bar_kwargs/FOOBAR.--foobar_9_2_5_1',
             'foo_bar_kwargs/FOOBAR.--foobar_1_6_4_3']
        ),
    ],
)
def test_param_combo_path_sanitizer(test_base_path, test_param_combos, expected_paths):
    """
    Test sanitization of combo strings for use in workspace paths.
    """
    params = ParameterGenerator()
    for param_key, param_dict in test_param_combos.items():
        if "name" not in param_dict:
            params.add_parameter(param_key, param_dict['values'], param_dict['labels'])
        else:
            params.add_parameter(param_key, param_dict['values'], param_dict['labels'], param_dict['name'])

    for idx, combo in enumerate(params):

        combo_str = combo.get_param_string(list(test_param_combos.keys()))
        print(f"{combo_str=}")        
        test_path = make_safe_path(test_base_path, combo_str)
        assert test_path == expected_paths[idx]


@given(
    strategies.text(
        min_size=3,
        max_size=20,
        alphabet=strategies.characters(
            codec='ascii', min_codepoint=32, max_codepoint=126
        )
    ),
)
@settings(max_examples=100, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_path_sanitizer(tmpdir, test_path_str):
    """
    Test sanitization of misc strings for use in workspace paths.
    Similar to combo_path_sanitizer, but uses simpler string inputs for easier random testing

    Notes
    -----
    Explicitly checking for empty paths for now to handle hypothesis' tendency to generate
    strings that are all punctuation characters since those characters are simply removed from
    from the safe paths...   Need better strategy for that kind of replacement long term
    (force hashing?)
    """
    # print(f"{test_path_str=}")
    # print(f"{tmpdir=}")         # switch with tmp_path when make_safe_path converted to pathlib
    test_path_name = make_safe_path("", test_path_str)

    if all(c in string.punctuation for c in test_path_name):
        pytest.xfail("Handling of all-punctuation (i.e. '.') paths is not well suppored yet; know limitation pending rework")
        
    test_path = Path(tmpdir) / test_path_name
    # print(f"{test_path=}")
    path_found = False
    try:
        test_path.touch()
        for path in Path(tmpdir).iterdir():
            if path.absolute() == test_path.absolute() and path.is_file():
                path_found = True
                break
            
    except FileNotFoundError:
        pprint(f"Creating '{test_path}' yielded FileNotFoundError")
        path_found = False
    except NotADirectoryError:
        pprint(f"Creating '{test_path}' yielded NotADirectoryError")
        path_found = False


    if test_path_name:
        assert path_found == True
    else:
        # Handle case of strings that only contain characters make_safe_path says are invalid,
        # leading to empty strings.
        assert test_path_name == ""


@pytest.mark.parametrize(
    "dict_to_flatten, print_none, expected_strings",
    [
        (
            {"foo": {"bar": 2}},
            True,
            ["foo.bar=2"],
        ),
        (
            {"foo": {"bar": 2, "foo2": 42}},
            True,
            ["foo.bar=2", "foo.foo2=42"],
        ),
        (
            {"foo": {"bar": 2, "foo3": None}},
            False,
            ["foo.bar=2", "foo.foo3"],
        ),
        (
            {"foo": {"bar": 2, "foo3": None}},
            True,
            ["foo.bar=2", "foo.foo3=None"],
        ),
        (
            {
                "foo": {"bar": 2, "foo2": 42},
                "foobar": {"too_many_foos": 9001},
            },
            True,
            ["foo.bar=2", "foo.foo2=42", "foobar.too_many_foos=9001"],
        ),
    ],
)
def test_dict_to_strings(dict_to_flatten, print_none, expected_strings):
    """
    Test flattening of dictionaries to lists of dot syntax strings
    """
    dot_strings = dict_to_dot_strings(dict_to_flatten, print_none=print_none)

    assert dot_strings == expected_strings

@pytest.mark.parametrize(
    "dict_to_flatten, expected_tuples",
    [
        (
            {"foo": {"bar": 2}},
            [("foo.bar", 2)],
        ),
        (
            {"foo": {"bar": 2, "foo2": 42}},
            [("foo.bar", 2), ("foo.foo2", 42)],
        ),
        (
            {"foo": {"bar": 2, "foo3": None}},
            [("foo.bar", 2), ("foo.foo3", None)],
        ),
        (
            {
                "foo": {"bar": 2, "foo2": 42},
                "foobar": {"too_many_foos": 9001},
                "foo.foobar": 19
            },
            [("foo.bar", 2), ("foo.foo2", 42), ("foobar.too_many_foos", 9001), ("foo.foobar", 19)],
        ),
        (
            {
                "foo": {"bar": [2, 4], "foo2": 42},
                "foobar": {"too_many_foos": 9001},
                "foo.foobar": [19, 21]
            },
            [("foo.bar", [2, 4]), ("foo.foo2", 42), ("foobar.too_many_foos", 9001), ("foo.foobar", [19, 21])],
        ),
    ],
)
def test_dict_to_dotpath_tuples(dict_to_flatten, expected_tuples):
    """
    Test flattening of dictionaries to lists of dot syntax strings
    """
    dotpath_tuples = list(iter_dotpath_items(dict_to_flatten))

    assert dotpath_tuples == expected_tuples


@pytest.mark.parametrize(
    "dotpaths_to_unflatten, expected_dicts",
    [
        (
            [("foo.bar", 2)],
            {"foo": {"bar": 2}},
        ),
        (
            [("foo.bar", 2), ("foo.foo2", 42)],
            {"foo": {"bar": 2, "foo2": 42}},
        ),
        (
            [("foo.bar", 2), ("foo.foo3", None)],
            {"foo": {"bar": 2, "foo3": None}},
        ),
        (
            [("foo.bar", 2), ("foo.foo2", 42), ("foobar.too_many_foos", 9001), ("foo.foobar", 19)],
            {
                "foo": {"bar": 2, "foo2": 42, "foobar": 19},
                "foobar": {"too_many_foos": 9001},
            },
        ),
        (
            [("foo.bar", [2, 4]), ("foo.foo2", 42), ("foobar.too_many_foos", 9001), ("foo.foobar", [19, 21])],
            {
                "foo": {"bar": [2, 4], "foo2": 42, "foobar": [19, 21]},
                "foobar": {"too_many_foos": 9001},
            },
        ),
    ],
)
def test_unflatten_dotpath_tuples(dotpaths_to_unflatten, expected_dicts):
    """
    Test unflattening of mixed dotpath tuples to nested dicts
    """
    unflattened_dict = unflatten_dotpath_tuples(dotpaths_to_unflatten)

    assert unflattened_dict == expected_dicts


@pytest.mark.parametrize(
    "dict_to_flatten, expected_dict",
    [
        (
            {
                "foo": {"bar": 2, "foo2": 42},
                "foo.foobar": 19,
                "foobar": {"too_many_foos": 9001},
            },
            {
                "foo": {"bar": 2, "foo2": 42, "foobar": 19},
                "foobar": {"too_many_foos": 9001},
            },
        ),
        (
            {
                "foo": {"bar": [2, 4], "foo2": 42},
                "foo.foobar": [19, 21],
                "foobar": {"too_many_foos": 9001},
            },
            {
                "foo": {"bar": [2, 4], "foo2": 42, "foobar": [19, 21]},
                "foobar": {"too_many_foos": 9001},
            },
        ),
    ],
)
def test_unflatten_dotpath_dict(dict_to_flatten, expected_dict):
    """
    Test unflattening dicts that may have dotpath style keys to pure
    dictionary
    """
    flattened_dict = unflatten_dotpath_dict(dict_to_flatten)

    assert flattened_dict == expected_dict


@pytest.mark.parametrize(
    "dict_to_coerce, transform, expected_dict",
    [
        (
            {
                "foo": {"bar": 2, "foo2": None},
                "foobar": None,
                "foobar2": 3
            },
            (lambda x: 1 if x is None else x),
            {
                "foo": {"bar": 2, "foo2": 1},
                "foobar": 1,
                "foobar2": 3
            },
        ),
    ],
)
def test_coerce_dict_values(dict_to_coerce, transform, expected_dict):
    """
    Test unflattening dicts that may have dotpath style keys to pure
    dictionary
    """
    coerced_dict = coerce_dict_values(dict_to_coerce, transform)
    
    assert coerced_dict == expected_dict
    assert coerced_dict != dict_to_coerce

@pytest.mark.parametrize(
    "base_dict, update_dict, expected",
    [
        # Simple overwrite
        (
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
            {"a": 1, "b": 3, "c": 4}
        ),
        # Nested dictionary merge (recursive)
        (
            {"a": {"b": 1, "c": 2}},
            {"a": {"c": 3, "d": 4}},
            {"a": {"b": 1, "c": 3, "d": 4}}
        ),
        # Overwrite with non-dict in update_dict
        (
            {"a": {"b": 1}, "c": 2},
            {"a": 42},
            {"a": 42, "c": 2}
        ),
        # Deeply nested merge
        (
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"d": 2}}},
            {"a": {"b": {"c": 1, "d": 2}}}
        ),
        # Mixed: recursive merge and overwrite
        (
            {"x": {"y": 1}, "z": 10},
            {"x": {"y": 2, "w": 3}, "z": {"nested": 5}},
            {"x": {"y": 2, "w": 3}, "z": {"nested": 5}}
        ),
    ]
)
def test_update_recursive(base_dict, update_dict, expected):
    # Use deepcopy to avoid mutating test data
    result = update_recursive(base_dict, update_dict)
    assert result == expected
    # Also check that base_dict itself was mutated as expected
    assert base_dict == expected
