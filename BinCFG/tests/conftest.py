import pytest

@pytest.fixture(scope="session")
def print_hashes():
    return False