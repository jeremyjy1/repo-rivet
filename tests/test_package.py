import repo_rivet


def test_package_version() -> None:
    assert repo_rivet.__version__ == "0.1.0"
