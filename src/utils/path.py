import pathlib


def get_repo_dir() -> pathlib.PosixPath:
    this_dir = pathlib.Path(__file__).parent
    return (this_dir / "../..").resolve()
