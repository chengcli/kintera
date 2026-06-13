import os
from pathlib import Path

import kintera


def test_packaged_nasa9_is_available_outside_source_tree(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    packaged_data = (Path(kintera.__file__).parent / "data").resolve()
    search_paths = kintera.get_search_paths()
    kintera.set_search_paths(str(packaged_data))
    try:
        resource = Path(kintera.find_resource("nasa9.dat")).resolve()
    finally:
        kintera.set_search_paths(os.pathsep.join(search_paths))

    assert resource == packaged_data / "nasa9.dat"
