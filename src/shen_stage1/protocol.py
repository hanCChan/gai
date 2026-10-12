"""Development-only signal access policy, independent of PyTorch."""
from pathlib import Path

DEVELOPMENT_FILES = {
    f"{folder}/{rx}_day{day}_wireless_{kind}.h5"
    for rx in ("n210_1", "rtl_6")
    for folder, day, kind in (("Train", 1, "train"), ("Test", 1, "test"), ("Test", 2, "test"))
}
ROLE_PATHS = {
    "train": "Train/{receiver}_day1_wireless_train.h5",
    "validation": "Test/{receiver}_day2_wireless_test.h5",
    "same_day_diagnostic": "Test/{receiver}_day1_wireless_test.h5",
}


def development_path(root: Path, relative: str) -> Path:
    if relative not in DEVELOPMENT_FILES:
        raise PermissionError(f"Signal access is disabled for {relative}")
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise PermissionError("Dataset path escapes the read-only data root")
    if path != base / relative:
        raise PermissionError("Development file cannot alias another file through a symlink")
    return path
