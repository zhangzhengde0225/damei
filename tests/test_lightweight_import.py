import json
import subprocess
import sys


def test_import_damei_does_not_load_optional_dependencies():
    code = """
import json
import sys
import damei
import damei.comm
import damei.misc
from damei import Tools, data, nn

result = {
    "version": damei.__version__,
    "system": damei.current_system(),
    "colors_len": len(damei.colors(2)),
    "time_ok": bool(damei.misc.current_time()),
    "comm_has_push_stream": hasattr(damei.comm, "push_stream"),
    "lazy_imports": [repr(data), repr(nn)],
    "tools_is_class": isinstance(Tools, type),
    "table": damei.misc.list2table(["A", 1.234], float_bit=1),
    "loaded": {
        name: name in sys.modules
        for name in ["numpy", "easydict", "cv2", "torch"]
    },
}
print(json.dumps(result, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    result = json.loads(proc.stdout)
    assert result["version"]
    assert result["system"] in {"linux", "windows", "macos"}
    assert result["colors_len"] == 2
    assert result["time_ok"] is True
    assert result["comm_has_push_stream"] is True
    assert all("lazy" in item for item in result["lazy_imports"])
    assert result["tools_is_class"] is True
    assert "1.2" in result["table"]
    assert result["loaded"] == {
        "numpy": False,
        "easydict": False,
        "cv2": False,
        "torch": False,
    }
