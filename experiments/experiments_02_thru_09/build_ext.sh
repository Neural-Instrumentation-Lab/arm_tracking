#!/usr/bin/env bash
# build_ext.sh — builds pin_ext.so inside the Dev Container.
# Run from anywhere; the .so lands in the same directory as this script.
# Usage:  bash build_ext.sh [--python /path/to/python]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --python) PYTHON="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== pin_ext build ==="
echo "Python : $($PYTHON --version)"
echo "Src    : $SCRIPT_DIR/pin_ext.cpp"

# ── locate includes/libs via Python ──────────────────────────────────────────
eval "$($PYTHON - <<'PYEOF'
import sys, os, sysconfig, pybind11

prefix   = sys.prefix
py_inc   = sysconfig.get_path("include")
ext_suf  = sysconfig.get_config_var("EXT_SUFFIX")
pb_inc   = pybind11.get_include()

# pinocchio headers
pin_inc = ""
for d in [prefix + "/include", "/usr/local/include"]:
    if os.path.exists(os.path.join(d, "pinocchio", "multibody", "model.hpp")):
        pin_inc = d
        break
if not pin_inc:
    print("echo 'ERROR: pinocchio C++ headers not found'; exit 1")
    sys.exit(0)

# Eigen headers
eigen_inc = ""
for d in [prefix + "/include/eigen3",
          prefix + "/include",
          "/usr/local/include/eigen3",
          "/usr/include/eigen3"]:
    if os.path.exists(os.path.join(d, "Eigen", "Core")):
        eigen_inc = d
        break
if not eigen_inc:
    # pinocchio headers often bundle Eigen under its own include path
    eigen_inc = pin_inc   # Eigen/Core should be found relative to pin_inc

# pinocchio shared library
import pinocchio as _pin
pin_pkg  = os.path.dirname(_pin.__file__)
pin_lib_dir = ""
pin_lib     = ""
for d in [pin_pkg, prefix + "/lib", "/usr/local/lib"]:
    for name in ["pinocchio_default", "pinocchio"]:
        if any(os.path.exists(os.path.join(d, "lib" + name + ext))
               for ext in [".so", ".so.3", ".so.2"]):
            pin_lib_dir = d
            pin_lib     = name
            break
    if pin_lib:
        break

if not pin_lib:
    print("echo 'ERROR: libpinocchio not found'; exit 1")
    sys.exit(0)

print(f"PYTHON_INC={py_inc!r}")
print(f"PYBIND_INC={pb_inc!r}")
print(f"PIN_INC={pin_inc!r}")
print(f"EIGEN_INC={eigen_inc!r}")
print(f"PIN_LIB_DIR={pin_lib_dir!r}")
print(f"PIN_LIB={pin_lib!r}")
print(f"EXT_SUFFIX={ext_suf!r}")
PYEOF
)"

echo "  python include : $PYTHON_INC"
echo "  pybind11 inc   : $PYBIND_INC"
echo "  pinocchio inc  : $PIN_INC"
echo "  Eigen inc      : $EIGEN_INC"
echo "  pinocchio lib  : $PIN_LIB_DIR / lib$PIN_LIB"
echo "  ext suffix     : $EXT_SUFFIX"

OUTPUT="$SCRIPT_DIR/pin_ext${EXT_SUFFIX}"

# ── compile ──────────────────────────────────────────────────────────────────
g++ -O3 -shared -fPIC -std=c++17 \
    -I"$PYTHON_INC" \
    -I"$PYBIND_INC" \
    -I"$PIN_INC"    \
    -I"$EIGEN_INC"  \
    "$SCRIPT_DIR/pin_ext.cpp" \
    -L"$PIN_LIB_DIR" -l"$PIN_LIB" \
    -Wl,-rpath,"$PIN_LIB_DIR" \
    -o "$OUTPUT"

echo "=== Built: $OUTPUT ==="
