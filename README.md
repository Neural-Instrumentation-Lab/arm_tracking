# Arm Tracking

Code from the arm-tracking project originally developed through Iyad Obeid's NSF Career grant

`matlab/` has a Matlab implementation

`cpp/` has a C++ implementation

Its not clear this is the best code I have from that time but its what I was able to find. Useable as a starting point.

## Installation Instructions

### Prerequisites: 
1. Local installation of `Docker Desktop`
1. WSL (if running on Windows)
1. VSCode
1. In VSCode, install the `Dev Containers` extension
1. In VSCode, install the `WSL` extension (if on Windows)

### Actions
1. Run Docker
1. Run WSL, and clone this repositiory
1. In VSCode, open folder for the repository
1. A pop-up should appear asking to re-open this folder in a
Dev Container - click `Reopen in Container`
1. Open a Python file then look at the Python interpreter 
specified on the bottom right of your window. Click on it 
and select `/usr/local/bin/python`
1. Click the run button to run the code. You can also use debug 
features.

## C++ Acceleration (`pin_ext`)

The inner substep integration loops in `forwardDynamics` and `inverseDynamics`
(100 pinocchio calls per trajectory sample) are ported to C++ via a pybind11
extension module (`pin_ext`). The Python code falls back silently to pure Python
if the extension is not built.

**Relevant files**
- `experiments/experiments_02_thru_09/pin_ext.cpp` — C++ source
- `experiments/experiments_02_thru_09/build_ext.sh` — build script

**The extension is compiled automatically** when the Dev Container image is
built, so a fresh `Reopen in Container` is sufficient after pulling the repo.

**Iterative development** (editing `pin_ext.cpp`) does not require a container
rebuild. The workspace is bind-mounted, so just run inside a container terminal:

```bash
bash experiments/experiments_02_thru_09/build_ext.sh
```

The resulting `.so` lands next to the Python files and is picked up immediately
on the next run.