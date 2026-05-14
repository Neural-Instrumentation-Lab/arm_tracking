# Arm Tracking

Code from the arm-tracking project originally developed through Iyad Obeid's NSF Career grant

`matlab/` has a Matlab implementation

`cpp/` has a C++ implementation

Its not clear this is the best code I have from that time but its what I was able to find. Useable as a starting point.

## Setup and Usage

### Prerequisites
1. Local installation of `Docker Desktop`
1. WSL (if running on Windows)
1. VSCode
1. In VSCode, install the `Dev Containers` extension
1. In VSCode, install the `WSL` extension (if on Windows)

### Python

1. Run Docker
1. Run WSL and clone this repository
1. In VSCode, open the repository folder
1. A pop-up should appear asking to re-open this folder in a Dev Container — click `Reopen in Container`
1. Open a Python file, then click the Python interpreter shown in the bottom right and select `/usr/local/bin/python`
1. Click the run button to run the code. Debug features are also available.

### MATLAB

Your project files are automatically mounted into `/home/matlab/Documents/MATLAB` inside the container.

**Browser UI** (recommended):
```bash
./matlab.sh browser
```
Then open http://localhost:8888 in your browser.

**Command line** (interactive terminal):
```bash
./matlab.sh cli
```

**Licensing:** On first run you will be prompted to log in with your MathWorks account. If you use a network license server, set `MLM_LICENSE_FILE=port@server` in your environment before running.