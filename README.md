# Arm Tracking

Code from the arm-tracking project originally developed through Iyad Obeid's NSF Career grant

`matlab/` has a Matlab implementation

`cpp/` has a C++ implementation

Its not clear this is the best code I have from that time but its what I was able to find. Useable as a starting point.

## Installation Instructions

* Prerequisite: `conda` or `anaconda`

1. Use conda to create an environment from the `environment.yml` file
```bash
conda env create -f environment.yml
```

2. Load the environment with
```bash
conda activate arm-tracking
```

3. Run the code:
```bash
python model_v00.py
```

Note that `requirements.txt` is deprecated in favor of `environment.yml`. As stated, use `conda` not `venv`/`pip` to create the environment. 