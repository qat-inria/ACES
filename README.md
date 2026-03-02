# ACES
Verification + Noise estimation

[Result page](https://qat.inria.fr/gospel).

## Setup virtual environment and run `pytest`

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest
```
## Run `gospel/scripts/aces.py`

```
python aces.py
```

## Run `gospel/scipts/hot_gate.py`

```
python hot_gate.py canonical heatmap.svg --nqubits 5 --nlayers 2 --shots 10000 --depol-prob 0.1 --seed 12345
```
