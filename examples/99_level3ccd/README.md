# Level3 Control Co-Design (CCD) problem using Open-Loop Optimal Control (OLOC)

Developed by:
Yong Hoon Lee (University of Illinois at Urbana-Champaign) and
Saeid Bayat (University of Illinois at Urbana-Champaign)

## Prerequisite

```
python -m pip install dymos
```

Optionally, IPOPT included in the pyoptsparse can be installed and used with dymos. Please refer to the [Installation Guide](https://openmdao.github.io/dymos/installation.html).

## Usage

```
python level3ccd_run0_createdesigns.py
python level3ccd_run1_evaldesigns.py -np 16
python level3ccd_run2_construct.py
python level3ccd_run3_optimization.py
```
