# Validus Quant Research Case Study

This is a package that tackles the Validus quant research case study of a private fund manager facing FX volatility and risk.

## Installation

Use the requirements.txt with pip to install download all necessary requirements.

Preferably using VirtualEnv to create a sandbox for all these requirements.

```bash
pip3 install -r requirements.txt
```

## Usage

```bash

source venv/bin/activate
/venv/bin/python3 main.py

```

## Dataset

Dataset exploration uses jupyter

## Flags

PLOT: in main to showcase the plots for the strategies
DEBUG: in main to show plots for testing the Heston model

Model is calibrated with the file `calibrated_heston_params.pkl`
