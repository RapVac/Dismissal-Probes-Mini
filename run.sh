#!/bin/bash

date
python3 ./setup.py

date
python3 ./train_probes.py

date
python3 -i ./benchmark.py
