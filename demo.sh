#!/bin/bash

cd trans4trans

eval "$(conda shell.bash hook)"
conda activate trans4trans

python demo_r200.py --config-file configs/trans10kv2/pvt_tiny_FPT.yaml
