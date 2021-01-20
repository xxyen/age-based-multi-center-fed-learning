#!/usr/bin/env bash

python3 experiment.py -dataset femnist -experiment fedsem -configuration four.yaml

python3 experiment.py -dataset femnist -experiment fedsem -configuration two.yaml
