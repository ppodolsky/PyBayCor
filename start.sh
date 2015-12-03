#!/usr/bin/env bash
cd /input
export PYTHONPATH=$PYTHONPATH:/home/pygammacombo
python3 combiner_pymc.py $@
cp -r /input/output/ /