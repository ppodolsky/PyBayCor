#!/usr/bin/env bash
cd /input
for i in "$@"
do
case $i in
    -n=*|--prefix=*)
    N="${i#*=}"
    ;;
    -c=*|--searchpath=*)
    COMBINATION="${i#*=}"
    ;;
    *)
    ;;
esac
done
export PYTHONPATH=$PYTHONPATH:/home/pygammacombo
python3 combiner_pymc.py ${N}
cp -r /input/output /ouput