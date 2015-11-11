#!/bin/bash -xe

for m in {1..9..1}
do
    python my_nnet.py simple 0.$m 0.2
done
