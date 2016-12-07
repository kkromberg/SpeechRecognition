#!/bin/bash

# $1 is the log from the neural network training output 

cat $1 | grep "Epoch train" | cut -f 5 -d' ' > training.error.tmp;
cat $1 | grep "Epoch cv"    | cut -f 8 -d' ' > testing.error.tmp;

python plot.py

#rm training.error.tmp testing.error.tmp
