#!/usr/bin/gnuplot
#
# Filename: plots.plt
#
# Copyright 2020 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
#
# Usage: gnuplot plots.plt

set term pngcairo font "OpenSans, 28" size 1920, 1080

set output "training.png";
set title "Training regime";
plot "ft-train.csv" using 0:1 with lines title "f", 'zt-train.csv' using 0:1 with lines title "z";

set output "training-weights.png";
set title "Training regime: weights";
plot 'wo_mag-train.csv' using 0:1 with lines title "|w|";

set output "testing.png"
set title "Testing regime";
plot "ft-test.csv" using 0:1 with lines title "f", 'zt-test.csv' using 0:1 with lines title "z";
