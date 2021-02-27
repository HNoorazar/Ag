#!/bin/bash

for runname in $(seq 10 57)
do
qdel $runname.mgt2-ib.local
done
