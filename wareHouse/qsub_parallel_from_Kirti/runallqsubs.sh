#!/bin/bash
for (( i = 1; i <= 87; i++ ))
do
qsub ./qsub_set$i
done
