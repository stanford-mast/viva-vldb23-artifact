#!/bin/bash

queries=("deepface" "angrybernie" "amsterdamdock" "dunk")

for q in ${queries[@]}; do
    bash query_sweep_${q}.sh
done

