#!/bin/bash
query_name="angrybernie"
canary_input="angry-bernie-ground-truth.mp4"
input_video=("sotujt-60min.mp4.orig")
selectivity_fraction=("0.05")
proxy_threshold=("0.8")
f1_threshold=("0.8")
hints_plan=("all_hints.py")
#hints_plan=("all_hints.py equals_hints.py no_hints.py proxy_hints.py superset_hints.py")
costminmax=("min")

# do_warmup is run once per video input and not timed
for v in ${input_video[@]}; do
    do_warmup="1"
    for s in ${selectivity_fraction[@]}; do
        for p in ${proxy_threshold[@]}; do
            for f in ${f1_threshold[@]}; do
                for x in ${hints_plan[@]}; do
                    for e in ${costminmax[@]}; do
                        bash run_experiment.sh -q ${query_name} -s ${s} -c ${canary_input} \
                                               -p ${p} -f ${f} -v ${v} \
                                               -e ${e} -x ${x} -w ${do_warmup}
                        do_warmup="0"
                    done # costminmax
                done # hints_plan
            done # f1_threshold
        done # proxy_threshold
    done # selectivity_fraction
done # input_video

echo "Done!!"
exit 0
