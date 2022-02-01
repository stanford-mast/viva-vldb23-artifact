#!/bin/bash
query_name="angrybernie"
canary_input="angry-bernie-ground-truth.mp4"
opttarget=("performance" "cost")
platforms=("gpu" "cpu")
input_video=("sotujt-60min.mp4.orig")
selectivity_fraction=("0.05")
proxy_threshold=("0.8")
f1_threshold=("0.8")
hints_plan=("all_hints.py")
costminmax=("min")

for v in ${input_video[@]}; do
    for s in ${selectivity_fraction[@]}; do
        for p in ${proxy_threshold[@]}; do
            for x in ${hints_plan[@]}; do
                bash run_acc_sel.sh -q ${query_name} -s ${s} -c ${canary_input} \
                    -p ${p} -v ${v} -x ${x}
            done # hints_plan
        done # proxy_threshold
    done # selectivity_fraction
done # input_video

# do_warmup is run once per video input and not timed
for l in ${platforms[@]}; do
    for o in ${opttarget[@]}; do
        for v in ${input_video[@]}; do
            do_warmup="1"
            for s in ${selectivity_fraction[@]}; do
                for p in ${proxy_threshold[@]}; do
                    for f in ${f1_threshold[@]}; do
                        for x in ${hints_plan[@]}; do
                            for e in ${costminmax[@]}; do
                                bash run_experiment.sh -q ${query_name} -s ${s} -c ${canary_input} \
                                                       -p ${p} -f ${f} -v ${v} \
                                                       -e ${e} -x ${x} -w ${do_warmup} -l ${l} -o ${o}
                                do_warmup="0"
                            done # costminmax
                        done # hints_plan
                    done # f1_threshold
                done # proxy_threshold
            done # selectivity_fraction
        done # input_video
    done # opttarget
done # platforms

echo "Done!!"
