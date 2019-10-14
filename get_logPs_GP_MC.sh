#!/bin/bash
#grep -e boolfun run_experiment-330156.out | sed -n -e 's/^.*.boolfun.: //p' | head -c 130
#sed -n '/pre-confusion-correction/{x;p;d;}; x'

for i in `seq 321410 321557`;
do
    export i=$i
    fun=$(grep -e boolfun run_experiment-${i}.out | sed -n -e 's/^.*.boolfun.: //p' | head -c 130)
    logP=$(sed -n '/pre-confusion-correction/{x;p;d;}; x' run_experiment-${i}.out )
    echo ${fun}$logP >> GPMC_logPs.txt
done
