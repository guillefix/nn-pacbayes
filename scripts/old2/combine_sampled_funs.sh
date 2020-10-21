#!/bin/bash

cat results/sgd_bayes_10000_784_2_32_sgd_mse_64__*_nn_train_functions.txt | sed -e 's/^M//' | awk '{tot+=1;A[$1]+=1}END{for(i in A) print i,(A[i]/tot)}' | sort -grk 2 > sgd_bayes_10000_784_2_32_sgd_mse_64__combined.txt

