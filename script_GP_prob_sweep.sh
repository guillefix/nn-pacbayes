#!/bin/bash
addqueue -n 100 -m 5 -s '/usr/bin/awk "{print $1}" /users/guillefix/nn-pacbayes/unique_prob_set_1e7_2_7_40_40_1_1.000000_0.000000_relu_fun_samples.txt | parallel --jobs 100 -I{} ./meta_script {}'
#/usr/bin/cut -f1 /users/guillefix/nn-pacbayes/unique_prob_set_1e7_2_7_40_40_1_1.000000_0.000000_relu_fun_samples.txt | xargs -I{} ./meta_script {}

#sed -n 'g;n;n;p' /users/guillefix/nn-pacbayes/unique_prob_set_1e7_2_7_40_40_1_1.000000_0.000000_relu_fun_samples.txt | /usr/bin/cut -f1 | xargs -I{} ./meta_script {}
#sed -n 'g;n;n;p' /users/guillefix/nn-pacbayes/unique_prob_set_1e7_2_7_40_40_1_1.000000_0.000000_relu_fun_samples.txt | tail -19 | /usr/bin/cut -f1 | xargs -I{} ./meta_script {}
