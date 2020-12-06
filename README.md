

__Requirements__

tensorflow (tested for >= 1.14)
pytorch, torchvision,
numpy, matplotlib, scipy, pandas
mpi4py,
GPy
Linux

__Experiments__

The experiments for the paper XXX can be run by running `./meta_script_msweep [dataset] [architecture] [pooling_type]` where dataset can be one of `mnist EMNIST mnist-fashion KMNIST cifar`, and architecture can be one of `fc cnn resnet50 resnet101 resnet152 resnetv2_50 resnetv2_101 resnetv2_152 resnext50 resnext101 densenet121 densenet169 densenet201 mobilenetv2 nasnet vgg19 vgg16`, and poolyng_type can be one of `none avg max`. You also need to run `./make_directories` the first time to create the directories where temporary data is stored.

With the current settings `./meta_script_msweep` will both train the network and compute the PAC-Bayes bounds (for training set sizes `1 3 11 36 122 407 1357 4516 15026 49999`). You can select to only train or only compute the bound, by setting the relevant flags to 0 or 1 at the beginning of the file (note that compute_bound requires having run compute_kernel before).

The code first creates the architecture (as a keras json file), then the dataset (both train and test), then computes the NNGP kernel, then computes the bound, and finally trains the network. The result of each step are saved in folders so that all of these steps can be run in different calls.

The code also allows parallelization by setting n_procs to >1, and multiple gpus by setting ngpus>1. ngpus=0 will run on CPU.

You can run the individual python files (see `./run_experiment` to see which python scripts are used for different computations) with the `--helpfull` flag to see the full list of options available in running the experiments. These options can be appended to the line running `./run_experiment` in `./meta_script_msweep`.

The flag `train_method` in `./meta_script_msweep` can be set to NN, GP, or NTK. Depending on this flag, it will train on the given data using the NNGP or NTK approximation of the given architecture. Any keras architecture is accepted. NNGP is computed via Monte Carlo approximation, and NTK is computed via product of Jacobians at initialization. These should provide good approximations if the network is wide enough. Note that if using train_method=GP or NTK, you need to have `compute_kernel=1` (or have run that previously), in order to train with it (`train=1`). The wiki has some more details on how to use the NNGP code; the NTK computations have similar API, and are found in the file `nngp_kernel/empirical_ntk.py`.

__Plotting__

The plots for the experiments in Sections 7.2 and 7.3 in XXX can be obtained by running the python script `analyze/analyze_msweep_new2.py`. You can set the batch_size to 256 for results for that batch size. I recommend runing this on Atom with the Hydrogen plugin :)

The plots for experiments in Figure 1 can be obtained by running `analyze/analyze_NNGP_data.py`.

For any questions, please contact guillefix [at] gmail.com
