import numpy as np
import tensorflow as tf

import load_dataset
#from gpflow import settings
# import tqdm

data_folder = "data/"

def main(_):

    FLAGS = tf.app.flags.FLAGS

    globals().update(FLAGS.flag_values_dict())

    filter_sizes = [[5,5],[2,2]]*5
    filter_sizes = filter_sizes[:number_layers]
    padding=["VALID", "SAME"]*5
    padding= padding[:number_layers]
    strides=[[1, 1]] * 10
    strides= strides[:number_layers]

    import pickle

    if dataset == "cifar":
            (train_images,train_labels),(test_images,test_labels) = pickle.load(open(data_folder+"cifar10_dataset.p","rb"))
            image_size=32
            number_channels=3
            train_labels = [label[0] for label in train_labels]
            test_labels = [label[0] for label in test_labels]
    elif dataset == "mnist":
            (train_images,train_labels),(test_images,test_labels) = pickle.load(open(data_folder+"mnist_dataset.p","rb"))
            train_images = np.expand_dims(train_images,-1)
            test_images = np.expand_dims(test_images,-1)
            image_size=28
            number_channels=1
    elif dataset == "mnist-fashion":
            (train_images,train_labels),(test_images,test_labels) = pickle.load(open(data_folder+"mnist_fashion_dataset.p","rb"))
            train_images = np.expand_dims(train_images,-1)
            test_images = np.expand_dims(test_images,-1)
            image_size=28
            number_channels=1
    else:
            raise NotImplementedError

    from math import ceil

    total_samples = ceil(m*(1.0+confusion))

    indices = np.random.choice(range(len(train_images)), size=total_samples, replace=False)
    train_images = train_images[indices,:,:,:]/255.0
    test_images = test_images/255.0
    train_labels = np.take(train_labels,indices)

    tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
    flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW
    tp_order = np.concatenate([[0,len(test_images.shape)-1], np.arange(1, len(test_images.shape)-1)])
    flat_test_images = np.transpose(test_images, tp_order)  # NHWC -> NCHW
    #flat_train_images = train_images
    #flat_test_images = test_images

    flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
    flat_test_images = np.array([train_image.flatten() for train_image in flat_test_images])

    print(confusion,network,dataset)

    input_dim = flat_train_images.shape[1]
    # %%
    def corrupted_label(label,label_corruption,zero_one=False,binarized=True):
        if binarized:
            if zero_one:
                if np.random.rand() < label_corruption:
                    return np.random.choice([0,1])
                else:
                    return float((label>=5))
            else:
                if np.random.rand() < label_corruption:
                    return np.random.choice([-1.0,1.0])
                else:
                    return float((label>=5))*2.0-1
        else:
            if np.random.rand() < label_corruption:
                return np.random.choice(range(10))
            else:
                return float(label)
    if random_labels:
	    ys = [[corrupted_label(label,label_corruption,zero_one=True,binarized=binarized)] for label in train_labels[:m]] + [[corrupted_label(label,1.0,zero_one=True,binarized=binarized)] for label in train_labels[m:]]
    else:
	    ys = [[corrupted_label(label,label_corruption,zero_one=True,binarized=binarized)] for label in train_labels[:m]] + [[float((label<5))] for label in train_labels[m:]]

    # COMPUTE KERNEL
    from GP_prob_gpy import GP_prob

    if network=="cnn":
        if compute_bound:
            from cnn_kernel import kernel_matrix
            K = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

        #ys = np.array([(y+1)/2 for y in ys])
        #train_images = np.stack([np.reshape(image,(image_size,image_size,number_channels)) for image in train_images])
        #test_images = np.stack([np.reshape(image,(image_size,image_size,number_channels)) for image in test_images])
        test_ys = np.array([corrupted_label(label,label_corruption,zero_one=True,binarized=binarized) for label in test_labels])

    elif network=="resnet":
        if compute_bound:
            from resnet_kernel import kernel_matrix
            K = kernel_matrix(flat_train_images,depth=number_layers,image_size=image_size,number_channels=number_channels,n_blocks=3,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

        #train_images = np.stack([np.reshape(image,(image_size,image_size,number_channels)) for image in train_images])
        #test_images = np.stack([np.reshape(image,(image_size,image_size,number_channels)) for image in test_images])
        test_ys = np.array([corrupted_label(label,label_corruption,zero_one=True,binarized=binarized) for label in test_labels])

    elif network == "fc":
        if compute_bound:
            from fc_kernel import kernel_matrix
            K = kernel_matrix(flat_train_images,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)

        #ys = np.array([(y+1)/2 for y in ys])
        # train_images = [np.reshape(image,(28,28,1)) for image in train_images]
        # test_images = [np.reshape(image,(28,28,1)) for image in test_images]
        train_images = flat_train_images
        test_images = flat_test_images
        test_ys = np.array([corrupted_label(label,label_corruption,zero_one=True,binarized=binarized) for label in test_labels])

    if compute_bound:
        logPU = GP_prob(K,flat_train_images,np.array(ys))

        # m=128
        delta = 2**-10
        bound = (-logPU+2*np.log(total_samples)+1-np.log(delta))/total_samples
        rho = confusion/(1.0+confusion)
        bound = (bound - 0.5*rho)/(1-rho) #to correct for the confusion changing the training data distribution!
        # not fixing it because analyze assumes the wrong formula :PP
        #bound = (bound - 0.5*confusion)/(1-confusion) 
        print("Bound: ", bound)
        print("Accuracy bound: ", 1-bound)
        useful_flags = FLAGS.flag_values_dict().copy()
        del useful_flags["data_dir"]
        del useful_flags["helpfull"]
        del useful_flags["help"]
        del useful_flags["helpshort"]
        if "h" in useful_flags: del useful_flags["h"]
        del useful_flags["f"]
        del useful_flags["prefix"]
        with open(prefix+"bounds.txt","a") as file:
            file.write("#")
            for key, value in sorted(useful_flags.items()):
                file.write("{}\t".format(key))
            file.write("bound")
            file.write("\n")
            for key, value in sorted(useful_flags.items()):
                file.write("{}\t".format(value))
            file.write("{}".format(bound))
            file.write("\n")

    ys = [y[0] for y in ys]
    np.save(open(data_folder+network+"_"+dataset+"_"+str(label_corruption)+"_"+str(confusion)+"_ys.np","wb"),ys)
    np.save(open(data_folder+network+"_"+dataset+"_"+str(label_corruption)+"_"+str(confusion)+"_train_images.np","wb"),train_images)
    np.save(open(data_folder+network+"_"+dataset+"_"+str(label_corruption)+"_"+str(confusion)+"_test_images.np","wb"),test_images)
    np.save(open(data_folder+network+"_"+dataset+"_"+str(label_corruption)+"_"+str(confusion)+"_test_ys.np","wb"),test_ys)

if __name__ == '__main__':

    f = tf.app.flags

    f.DEFINE_integer('m',None,"Number of training examples")
    f.DEFINE_float('label_corruption', 0.0, "Fraction of corrupted labels")
    f.DEFINE_float('confusion',0.0,"Number of confusion samples to add to training data, as a fraction of m")
    f.DEFINE_string('dataset', None, "The dataset to use")
    f.DEFINE_boolean('binarized', True, "Whether to convert classification labels to binary")
    f.DEFINE_float('sigmaw', 1.0, "The variance parameter of the weights; their variance will be sigmaw/sqrt(number of inputs to neuron")
    f.DEFINE_float('sigmab', 1.0, "The variance of the biases")
    f.DEFINE_string('network', None, "The type of network to use")
    f.DEFINE_integer('number_layers', None, "The number of layers in the network")
    f.DEFINE_boolean('compute_bound', False, "Whether to compute the PAC-Bayes bound or just generate the training data")
    f.DEFINE_boolean('random_labels', True, "Whether the confusion data is constructed by randomizing the labels, or by taking a wrong label")
    f.DEFINE_integer('n_gpus', 1, "Number of GPUs to use")
    f.DEFINE_string('prefix', "", "A prefix to use for the result files")


    tf.app.run()
    import gc; gc.collect()
