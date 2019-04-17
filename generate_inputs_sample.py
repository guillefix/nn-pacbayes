
import numpy as np
import tensorflow as tf
import keras
import pickle

data_folder = "data/"
datasets_folder = "datasets/"

def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    if dataset == "cifar":
            (train_images,train_labels),(test_images,test_labels) = pickle.load(open(datasets_folder+"cifar10_dataset.p","rb"))
            train_labels = [label[0] for label in train_labels]
            test_labels = [label[0] for label in test_labels]
    elif dataset == "mnist":
            (train_images,train_labels),(test_images,test_labels) = pickle.load(open(datasets_folder+"mnist_dataset.p","rb"))
            train_images = np.expand_dims(train_images,-1)
            test_images = np.expand_dims(test_images,-1)
    elif dataset == "mnist-fashion":
            (train_images,train_labels),(test_images,test_labels) = pickle.load(open(datasets_folder+"mnist_fashion_dataset.p","rb"))
            train_images = np.expand_dims(train_images,-1)
            test_images = np.expand_dims(test_images,-1)
    else:
            raise NotImplementedError

    if network != "fc":
        image_size = train_images.shape[1]
        number_channels = train_images.shape[-1]

    from math import ceil

    indices = np.random.choice(range(len(train_images)), size=total_samples, replace=False)
    train_images = (train_images[indices,:,:,:]/255.0).astype(np.float32) #NHWC
    if training:
        test_images = test_images/255.0
        train_labels = np.take(train_labels,indices)

    #flattened images, as the kernel function takes flattened vectors (row major for NCHW images)
    tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
    flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW
    flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
    if training:
        flat_test_images = np.transpose(test_images, tp_order)  # NHWC -> NCHW
        flat_test_images = np.array([test_image.flatten() for test_image in flat_test_images])

    #WHITENING using training_images
    if whitening:
        #flatten to compute SVD matrix
        print("whitening")
        x = train_images
        flat_x = flat_train_images
        sigma = np.matmul(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = np.linalg.svd(sigma)
        zca_epsilon = 1e-10  # avoid division by 0
        d = np.diag(1. / np.sqrt(s + zca_epsilon))
        Q = np.matmul(np.matmul(u, d), u.T)
        flat_x = np.matmul(flat_x, Q.T)
        flat_train_images = flat_x

        #normalize each channel (3 colors for e.g.)
        #to do this we reshape the tensor to NHWC form
        x = flat_x.reshape((x.shape[0], x.shape[3], x.shape[1],x.shape[2]))
        tp_order = np.concatenate([[0], np.arange(2, len(train_images.shape)), [1]])
        train_images =  np.transpose(x, tp_order)
        x = train_images
        x_mean = np.mean(x, axis=(0,1,2))
        x_std = np.std(x, axis=(0,1,2))
        x = (x - x_mean) / x_std
        train_images = x

        #test images
        if training:
            flat_test_images = np.matmul(flat_test_images, Q.T)
            test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[3], test_images.shape[1],test_images.shape[2]))
            test_images =  np.transpose(test_images, tp_order)
            # test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2],test_images.shape[3]))
            test_images = (test_images - x_mean) / x_std


        #flatten again after normalizing
        tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
        flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW
        flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
        if training:
            flat_test_images = np.transpose(test_images, tp_order)  # NHWC -> NCHW
            flat_test_images = np.array([test_image.flatten() for test_image in flat_test_images])


    if network == "fc":
        train_images = flat_train_images
        if training:
            test_images = flat_test_images

    print(network, dataset)

    input_dim = flat_train_images.shape[1]

    #corrupting images, and adding confusion data

    # %%
    if training:
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

        test_ys = np.array([corrupted_label(label,label_corruption,zero_one=True,binarized=binarized) for label in test_labels])

    '''SAVING DATA SAMPLES'''

    import h5py
    filename=data_folder
    for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels"]:
        filename+=str(FLAGS[flag])+"_"
    filename += "data.h5"
    h5f = h5py.File(filename,"w")
    h5f.create_dataset('train_images', data=train_images)

    if training:
        ys = [y[0] for y in ys]
        h5f.create_dataset('ys', data=ys)
        h5f.create_dataset('test_images', data=test_images)
        h5f.create_dataset('test_ys', data=test_ys)

    h5f.close()

if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    tf.app.run()
    import gc; gc.collect()
