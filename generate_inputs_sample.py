import numpy as np
import tensorflow as tf
import keras
import pickle
import torchvision
from torchvision import transforms, utils
from math import ceil

from utils import preprocess_flags, save_data
from utils import data_folder,datasets_folder

def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)
    global m

    print("Generating input samples", dataset, m)

    from math import ceil

    if network in ["cnn","fc","inception_resnet_v2", "inception_v3","xception"]:
        if network not in ["cnn","fc"]:
            if network == "xception":
                image_size=71
            else:
                image_size=75
        else:
            image_size=None
    else:
        image_size=32

    if dataset == "cifar":
        d = torchvision.datasets.CIFAR10("./datasets",download=True,
                transform=transforms.Compose(
                    [transforms.ToPILImage()]+
                    ([transforms.Resize(image_size)] if image_size is not None else [])+
                    [transforms.ToTensor()]
                ))
        print(d)
        mm = int(ceil(d.data.shape[0]*5/6))
        (train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
        num_classes = 10
    elif dataset == "mnist":
        num_classes = 10
        d = torchvision.datasets.MNIST("./datasets",download=True,
                transform=transforms.Compose(
                    [transforms.ToPILImage()]+
                    ([transforms.Resize(image_size)] if image_size is not None else [])+
                    [transforms.ToTensor()]
                ))
        print(d)
        mm = int(ceil(d.data.shape[0]*5/6))
        (train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
    elif dataset == "mnist-fashion":
        num_classes = 10
        d = torchvision.datasets.FashionMNIST("./datasets",download=True,
                transform=transforms.Compose(
                    [transforms.ToPILImage()]+
                    ([transforms.Resize(image_size)] if image_size is not None else [])+
                    [transforms.ToTensor()]
                ))
        print(d)
        mm = int(ceil(d.data.shape[0]*5/6))
        (train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
    elif dataset == "KMNIST":
        d = torchvision.datasets.KMNIST("./datasets",download=True,
                transform=transforms.Compose(
                    [transforms.Resize(image_size)] if image_size is not None else []
                ),
            )
        mm = ceil(d.data.shape[0]*5/6)
        (train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
        num_classes = 10
    elif dataset == "EMNIST":
        d = torchvision.datasets.EMNIST("./datasets",download=True,
                transform=transforms.Compose(
                    [transforms.ToPILImage()]+
                    ([transforms.Resize(image_size)] if image_size is not None else [])+
                    [transforms.ToTensor()]
                ),
                split="byclass")
        print(d)
        mm = int(ceil(d.data.shape[0]*5/6))
        (train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
        num_classes = 62

    #TODO: add custom datasets

    #non-image-like datasets:
    elif dataset == "boolean":
        assert network == "fc"
        num_classes = 2
        #we ignore the 0 input, because it casues problems when computing the kernel matrix :P
        if centering:
            inputs = np.array([[float(l)*2.0-1 for l in "{0:07b}".format(i)] for i in range(1,2**7)])
        else:
            inputs = np.array([[float(l) for l in "{0:07b}".format(i)] for i in range(1,2**7)])

        if boolfun is not "none":
            fun = boolfun
        elif boolfun_comp is not "none":
            # open("boolfun_comps.txt","w").write("\n".join(list(funs.keys())))
            funs = pickle.load(open("funs_per_complexity.p","rb"))
            fun = np.random.choice(funs[boolfun_comp])
            print("complexity", boolfun_comp)
        else:
            funs = pickle.load(open("funs_per_complexity.p","rb"))
            comp = np.random.choice(list(funs.keys()))
            print("complexity", comp)
            fun = np.random.choice(funs[comp])
            # funs = {}
            # with open("LZ_freq_1e6_7_40_40_1_relu.txt","r") as f:
            #     for line in f.readlines():
            #         fun,comp,freq = line.strip().split("\t")
            #         if comp not in funs:
            #             funs[comp] = [fun]
            #         else:
            #             funs[comp].append(fun)
            # pickle.dump(funs,open("funs_per_complexity.p","wb"))

        labels=np.array([[int(xx)*2.0-1] for xx in list(fun)])
    else:
        raise NotImplementedError

    global threshold
    if threshold==-1:
        threshold=ceil(num_classes/2)

    ##adding channel dimenions for image datasets without them
    if dataset in ["mnist","mnist-fashion","KMNIST","EMNIST"]:
        train_images = np.expand_dims(train_images,-1)
        test_images = np.expand_dims(test_images,-1)
        if network not in ["cnn","fc"]:
            train_images = np.tile(train_images,(1,1,1,3))
            test_images = np.tile(test_images,(1,1,1,3))
            train_images = np.stack([d.transform(image) for image in train_images])
            train_images = np.transpose(train_images,(0,2,3,1))
            test_images = np.stack([d.transform(image) for image in test_images])
            test_images = np.transpose(test_images,(0,2,3,1))

    if network != "fc":
        image_size = train_images.shape[1]
        number_channels = train_images.shape[-1]

    ##get random training sample##

    # np.random.seed(42069)

    #for datasets that are not images, like the boolean one
    if dataset == "boolean":
        if oversampling:
            probs = list(map(lambda x: threshold/(num_classes*len(inputs)) if x>=threshold else (num_classes-threshold)/(num_classes*len(inputs)), inputs))
            probs = np.array(probs)
            probs /= np.sum(probs)
            indices = np.random.choice(range(len(inputs)), size=int(total_samples), replace=False, p=probs)
        elif oversampling2:
            indices = sum([[i]*(num_classes-threshold) for i in indices if train_labels[i]<threshold] \
                + [[i]*threshold for i in indices if train_labels[i]>=threshold],[])

            m*=int((2*(num_classes-threshold)*threshold/(num_classes)))
        else:
            indices = np.random.choice(range(int(len(inputs))), size=int(total_samples), replace=False)
        # print(indices)
        test_indices = np.array([i for i in range(len(inputs)) if i not in indices])
        train_inputs = inputs[indices,:].astype(np.float32)
        train_labels = labels[indices]
        if training:
            test_inputs = inputs[test_indices,:]
            flat_test_images = test_inputs
            test_labels = labels[test_indices]

        flat_train_images = train_inputs

    else:

        if oversampling:
            probs = list(map(lambda x: threshold/(num_classes) if x>=threshold else (num_classes-threshold)/(num_classes), train_labels))
            probs = np.array(probs)
            probs /= np.sum(probs)
            indices = np.random.choice(range(len(train_images)), size=int(total_samples), replace=False, p=probs)
        elif oversampling2:
            indices = sum([[i]*(num_classes-threshold) for i in indices if train_labels[i]<threshold] \
                + [[i]*threshold for i in indices if train_labels[i]>=threshold],[])

            m*=int((2*(num_classes-threshold)*threshold/(num_classes)))
        else:
            indices = np.random.choice(range(len(train_images)), size=int(total_samples), replace=False)
        # print(indices)
        train_images = (train_images[indices,:,:,:]/255.0).astype(np.float32) #NHWC
        if training:
            test_images = test_images/255.0
            train_labels = np.take(train_labels,indices)
            print(len([x for x in train_labels if x<threshold])/len(train_images))


        #flattened images, as the kernel function takes flattened vectors (row major for NCHW images)
        tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
        flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW
        flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
        if training:
            flat_test_images = np.transpose(test_images, tp_order)  # NHWC -> NCHW
            flat_test_images = np.array([test_image.flatten() for test_image in flat_test_images])

        if channel_normalization:
            #flatten to compute SVD matrix
            print("channel normalizing")
            x = train_images
            flat_x = flat_train_images

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


        if centering:
            #flatten to compute SVD matrix
            print("centering")
            x = train_images
            flat_x = flat_train_images
            flat_x -= flat_x.mean(axis=0)

            x = flat_x.reshape((x.shape[0], x.shape[3], x.shape[1],x.shape[2]))
            tp_order = np.concatenate([[0], np.arange(2, len(train_images.shape)), [1]])
            train_images =  np.transpose(x, tp_order)

            #test images
            if training:
                flat_test_images -= flat_test_images.mean(axis=0)
                test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[3], test_images.shape[1],test_images.shape[2]))
                test_images =  np.transpose(test_images, tp_order)
                # test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2],test_images.shape[3]))
                # test_images = (test_images - x_mean) / x_std

        #WHITENING using training_images
        if whitening:
            #flatten to compute SVD matrix
            print("ZCA whitening")
            x = train_images
            flat_x = flat_train_images
            flat_x -= flat_x.mean(axis=0)
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
            # x = train_images
            # x_mean = np.mean(x, axis=(0,1,2))
            # x_std = np.std(x, axis=(0,1,2))
            # x = (x - x_mean) / x_std
            # train_images = x

            #test images
            if training:
                flat_test_images -= flat_test_images.mean(axis=0)
                flat_test_images = np.matmul(flat_test_images, Q.T)
                test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[3], test_images.shape[1],test_images.shape[2]))
                test_images =  np.transpose(test_images, tp_order)
                # test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2],test_images.shape[3]))
                # test_images = (test_images - x_mean) / x_std


            #flatten again after normalizing
            # tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
            # flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW
            # flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
            # if training:
            #     flat_test_images = np.transpose(test_images, tp_order)  # NHWC -> NCHW
            #     flat_test_images = np.array([test_image.flatten() for test_image in flat_test_images])


    if network == "fc":
        train_images = flat_train_images
        if training:
            test_images = flat_test_images


        input_dim = flat_train_images.shape[1]

    #corrupting images, and adding confusion data

    def binarize(label, threshold):
        return label>=threshold

    # %%
    if training:
        def process_labels(label,label_corruption,threshold,zero_one=False,binarized=True):
            if binarized:
                if zero_one:
                    if np.random.rand() < label_corruption:
                        return np.random.choice([0,1])
                    else:
                        return float(binarize(label,threshold))
                else:
                    if np.random.rand() < label_corruption:
                        return np.random.choice([-1.0,1.0])
                    else:
                        return float(binarize(label,threshold))*2.0-1
            else:
                if np.random.rand() < label_corruption:
                    return np.random.choice(range(num_classes))
                else:
                    return float(label)

        if random_labels:
    	    ys = [[process_labels(label,label_corruption,threshold,zero_one=True,binarized=binarized)] for label in train_labels[:m]] + [[process_labels(label,1.0,threshold,zero_one=True,binarized=binarized)] for label in train_labels[m:]]
        else:
    	    ys = [[process_labels(label,label_corruption,threshold,zero_one=True,binarized=binarized)] for label in train_labels[:m]] + [[float(not binarize(label,threshold))] for label in train_labels[m:]]

        test_ys = np.array([process_labels(label,label_corruption,threshold,zero_one=True,binarized=binarized) for label in test_labels])

    '''SAVING DATA SAMPLES'''

    save_data(train_images,ys,test_images,test_ys,FLAGS)


if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    tf.app.run()
    import gc; gc.collect()
