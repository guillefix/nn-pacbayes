import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle
import torchvision
from torchvision import transforms, utils
from math import ceil
import keras_applications
import torch

from utils import preprocess_flags, save_data
from utils import data_folder,datasets_folder

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    print(FLAGS)
    globals().update(FLAGS)
    global m, total_samples, num_classes

    print("Generating input samples", dataset, m)

    from math import ceil

    if dataset == "cifar":
            image_size=32
            number_channels=3
    elif dataset == "imagenet":
            image_size=256
            number_channels=3
    elif dataset == "mnist":
            image_size=28
            number_channels=1
    elif dataset == "mnist-fashion":
            image_size=28
            number_channels=1
    elif dataset == "KMNIST":
            image_size=28
            number_channels=1
    elif dataset == "EMNIST":
            image_size=28
            number_channels=1
    elif dataset == "boolean":
        input_dim = 7
        image_size = None
    elif dataset == "calabiyau":
        input_dim = 180
    else:
        raise NotImplementedError

    if network in ["cnn","fc","inception_resnet_v2", "inception_v3","xception"]:
        if network not in ["cnn","fc"]:
            if network == "xception":
                image_size=max(image_size,71)
            else:
                image_size=max(image_size,75)
    else:
        image_size=max(image_size,32)

    if dataset is not "boolean" or dataset is not "calabiyau":
        image_width = image_height = image_size

    #image datasets
    aliases = {"cifar":"CIFAR10","mnist":"MNIST","mnist-fashion":"FashionMNIST","imagenet":"ImageNet"}
    if dataset in ["cifar","mnist","mnist-fashion","KMNIST","EMNIST","imagenet"]:
        if dataset in aliases:
            dataset_attr = aliases[dataset]
        else:
            dataset_attr = dataset
        dataset_constructor = getattr(torchvision.datasets,dataset_attr)
        transformation = transforms.Compose(
            [transforms.ToPILImage()]+
            ([transforms.Resize(image_size)] if image_size is not None else [])+
            [transforms.ToTensor()]
        ) 
        extra_kwargs = {}
        if dataset == "EMNIST":
            extra_kwargs = {"split":"balanced"}
        d1 = dataset_constructor("./datasets",download=True,
                transform=transformation, train=True, **extra_kwargs)
        d2 = dataset_constructor("./datasets",download=True,
                transform=transformation, train=False, **extra_kwargs)
        num_classes = len(d1.classes)
        #mm = int(ceil(d.data.shape[0]*5/6))
        full_data = np.concatenate([d1.data,d2.data])
        full_targets = np.concatenate([d1.targets,d2.targets])
        if out_of_sample_test_error:
            #if extended_test_set:
            #    (train_images,train_labels),(test_images,test_labels) = (data[:mm], targets[:mm]),(data[mm:],targets[mm:])
            #else:
            (train_images,train_labels),(test_images,test_labels) = (d1.data, d1.targets),(d2.data,d2.targets)
        else:
            (train_images,train_labels),(test_images,test_labels) = (d1.data, d1.targets),(full_data,full_targets)
        #train_images = torch.Tensor(train_images)
        #test_images = torch.Tensor(test_images)
        print(train_images.min(), train_images.max())

    #TODO: add custom datasets

    #non-image-like datasets:
    else:
        if dataset == "boolean":
            assert network == "fc"
            num_classes = 2
            #we ignore the 0 input, because it casues problems when computing the kernel matrix :P when sigmab==0 though
            if centering:
                inputs = np.array([[float(l)*2.0-1 for l in "{0:07b}".format(i)] for i in range(0,2**7)])
            else:
                if sigmab==0:
                    inputs = np.array([[float(l) for l in "{0:07b}".format(i)] for i in range(1,2**7)])
                else:
                    inputs = np.array([[float(l) for l in "{0:07b}".format(i)] for i in range(0,2**7)])

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

            print("fun",fun)

            if sigmab==0 and not centering:
                labels=np.array([[int(xx)*2.0-1] for xx in list(fun)[1:]]) #start from 1 because we ignored the 0th input
            else:
                labels=np.array([[int(xx)*2.0-1] for xx in list(fun)[0:]])
        elif dataset == "calabiyau":
            assert network == "fc"
            num_classes = 2
            #we ignore the 0 input, because it casues problems when computing the kernel matrix :P
            data = np.load("datasets/calabiyau.npz")
            inputs, labels = data["inputs"], data["targets"]
            if whitening:
                inputs = inputs - inputs.mean(0)
        else:
            raise NotImplementedError

    global threshold
    if threshold==-1:
        threshold=ceil(num_classes/2)

    # print(train_images.shape)
    ##get random training sample##
    # and perform some more processing

    # np.random.seed(42069)

    '''GET TRAINING SAMPLE INDICES'''
    '''AND DO PRE-PROCESSING if it's an image dataset'''
    #for datasets that are not images, like the boolean one
    if dataset == "boolean" or dataset == "calabiyau":
        if not random_training_set:
            raise NotImplementedError

        if booltrain_set is not None:
            indices = [i for i,x in enumerate(booltrain_set) if x == "1"]
        elif oversampling:
            probs = list(map(lambda x: threshold/(num_classes*len(inputs)) if x>=threshold else (num_classes-threshold)/(num_classes*len(inputs)), inputs))
            probs = np.array(probs)
            probs /= np.sum(probs)
            indices = np.random.choice(range(len(inputs)), size=int(total_samples), replace=False, p=probs)
        elif oversampling2:
            indices = np.random.choice(range(len(inputs)), size=int(total_samples), replace=False)
            indices = sum([[i]*(num_classes-threshold) for i in indices if train_labels[i]<threshold] \
                + [[i]*threshold for i in indices if train_labels[i]>=threshold],[])
            #print("Indices: ", indices)

            m*=int((2*(num_classes-threshold)*threshold/(num_classes)))
        else:
            indices = np.random.choice(range(int(len(inputs))), size=int(total_samples), replace=False)
        # print(indices)
        print("train_set", "".join([("1" if i in indices else "0") for i in range(int(len(inputs)))]))
        if out_of_sample_test_error:
            test_indices = np.array([i for i in range(len(inputs)) if i not in indices])
        else:
            test_indices = np.array(range(len(inputs)))
        train_inputs = inputs[indices,:].astype(np.float32)
        train_labels = labels[indices]
        if training:
            test_inputs = inputs[test_indices,:]
            flat_test_images = test_inputs
            test_labels = labels[test_indices]

        flat_train_images = train_inputs

    #for image datasets
    else:
        #data processing functions assume the images have values in range [0,255]
        #global train_images_obs
        #train_images_obs=train_images
        max1 = torch.max(train_images).item()
        max2 = torch.max(test_images).item()
        print("maxs", max1,max2)
        max_val = max(max1,max2)
        #train_images  = train_images.numpy().astype(np.float32)*255.0/max_val
        #test_images = test_images.numpy().astype(np.float32)*255.0/max_val
        train_images  = train_images.numpy().astype(np.uint8)
        test_images = test_images.numpy().astype(np.uint8)


        #GET TRAINIG SAMPLE INDICES
        if random_training_set:
            if oversampling:
                probs = list(map(lambda x: threshold/(num_classes) if x>=threshold else (num_classes-threshold)/(num_classes), train_labels))
                probs = np.array(probs)
                probs /= np.sum(probs)
                indices = np.random.choice(range(len(train_images)), size=int(total_samples), replace=False, p=probs)
            elif oversampling2:
                indices = np.random.choice(range(len(train_images)), size=int(total_samples), replace=False)
                indices = sum([[i]*(num_classes-threshold) for i in indices if train_labels[i]<threshold] \
                    + [[i]*threshold for i in indices if train_labels[i]>=threshold],[])

                m*=int((2*(num_classes-threshold)*threshold/(num_classes)))
            else:
                indices = np.random.choice(range(len(train_images)), size=int(total_samples), replace=False)
        else:
            indices = np.arange(int(total_samples))
        # print(indices)

        #if network == "nasnet":
        #    train_images = (train_images[indices,:,:,:]).astype(np.float32) #NHWC
        #    train_images = keras_applications.nasnet.preprocess_input(train_images, backend=tf.keras.backend)
        #    if training:
        #        test_images = keras_applications.nasnet.preprocess_input(test_images, backend=tf.keras.backend)
        #        train_labels = np.take(train_labels,indices)
        #        print(len([x for x in train_labels if x<threshold])/len(train_images))

        #elif network == "vgg19":
        #    train_images = (train_images[indices,:,:,:]).astype(np.float32) #NHWC
        #    train_images = keras_applications.vgg19.preprocess_input(train_images, backend=tf.keras.backend)
        #    if training:
        #        test_images = keras_applications.vgg19.preprocess_input(test_images, backend=tf.keras.backend)
        #        train_labels = np.take(train_labels,indices)
        #        print(len([x for x in train_labels if x<threshold])/len(train_images))

        #elif network == "vgg16":
        #    train_images = (train_images[indices,:,:,:]).astype(np.float32) #NHWC
        #    train_images = keras_applications.vgg16.preprocess_input(train_images, backend=tf.keras.backend)
        #    if training:
        #        test_images = keras_applications.vgg16.preprocess_input(test_images, backend=tf.keras.backend)
        #        train_labels = np.take(train_labels,indices)
        #        print(len([x for x in train_labels if x<threshold])/len(train_images))

        #elif network == "resnet50" or network == "resnet101" or network == "renset152":
        ## elif network == "resnet101" or network == "renset152":
        #    train_images = (train_images[indices,:,:,:]).astype(np.float32) #NHWC
        #    train_images = keras_applications.resnet.preprocess_input(train_images, backend=tf.keras.backend)
        #    # train_images = train_images/255.0
        #    # import matplotlib.pyplot as plt
        #    # # print(train_images)
        #    # plt.imshow(train_images[0])
        #    if training:
        #        test_images = keras_applications.resnet.preprocess_input(test_images, backend=tf.keras.backend)
        #        train_labels = np.take(train_labels,indices)
        #        # test_images = test_images/255.0
        #        print(len([x for x in train_labels if x<threshold])/len(train_images))

        #elif network in ["resnet_v2_50","resnetv2_101", "resnetv2_152"]:
        #    train_images = (train_images[indices,:,:,:]).astype(np.float32) #NHWC
        #    train_images = keras_applications.resnet_v2.preprocess_input(train_images, backend=tf.keras.backend)
        #    if training:
        #        test_images = keras_applications.resnet_v2.preprocess_input(test_images, backend=tf.keras.backend)
        #        train_labels = np.take(train_labels,indices)
        #        print(len([x for x in train_labels if x<threshold])/len(train_images))

        #else:
        if True:
            train_images = train_images[indices]
            if training:
                test_images = test_images
                train_labels = np.take(train_labels,indices)
                print(len([x for x in train_labels if x<threshold])/len(train_images))

            ##adding channel dimenions for image datasets without them
            if dataset in ["mnist","mnist-fashion","KMNIST","EMNIST"]:
                train_images = np.expand_dims(train_images,-1).astype(np.uint8)
                test_images = np.expand_dims(test_images,-1).astype(np.uint8)
            ## for non-flexible architectures, transform the data
                if network not in ["cnn","fc"]:
                    train_images = np.tile(train_images,(1,1,1,3))
                    test_images = np.tile(test_images,(1,1,1,3))
                    #print(train_images.dtype)
                    # plt.imshow(train_images[0])
                    # plt.show()
                    #print(train_images.shape)
            if network in ["cnn","fc"]:
                #normalize the images pixels to be in [0,1]
                train_images = train_images.astype(np.float32)/255.0
                if training:
                    test_images = test_images.astype(np.float32)/255.0
            else:
                #note that the transformation to PIL and back to Tensor normalizes the image pixels to be in [0,1]
                assert train_images.dtype == "uint8" #otherwise ToPILImage wants the input to be NCHW. wtff
                train_images = np.stack([d.transform(image) for image in train_images])
                train_images = np.transpose(train_images,(0,2,3,1)) # this is because the pytorch transform changes it to NCHW for some reason :P
                if unnormalized_images:
                    train_images = train_images*255.0
                if training:
                    test_images = np.stack([d.transform(image) for image in test_images])
                    test_images = np.transpose(test_images,(0,2,3,1))
                    if unnormalized_images:
                        test_images = test_images*255.0
                print(train_images.shape)
                print("max after transforming", train_images.max())

            #check correct dimensions
            if network != "fc":
                image_size = train_images.shape[1]
                assert train_images.shape[1] == train_images.shape[2]
                number_channels = train_images.shape[-1]

            flat_train_images = np.array([train_image.flatten() for train_image in train_images])
            if training:
                flat_test_images = np.array([test_image.flatten() for test_image in test_images])

            if channel_normalization:
                #flatten to compute SVD matrix
                print("channel normalizing")
                x = train_images
                flat_x = flat_train_images

                #normalize each channel (3 colors for e.g.)
                x_mean = np.mean(x, axis=(0,1,2))
                x_std = np.std(x, axis=(0,1,2))
                x = (x - x_mean) / x_std
                train_images = x

                #test images
                if training:
                    test_images = (test_images - x_mean) / x_std


                #flatten again after normalizing
                flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
                if training:
                    flat_test_images = np.array([test_image.flatten() for test_image in flat_test_images])


            if centering:
                #flatten to compute SVD matrix
                print("centering")
                x = train_images
                flat_x = flat_train_images
                flat_x -= flat_x.mean(axis=0)

                x = flat_x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

                #test images
                if training:
                    flat_test_images -= flat_test_images.mean(axis=0)
                    test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2],test_images.shape[3]))

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
                train_images = flat_x.reshape((x.shape[0], x.shape[3], x.shape[1],x.shape[2]))

                #test images
                if training:
                    flat_test_images -= flat_test_images.mean(axis=0)
                    flat_test_images = np.matmul(flat_test_images, Q.T)
                    test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2],test_images.shape[3]))
                    #test_images = flat_test_images.reshape((test_images.shape[0], test_images.shape[3], test_images.shape[1],test_images.shape[2]))
                    #test_images =  np.transpose(test_images, tp_order)

        #flattened images, as the kernel function takes flattened vectors (row major for NCHW images)
        tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
        if n_gpus>0:
            flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW
            flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
            if training:
                flat_test_images = np.transpose(test_images, tp_order)  # NHWC -> NCHW
                flat_test_images = np.array([test_image.flatten() for test_image in flat_test_images])


    if network == "fc":
        train_images = flat_train_images
        if training:
            test_images = flat_test_images

    #corrupting images, and adding confusion data
    def binarize(label, threshold,method="threshold"):
        if method=="threshold":
            return label>=threshold
        elif method=="oddeven":
            return (label+1)%2

    # %%
    def process_labels(label,label_corruption,threshold,zero_one=False,binarized=True, binarization_method="threshold"):
        if binarized:
            if zero_one:
                if np.random.rand() < label_corruption:
                    return np.random.choice([0,1])
                else:
                    return float(binarize(label,threshold,binarization_method))
            else:
                if np.random.rand() < label_corruption:
                    return np.random.choice([-1.0,1.0])
                else:
                    return float(binarize(label,threshold,binarization_method))*2.0-1
        else:
            if np.random.rand() < label_corruption:
                return np.random.choice(range(num_classes))
            else:
                return float(label)


    #if the labels are to be generated by a neural network:
    if doing_regression:
        from utils import load_model
        model = load_model(FLAGS)
        ys = model.predict(train_images)[:,0]
        if training:
            test_ys = model.predict(test_images)[:,0]
    else:
        if nn_random_labels:
            from utils import load_model
            model = load_model(FLAGS)
            # data = tf.constant(train_images)
            train_labels = model.predict(train_images)[:,0]>0#, batch_size=data.shape[0], steps=1) > 0
            # print("generated function", "".join([str(int(y)) for y in train_labels]))
            if training:
                # data = tf.constant(test_images)
                test_labels = model.predict(test_images)[:,0]>0#, batch_size=data.shape[0], steps=1) > 0
            if binarized:
                threshold=1
                num_classes = 2

        if random_labels:
            print("zero_one", zero_one)
            ys = [process_labels(label,label_corruption,threshold,zero_one=True,binarized=binarized, binarization_method=binarization_method) for label in train_labels[:m]] + [process_labels(label,1.0,threshold,zero_one=True,binarized=binarized, binarization_method=binarization_method) for label in train_labels[m:]]
        else: #confusion/attack labels
            ys = [process_labels(label,label_corruption,threshold,zero_one=True,binarized=binarized, binarization_method=binarization_method) for label in train_labels[:m]] + [float(not binarize(label,threshold, binarization_method=binarization_method)) for label in train_labels[m:]]

        if training:
            test_ys = np.array([process_labels(label,label_corruption,threshold,zero_one=True,binarized=binarized, binarization_method=binarization_method) for label in test_labels])

    '''SAVING DATA SAMPLES'''
    if training:
        save_data(train_images,ys,test_images,test_ys,FLAGS)
    else:
        test_images = test_ys = []
        save_data(train_images,ys,test_images,test_ys,FLAGS)


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('out_of_sample_test_error', True, "Whether to test only on inputs outside of training data, or on whole dataset")
    f.DEFINE_boolean('unnormalized_images', False, "Whether to have the images in range [0,255.0], rather than the standard [0,1]")
    #f.DEFINE_boolean('extended_test_set', True, "Whether to extend the test set by the part of the training set not in the sample")
    f.DEFINE_boolean('random_training_set', True, "Whether to make the training set by sampling random instances from the full training set of the dataset, rather than just taking the m initial samples. Only implemented for images datasets ")
    f.DEFINE_string('booltrain_set', None, "when using the Boolean dataset option, you can provide the training set, encoded as a binary string (1 if input is to be included in training set, 0 otherwise), rather than randomly sampling one")
    f.DEFINE_string('binarization_method', "threshold", "the method to binarize the labels. At the moment we have implemented:  with a threshold, and by their parity (odd/even)")

    tf.compat.v1.app.run()
    #tf.app.run()
    import gc; gc.collect()
