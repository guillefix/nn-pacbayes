import numpy as np

def empirical_NTK(model,train_images):
    model.compile("sgd",loss=lambda target, pred: pred)
    import tensorflow.keras.backend as K
    num_layers = len(model.trainable_weights)
    trainable_weights = model.trainable_weights

    grads = model.optimizer.get_gradients(model.total_loss, trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets)
    f = K.function(symb_inputs, grads)

    def get_weight_grad(model, inputs, outputs):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        x, y, _= model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y)
        output_grad = np.concatenate([x.flatten() for x in output_grad])
        return output_grad

    params_per_layer = [np.prod(x.shape) for x in trainable_weights]
    tot_parameters = np.sum(params_per_layer)

    X = train_images
    Y = np.zeros((len(X),1))
    NTK = np.zeros((len(X),len(X)))
    chunk1 = 25
    chunk2 = 25 # it's benefitial to chunk in j2 too, in orden to reduce the python for loop. Even though we do more on numpy/pytorch (by reducing the chunking on j1, we do more grad computaiotns), python is much slower than those, and so tradeoff is worth it I think
    print("tot_parameters",tot_parameters)
    jac1 = np.zeros((chunk1,tot_parameters))
    jac2 = np.zeros((chunk2,tot_parameters))
    for j1 in range(len(X)//chunk1):
        print("chunk",j1,"out of",len(X)//chunk1)
        grad_features = []
        for i in range(chunk1):
            # print(i)
            gradient = get_weight_grad(model, train_images[j1*chunk1+i:j1*chunk1+i+1], Y[j1*chunk1+i:j1*chunk1+i+1])
            jac1[i,:] = gradient
        for j2 in range(j1,len(X)//chunk2):
            grad_features = []
            # print(j1,j2)
            for i in range(chunk2):
                gradient = get_weight_grad(model, train_images[j2*chunk2+i:j2*chunk2+i+1], Y[j2*chunk2+i:j2*chunk2+i+1])
                jac2[i,:] = gradient
            NTK[j1*chunk1:(j1+1)*chunk1,j2*chunk2:(j2+1)*chunk2] += np.matmul(jac1,jac2.T)

    NTK = (NTK+NTK.T)/2

    # filename = net+"_KMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_NTK.npy"
    # np.save(filename, NTK)
    return NTK
