import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian

def empirical_NTK(model,train_images):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    model.compile("sgd",loss=lambda target, pred: pred)
    import tensorflow.keras.backend as K
    num_layers = len(model.trainable_weights)
    trainable_weights = np.array(model.trainable_weights)

    # fs = []
    # params_per_chunk = []

    num_chunks = min(size, num_layers)
    layers_per_chunk = num_layers//num_chunks
    X = train_images
    NTK = np.zeros((len(X),len(X)))
    if rank < num_chunks:
        chunks = list(range(int(rank*layers_per_chunk),int((rank+1)*layers_per_chunk)))

        if rank < num_layers%num_chunks:
            chunks.append(num_chunks*layers_per_chunk+rank)
        params_per_layer = np.array([np.prod(x.shape) for x in trainable_weights])
        params_per_chunk = sum(params_per_layer[chunks])
        grads = model.optimizer.get_gradients(model.output, list(trainable_weights[chunks]))
        # grads = tf.keras.backend.gradients(model.output, list(trainable_weights[chunks]))
        # grads = tf.gradients(model.output, list(trainable_weights[chunks]))
        # symb_inputs = (model._feed_inputs + model._feed_targets)

        symb_inputs = model._feed_inputs
        #grads = jacobian(model.output,list(trainable_weights[chunks]))
        f = K.function(symb_inputs, grads)

        def get_weight_grad(model, inputs, outputs):
            """ Gets gradient of model for given inputs and outputs for all weights"""
            x, y, _= model._standardize_user_data(inputs, outputs)
            batch_size = inputs.shape[0]
            output_grad = f(x + y)
            #output_grad = f(x)
            #print(output_grad[0].shape)
            output_grad = np.concatenate([x.flatten() for x in output_grad])
            #output_grad = np.concatenate([x.reshape((batch_size,-1)) for x in output_grad])
            return output_grad

        X = train_images
        m = len(X)
        Y = np.zeros((len(X),1))
        NTK = np.zeros((len(X),len(X)))
        chunk1 = 25
        chunk2 = chunk1
        # it's benefitial to chunk in j2 too, in orden to reduce the python for loop. Even though we do more on numpy/pytorch (by reducing the chunking on j1, we do more grad computaiotns), python is much slower than those, and so tradeoff is worth it I think
        # print("tot_parameters",tot_parameters)
        jac1 = np.zeros((chunk1,params_per_chunk))
        jac2 = np.zeros((chunk2,params_per_chunk))
        num_chunk1s = m//chunk1
        if m%chunk1 > 0:
            num_chunk1s +=1
        num_chunk2s = m//chunk2
        if m%chunk2 > 0:
            num_chunk2s +=1
        for j1 in range(num_chunk1s):
            if m%chunk1>0 and j1 == num_chunk1s-1:
                num_inputs1 = m%chunk1
                jac1 = np.zeros((num_inputs1,params_per_chunk))
            else:
                num_inputs1 = chunk1
            print("chunk",j1,"out of",num_chunk1s)
            sys.stdout.flush()
            for i in range(num_inputs1):
                gradient = get_weight_grad(model, train_images[j1*chunk1+i:j1*chunk1+i+1], Y[j1*chunk1+i:j1*chunk1+i+1])
                jac1[i,:] = gradient
            #jac1 = get_weight_grad(model, train_images[j1*chunk1:j1*chunk1+num_inputs1], Y[j1*chunk1:j1*chunk1+num_inputs1])
            print(jac1.shape)
            jac2 = np.zeros((chunk2,params_per_chunk))
            for j2 in range(j1,num_chunk2s):
                if m%chunk2>0 and j2 == num_chunk2s-1:
                    num_inputs2 = m%chunk2
                    jac2 = np.zeros((num_inputs2,params_per_chunk))
                else:
                    num_inputs2 = chunk2
                print(j1,j2)
                print(num_inputs2)
                for i in range(num_inputs2):
                    gradient = get_weight_grad(model, train_images[j2*chunk2+i:j2*chunk2+i+1], Y[j2*chunk2+i:j2*chunk2+i+1])
                    jac2[i,:] = gradient
                #jac2 = get_weight_grad(model, train_images[j2*chunk2:j2*chunk2+num_inputs2], Y[j2*chunk2:j2*chunk2+num_inputs2])
                NTK[j1*chunk1:j1*chunk1+num_inputs1,j2*chunk2:j2*chunk2+num_inputs2] += np.matmul(jac1,jac2.T)

    ntk_recv = None
    if rank == 0:
        ntk_recv = np.zeros_like(NTK)
    comm.Reduce(NTK, ntk_recv, op=MPI.SUM, root=0)
    if rank == 0:
        NTK = (ntk_recv+ntk_recv.T)/2
        return NTK
