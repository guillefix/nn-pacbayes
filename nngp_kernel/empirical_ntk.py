import numpy as np
import sys

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
    if rank < num_chunks:
        chunks = list(range(int(rank*layers_per_chunk),int((rank+1)*layers_per_chunk)))

        if rank < num_layers%num_chunks:
            chunks.append(num_chunks*layers_per_chunk+rank)
        params_per_layer = np.array([np.prod(x.shape) for x in trainable_weights])
        params_per_chunk = sum(params_per_layer[chunks])
        grads = model.optimizer.get_gradients(model.total_loss, list(trainable_weights[chunks]))
        symb_inputs = (model._feed_inputs + model._feed_targets)
        f = K.function(symb_inputs, grads)

        def get_weight_grad(model, inputs, outputs):
            """ Gets gradient of model for given inputs and outputs for all weights"""
            x, y, _= model._standardize_user_data(inputs, outputs)
            output_grad = f(x + y)
            output_grad = np.concatenate([x.flatten() for x in output_grad])
            return output_grad

        X = train_images
        Y = np.zeros((len(X),1))
        NTK = np.zeros((len(X),len(X)))
        chunk1 = 25
        chunk2 = 25 # it's benefitial to chunk in j2 too, in orden to reduce the python for loop. Even though we do more on numpy/pytorch (by reducing the chunking on j1, we do more grad computaiotns), python is much slower than those, and so tradeoff is worth it I think
        # print("tot_parameters",tot_parameters)
        jac1 = np.zeros((chunk1,params_per_chunk))
        jac2 = np.zeros((chunk2,params_per_chunk))
        for j1 in range(len(X)//chunk1):
            print("chunk",j1,"out of",len(X)//chunk1)
            sys.stdout.flush()
            for i in range(chunk1):
                gradient = get_weight_grad(model, train_images[j1*chunk1+i:j1*chunk1+i+1], Y[j1*chunk1+i:j1*chunk1+i+1])
                jac1[i,:] = gradient
            for j2 in range(j1,len(X)//chunk2):
                print(j1,j2)
                for i in range(chunk2):
                    gradient = get_weight_grad(model, train_images[j2*chunk2+i:j2*chunk2+i+1], Y[j2*chunk2+i:j2*chunk2+i+1])
                    jac2[i,:] = gradient
                NTK[j1*chunk1:(j1+1)*chunk1,j2*chunk2:(j2+1)*chunk2] += np.matmul(jac1,jac2.T)

    ntk_recv = None
    if rank == 0:
        ntk_recv = np.zeros_like(NTK)
    comm.Reduce(NTK, ntk_recv, op=MPI.SUM, root=0)
    if rank == 0:
        NTK = (ntk_recv+ntk_recv.T)/2
        return NTK
