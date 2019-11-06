import numpy as np

def get_all_layers(model):
    layers = []
    for layer in model.layers:
        #if isinstance(layer,tf.python.keras.engine.training.Model):
        if isinstance(layer,tf.keras.Model):
            layers += get_all_layers(layer)
        else:
            layers += [layer]
    return layers

def is_normalization_layer(l):
    #return isinstance(l,tf.python.keras.layers.normalization.BatchNormalization) or isinstance(l,tf.python.keras.layers.normalization.LayerNormalization)
    return isinstance(l,tf.keras.layers.BatchNormalization) or isinstance(l,tf.keras.layers.LayerNormalization)

from scipy.stats import truncnorm
def reset_weights(model, weights, are_norm,sigmaw,sigmab):
    #initial_weights = model.get_weights()
    def initialize_var(w, is_norm):
        if is_norm:
            return w
        else:
            shape = w.shape
            if len(shape) == 1:
                #return tf.random.normal(shape,stddev=sigmab).eval(session=sess)
                return np.random.normal(0,sigmab,shape)
            else:
                #return tf.random.normal(shape,stddev=1.0/np.sqrt(np.prod(shape[:-1]))).eval(session=sess)
                #return np.random.normal(0,1.0/np.sqrt(np.prod(shape[:-1])),shape)
                #return np.random.normal(0,sigmaw/np.sqrt(shape[-2]),shape) #assumes NHWC so that we divide by number of channels as in GP limit
                return (sigmaw/np.sqrt(np.prod(shape[:-1])))*truncnorm.rvs(-np.sqrt(2),np.sqrt(2),size=shape) #assumes NHWC so that we divide by number of channels as in GP limit, and also works for fully connected

    new_weights = [initialize_var(w,are_norm[i]) for i,w in enumerate(weights)]
    model.set_weights(new_weights)
    #[l.set_weights([initialize_var(w.shape) for w in l.get_weights()]) for l in layers]
    #for l in layers:
    #    #if is_normalization_layer(l):
    #    #    # new_weights += l.get_weights()
    #    #    pass
    #    #else:
    #    new_weights = [initialize_var(w.shape) for w in l.get_weights()]
    #    l.set_weights(new_weights)

def simple_reset_weights(model,sigmaw,sigmab):
    initial_weights = model.get_weights()
    def initialize_var(shape):
        if len(shape) == 1:
           #return tf.random.normal(shape).eval(session=sess)
           return np.random.normal(0,sigmab,shape)
        else:
            return np.random.normal(0,sigmaw/np.sqrt(np.prod(shape[:-1])),shape)
    new_weights = [initialize_var(w.shape) for w in initial_weights]
    model.set_weights(new_weights)
