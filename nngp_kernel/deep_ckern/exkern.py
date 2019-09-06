from gpflow.params import Parameterized
import tensorflow as tf
import numpy as np

__all__ = ['ElementwiseExKern', 'ExReLU', 'ExErf']


class ElementwiseExKern(Parameterized):
    def K(self, cov, var1, var2=None):
        raise NotImplementedError

    def Kdiag(self, var):
        raise NotImplementedError

    def nlin(self, x):
        """
        The nonlinearity that this is computing the expected inner product of.
        Used for testing.
        """
        raise NotImplementedError


class ExReLU(ElementwiseExKern):
    def __init__(self, exponent=1, multiply_by_sqrt2=False, name=None):
        super(ExReLU, self).__init__(name=name)
        self.multiply_by_sqrt2 = multiply_by_sqrt2
        if exponent in {0, 1}:
            self.exponent = exponent
        else:
            raise NotImplementedError

    def K(self, cov, var1, var2=None):
        """
        cov: (N1*N2) x Dim1 x Dim2 x ...
        var1: N1 x Dim1 x Dim2 x ...
        var2: N2 x Dim1 x Dim2 x ...
        """
        if var2 is None:
            rsqrt1 = rsqrt2 = tf.rsqrt(var1)
            var2 = var1
        else:
            rsqrt1, rsqrt2 = tf.rsqrt(var1), tf.rsqrt(var2)

        inv_norms_prod = rsqrt1[:, None, ...] * rsqrt2  # N1 x N2 x ...
        shape_cov = tf.shape(cov)  # (N1*N2) x ...
        cos_theta = cov * tf.reshape(inv_norms_prod, shape_cov)
        theta = tf.acos(tf.clip_by_value(cos_theta, -1., 1., name="prevent_infty"))

        if self.exponent == 0:
            return .5 - theta/(2*np.pi)

        """
        (if on Emacs: use M-x org-toggle-latex-fragment and enjoy!)
        We need to calculate (cov, var1, var2 == $c, v_1, v_2$):
        $$ \frac{\sqrt{v_1 v_2}}{2\pi} \left( \sqrt{1 - \frac{c^2}{v_1 v_2}} + (\pi - \theta) \frac{c}{\sqrt{v_1}\sqrt{v_2}} \right) $$
        which is equivalent to:
        $$ \frac{1}{2\pi} \left( \sqrt{v_1 v_2 - c^2} + (\pi - \theta) c\right) $$
        """
        sq_norms_prod = tf.reshape(var1[:, None, ...] * var2, shape_cov)
        a = tf.nn.relu(sq_norms_prod - tf.square(cov), name="keep_positive")
        sin_theta_vv = tf.sqrt(a)
        J_vv = sin_theta_vv + (np.pi - theta) * cov
        if self.multiply_by_sqrt2:
            div = np.pi
        else:
            div = 2*np.pi
        return J_vv / div

    def Kdiag(self, var):
        if self.multiply_by_sqrt2:
            if self.exponent == 0:
                return tf.ones_like(var)
            else:
                return var
        else:
            if self.exponent == 0:
                return tf.ones_like(var)/2
            else:
                return var/2

    def nlin(self, x):
        if self.multiply_by_sqrt2:
            if self.exponent == 0:
                return ((1 + tf.sign(x))/np.sqrt(2))
            elif self.exponent == 1:
                return tf.nn.relu(x) * np.sqrt(2)
        else:
            if self.exponent == 0:
                return ((1 + tf.sign(x))/2)
            elif self.exponent == 1:
                return tf.nn.relu(x)


class ExErf(ElementwiseExKern):
    """The Gaussian error function as a nonlinearity. It's very similar to the
    tanh. Williams 1997"""
    def K(self, cov, var1, var2=None):
        if var2 is None:
            t1 = t2 = 1+2*var1
        else:
            t1, t2 = 1+2*var1, 1+2*var2
        vs = tf.reshape(t1[:, None, ...] * t2, tf.shape(cov))
        sin_theta = 2*cov / tf.sqrt(vs)
        return (2/np.pi) * tf.asin(sin_theta)

    def Kdiag(self, var):
        v2 = 2*var
        return (2/np.pi) * tf.asin(v2 / (1 + v2))

    def nlin(self, x):
        return tf.erf(x)
