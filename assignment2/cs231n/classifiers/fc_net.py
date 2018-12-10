from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################

        self.params['W1'] = weight_scale * \
            np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_scale * \
            np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        # Reshape the image data into rows
        # X_rows = X.reshape(X.shape[0], -1)

        # Get parameters
        # W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        # N = X_rows.shape[0]

        # Calculate scores
        # scores_hidden_layer = np.maximum(0, X_rows.dot(W1) + b1)
        # scores = scores_hidden_layer.dot(W2) + b2

        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        reg = self.reg

        # Use functions from layer_utils.py
        out, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(out, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # scores_correct = scores[range(N), y]

        # margin = -np.log(np.exp(scores_correct) / np.exp(scores).sum(axis=1))
        # loss = margin.mean() + 0.5 * self.reg * np.sum(W1 * W1) + \
        #     0.5 * self.reg * np.sum(W2 * W2)

        # # --------------------------------------------------------------------------

        # vertical_sum_of_e_to_fj = np.exp(scores).sum(axis=1, keepdims=True)

        # dW2_factors = np.exp(scores) / vertical_sum_of_e_to_fj
        # dW2_factors[range(N), y] -= 1

        # dW2 = scores_hidden_layer.T.dot(dW2_factors)
        # dW2 /= N
        # dW2 += self.reg * W2

        # grads['W2'] = dW2
        # grads['b2'] = dW2_factors.mean(axis=0)

        # # --------------------------------------------------------------------------

        # dW1_factors_after_ReLU = dW2_factors.dot(W2.T)

        # dW1_factors_before_ReLU = dW1_factors_after_ReLU.copy()
        # dW1_factors_before_ReLU[scores_hidden_layer <= 0] *= 0
        # dW1_factors_before_ReLU[scores_hidden_layer > 0] *= 1

        # # ---------------------------------------------------------------------------

        # dW1 = X.T.dot(dW1_factors_before_ReLU)
        # dW1 /= N
        # dW1 += self.reg * W1
        # grads['W1'] = dW1
        # grads['b1'] = dW1_factors_before_ReLU.mean(axis=0)

        # Use functions from layers.py
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)

        dx2, dw2, db2 = affine_backward(dout, cache2)

        grads['W2'] = dw2 + reg * W2
        grads['b2'] = db2

        dx1, dw1, db1 = affine_relu_backward(dx2, cache1)
        grads['W1'] = dw1 + reg * W1
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        # 0 hidden layers
        if (len(hidden_dims) == 0):

            self.params['W1'] = weight_scale * \
                np.random.randn(input_dim, num_classes)
            self.params['b1'] = np.zeros(num_classes)

        # 1 or more hidden layers
        else:

            curr_layer = input_dim
            next_layer = hidden_dims[0]

            for idx_hidden_layer in range(len(hidden_dims)):

                idx = str(idx_hidden_layer + 1)

                self.params['W' + idx] = weight_scale * \
                    np.random.randn(curr_layer, next_layer)
                self.params['b' + idx] = np.zeros(next_layer)

                # batchnorm part
                if use_batchnorm:
                    self.params['gamma' + idx] = np.ones((1, 1))
                    self.params['beta' + idx] = np.zeros((1, 1))

                # if not end
                if idx_hidden_layer != (len(hidden_dims) - 1):

                    curr_layer = hidden_dims[idx_hidden_layer]
                    next_layer = hidden_dims[idx_hidden_layer + 1]

            # The last one
            last_idx = str(self.num_layers)
            last_dim = hidden_dims[-1]

            # An extra layer
            self.params['W' + last_idx] = weight_scale * \
                np.random.randn(last_dim, last_dim)
            self.params['b' + last_idx] = np.zeros(last_dim)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    # Self defined functions==============================================
    def affine_batchnorm_forward(self, x, w, b, gamma, beta, bn_param):
        a, fc_cache = affine_forward(x, w, b)
        c, batch_cache = batchnorm_forward(a, gamma, beta, bn_param)
        out, relu_cache = relu_forward(c)
        cache = (fc_cache, batch_cache, relu_cache)
        return out, cache

    def affine_batchnorm_backward(self, dout, cache):
        fc_cache, batch_cache, relu_cache = cache
        da = relu_backward(dout, relu_cache)
        dc, dgamma, dbeta = batchnorm_backward_alt(da, batch_cache)
        dx, dw, db = affine_backward(dc, fc_cache)
        return dx, dw, db, np.sum(dgamma), np.sum(dbeta)

    def affine_relu_drop_forward(self, x, w, b, dropout_param):
        a, ar_cache = affine_relu_forward(x, w, b)
        out, drop_cache = dropout_forward(a, dropout_param)
        cache = (ar_cache, drop_cache)
        return out, cache

    def affine_relu_drop_backward(self, dout, cache):
        ar_cache, drop_cache = cache
        da = dropout_backward(dout, drop_cache)
        dx, dw, db = affine_relu_backward(da, ar_cache)
        return dx, dw, db

    # Self defined functions==============================================

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        params = self.params
        cache_list = []
        scores = X.copy()

        for idx_layer in range(self.num_layers):

            idx = str(idx_layer + 1)

            w = params['W' + idx]
            b = params['b' + idx]

            # Last layer
            if idx_layer + 1 == self.num_layers:
                scores, cache = affine_forward(scores, w, b)

            # Other layers
            else:
                if self.use_batchnorm:
                    gamma = params['gamma' + idx]
                    beta = params['beta' + idx]
                    bn_param = self.bn_params[idx_layer]

                    scores, cache = self.affine_batchnorm_forward(
                        scores, w, b, gamma, beta, bn_param)

                elif self.use_dropout:
                    scores, cache = self.affine_relu_drop_forward(
                        scores, w, b, self.dropout_param)

                else:
                    scores, cache = affine_relu_forward(scores, w, b)

            cache_list.append(cache)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        reg = self.reg
        loss, dout = softmax_loss(scores, y)

        # Add regs
        for idx_layer in range(self.num_layers):

            idx = str(idx_layer + 1)
            w = params['W' + idx]
            b = params['b' + idx]
            loss += 0.5 * reg * np.sum(w * w)

        # Gradients
        dx = dout.copy()

        for idx_layer in reversed(range(self.num_layers)):

            idx = str(idx_layer + 1)
            w = params['W' + idx]
            cache = cache_list[idx_layer]

            # Last layer
            if idx_layer + 1 == self.num_layers:
                dx, dw, db = affine_backward(dx, cache)

            # Other layers
            else:
                if self.use_batchnorm:
                    dx, dw, db, dgamma, dbeta = self.affine_batchnorm_backward(
                        dx, cache)
                    grads['gamma' + idx] = dgamma
                    grads['beta' + idx] = dbeta

                elif self.use_dropout:
                    dx, dw, db = self.affine_relu_drop_backward(dx, cache)

                else:
                    dx, dw, db = affine_relu_backward(dx, cache)

            grads['W' + idx] = dw + reg * w
            grads['b' + idx] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
