from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################

    x_reshaped = x.reshape(x.shape[0], -1)
    out = np.dot(x_reshaped, w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################

    x_reshaped = x.reshape(x.shape[0], -1)
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_reshaped.T.dot(dout)
    db = dout.sum(axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    dx = dout * np.heaviside(x,0) # can be implemented with: dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    num = y.shape[0]

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################

    x_fixed = x - np.max(x, axis=1, keepdims=True)
    softmax = np.exp(x_fixed)
    softmax /= np.sum(softmax, axis=1, keepdims=True)

    y_hat = np.log(softmax[range(num), y])  # extracting only the indices we want
    loss = - np.sum(y_hat) / num

    softmax[range(num), y] -= 1
    dx = softmax / num

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        x_std = (x_var + eps)**0.5
        x_normed = (x - x_mean) / x_std
        out = gamma * x_normed + beta # we multiply  every sample in x with the same gamma and beta
        cache = x, x_mean, x_var, x_std, x_normed, gamma
        
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        x_normed = (x - running_mean) / (running_var + eps)**0.5
        out = gamma * x_normed + beta

    #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, x_mean, x_var, x_std, x_normed, gamma = cache
    N, D = x.shape

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_normed, axis=0)

    # dx computation

    # dx_normed computation
    dx_normed = dout * gamma  # from every part of the computation

    # dvar computation
    dvar = (x - x_mean) * np.sum(dx_normed * (x - x_mean), axis=0) / (N * x_std ** 3)

    # dmean computation
    dmean = np.sum(dx_normed, axis=0) / (N * x_std)


    # total dx
    dx = dx_normed / x_std - dmean - dvar


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    x, x_mean, x_var, x_std, x_normed, gamma = cache

    #sacle and shift derivatives staying the same
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_normed, axis=0)

    #dx calculation
    dx = dout * gamma / x_std
    dx = dx - np.sum(dx,axis=0) * (len(dout)**-1) - x_normed * np.sum(dx * x_normed, axis=0) * (len(dout)**-1)

    # using len(dout) because it gets the first dimension and in bn we need to divide by N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################

    x_mean = np.mean(x, axis=1, keepdims=True)
    x_var = np.var(x, axis=1, keepdims=True)
    x_std = (x_var + eps) ** 0.5
    x_normed = (x - x_mean) / x_std
    out = gamma * x_normed + beta  # we multiply  every sample in x with the same gamma and beta
    cache = x, x_mean, x_var, x_std, x_normed, gamma

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    x, x_mean, x_var, x_std, x_normed, gamma = cache
    N, D = x.shape

    #sacle and shift derivatives staying the same
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_normed, axis=0)

    #dx calculation
    dx = dout * gamma / x_std
    dx = dx - np.sum(dx, axis=1, keepdims=True) * (D**-1) - x_normed * np.sum(dx * x_normed, axis=1, keepdims=True) * (D**-1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.


    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        # the mask gets 1 with probability p and 0 with 1-p and with the same shape as x,
        # so when multiply with x we get the desired dropout and do scaling with p to save the same
        # so save the same magnitude of the sum

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = mask * dout

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # init sizes
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    # output expceted sizes
    h_output = int(1 + (H + 2 * pad - HH) / stride)
    w_output = int(1 + (W + 2 * pad - WW) / stride)

    # padding the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    temp_out = np.zeros((N, F, h_output, w_output))

    for filter in range(F):
        for i in range(h_output):
            for j in range(w_output):

                # adding striding for each axis - taking the right window
                h_start = i * stride
                h_end = h_start + HH

                w_start = j * stride
                w_end = w_start + WW

                # taking only the wanted slice of x
                x_sliced = x_padded[:,:, h_start:h_end, w_start:w_end]

                temp_out[:,filter,i,j] = np.sum(w[filter,:,:,:] * x_sliced, axis=(1,2,3)) + b[filter]

    out = temp_out

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    N, F, h_output, w_output = dout.shape


    # dx needs to account for padding, so we create a padded buffer first
    dx_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')


    # we iterate over every point computed in the forward pass
    for n in range(N):
        for f in range(F):
            for i in range(h_output):
                for j in range(w_output):


                    h_start = i * stride
                    h_end = h_start + HH

                    w_start = j * stride
                    w_end = w_start + WW

                    # the gradient from above for this specific pixel
                    d_pixel = dout[n, f, i, j]

                    # filter Gradient (dw)
                    dw[f] += x_padded[n, :, h_start:h_end, w_start:w_end] * d_pixel

                    # input Gradient (dx)
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * d_pixel

    if pad == 0:
        dx = dx_padded
    else:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]

    # Calculate db efficiently by summing over (N, H, W)
    db = np.sum(dout, axis=(0, 2, 3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']


    h_output = int(1 + (H - pool_height) / stride)
    w_output = int(1 + (W - pool_width) / stride)

    # init
    out = np.zeros((N, C, h_output, w_output))

    for n in range(N):  # for each image
        for c in range(C):  # for each channel
            for i in range(h_output):  # slide vertically
                for j in range(w_output):  # slide horizontally

                    # limits of current window
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # extract the window from input x
                    x_slice = x[n, c, h_start:h_end, w_start:w_end]

                    # perform Max Pooling
                    out[n, c, i, j] = np.max(x_slice)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, h_output, w_output = dout.shape

    # init
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(h_output):
                for j in range(w_output):


                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    x_slice = x[n, c, h_start:h_end, w_start:w_end]
                    dout_val = dout[n, c, i, j]

                    # create a mask of the max value
                    # example: x_slice = [[1, 9], [2, 0]] -> mask = [[False, True], [False, False]]
                    mask = (x_slice == np.max(x_slice))

                    # route the gradient ONLY to the max value(s)
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout_val
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = x.shape

    # set x to the right shape for bn, it expects x.shape = N,D
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)

    # set the output after reshape back to the right shape
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = dout.shape

    # same process as the forward with the reshape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)

    # reshape back
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################

    N, C, H, W = x.shape

    # reshape
    x_reshaped = x.reshape(N, G, C // G, H, W)

    # calculate statistics
    x_mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)

    # normalization
    std = np.sqrt(x_var + eps)
    x_normed_reshaped = (x_reshaped - x_mean) / std

    # reshape back to apply scale and shift
    x_normed = x_normed_reshaped.reshape(N, C, H, W)
    out = gamma * x_normed + beta

    cache = (x_normed, x_normed_reshaped, std, gamma, G)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################

    x_normed, x_norm_reshaped, std, gamma, G = cache
    N, C, H, W = dout.shape

    # summing in shape that will save the channel shape
    dgamma = np.sum(dout * x_normed, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # like layer normalization
    dx_normed = dout * gamma
    dx_normed_reshaped = dx_normed.reshape(N, G, C // G, H, W)

    # acting on (C//G, H, W) shape as a batch
    term1 = dx_normed_reshaped
    term2 = np.mean(dx_normed_reshaped, axis=(2, 3, 4), keepdims=True)
    term3 = x_norm_reshaped * np.mean(dx_normed_reshaped * x_norm_reshaped, axis=(2, 3, 4), keepdims=True)

    dx_reshaped = (term1 - term2 - term3) / std

    # reshape back to the original format
    dx = dx_reshaped.reshape(N, C, H, W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
