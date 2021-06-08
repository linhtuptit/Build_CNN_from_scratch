import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    
    """
    Args: 
        im_train: set of images
        label_train: set of labels
        batch_size: size of the mini-batch for stochastic gradient descent
    Returns:
        mini_batch_x: cells that contain a set of image batches
        mini_batch_y: cells that contain a set of label batches
    """

    if(np.remainder(im_train.shape[1], batch_size) == 0):
        sz = int(im_train.shape[1] / batch_size)
    else:
        sz = int(math.floor(im_train.shape[1] / batch_size) + 1)

    mini_batch_x = [0] * sz
    mini_batch_y = [0] * sz

    m, n = im_train.shape
    trainSize = n
    labels_onehot = np.zeros([10, trainSize])

    for i in range(0, trainSize):
        labels_onehot[label_train[:,i], i] = 1
    
    p = np.random.permutation(trainSize)

    for i in range(0, int(trainSize/batch_size)):
        mini_batch_x[i] = im_train[:, p[i*batch_size:(i+1)*batch_size]]
        mini_batch_y[i] = labels_onehot[:, p[i*batch_size:(i+1)*batch_size]] 

    return mini_batch_x, mini_batch_y


def fc(x, w, b):

    """ 
    === Fully connected layer ===
    Args: 
        x: input vector of fully connected layer
        w: weight vector
        b: bias vector
    Returns:
        y: output of fully connected layer
    Description: 
        FC is linear transform of x, i.e. y = wx + b
    """

    y = w @ x + b
    return y


def fc_backward(dl_dy, x, w, b, y):

    """
    === Fully connected layer backward ===
    Args:
        dl_dy: loss derivative with respect to the output y
    Returns:
        dl_dx: loss derivative with respect to the input x
        dl_dw: loss derivative with respect to the weight w
        dl_db: loss derivative with respect to the bias b
    """

    dl_dw = np.dot(dl_dy, x.T)
    dl_db = dl_dy
    dl_dx = np.dot(w.T, dl_dy)
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):

    """
    Args:
        y_tilde: the prediction
        y: the ground truth lable
    Returns:
        l: the loss
        dl_dy: the loss derivative with respect to the prediction
    Descriptions:
        This function measures Euclidean distance l = ||y - y_hat||**2
    """

    l = (np.linalg.norm(y - y_tilde)**2)
    dl_dy = -(2 * (y - y_tilde))
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):

    """
    Args:
        x: the input to the soft-max
        y: the ground truth label
    Returns:
        l: the loss funtion
        dl_dy: loss derivative with respect to x
    Descriptions:
        This function measures cross-entropy between two distributions
        l = sum(ylog(y))
    """
    """
    sf = np.exp(x) / np.sum(np.exp(x))
    l = - np.dot(y.T, np.log(sf))
    """
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    sf = exp_x / exp_x.sum(axis=0)
    l = - np.dot(y.T, np.log(sf))
    dl_dy = (sf - y)
    return l, dl_dy


def relu(x):

    """
    Args:
        x: a general tensor, matrix and vector
    Returns:
        y: the output of ReLu with the same input size
    Descriptions:
        ReLu is an activation unit, defined by: y = max(0,x)
    """

    y = np.maximum(0, x)
    return y

def relu_backward(dl_dy, x, y):

    """
    Args:
        dl_dy: is the loss derivative with respect to the output y
    Returns:
        dl_dx: is the loss derivative with respect to the input x
    """

    """
    dl_dx = np.zeros([dl_dy.shape[0], 1])
    for i in range(0, dl_dy.shape[0]):
        if (y[i, 0] > 0):
            dl_dx[i, 0] = dl_dy[i, 0]
        else:
            dl_dx[i, 0] = 0
    """
    dl_dx = dl_dy * (y >= 0)
    return dl_dx


def im2col(x, block_size):

    """  
    Create a 2d matrix from a image ...
    ... on the format of a 3 dimension tensor [channels, rows, cols]
    Source: https://www.programmersought.com/article/41521899259/
    """

    x_shape = x.shape
    sx = x_shape[0] - block_size[0] + 1
    sy = x_shape[1] - block_size[1] + 1
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    for i in range(0, sy):
        for j in range(0, sx):
            result[:, i*sx+j] = x[j:j+block_size[0], i:i+block_size[1]].ravel(order='F')
    
    return result


def conv(x, w_conv, b_conv):

    """
    Args:
        x: an input to the convolutional operation
        w_conv: weights of the convolutional operation
        b_conv: bias of the convolutional operation
    Returns:
        y: the output of the convolutional operation
    Descriptions:
        Perform convolution operation
    """

    H, W = x.shape
    _, _, _, C2 = w_conv.shape
    y = np.zeros([H, W, C2])

    padding = 1
    X = np.pad(x, [padding, padding])
    X_transpose = im2col(X, [3, 3])
    w = np.reshape(w_conv, [9, 3])
    y_transpose = np.dot(X_transpose.T, w)
    y = np.reshape(y_transpose, [14, 14, 3])

    for iBias in range(0, C2):
        y[:,:,iBias] = y[:,:,iBias] + b_conv[iBias] 
    
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    
    """
    Args:
        dl_dy: the loss derivative with respec to y
        x: an input to the convolutional operation
        w_conv: weights of the convolutional operation
        b_conv: bias of the convolutional operation
    Returns:
        dl_dw: the loss derivatives with respect to weights
        dl_db: the loss derivatives with respect to bias
    """

    H, W = x.shape
    _, _, _, C2 = w_conv.shape
    padding = 1
    X = np.pad(x, [padding,padding])

    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    new_X = im2col(X, [3, 3])
    new_dl_dy = np.reshape(dl_dy, [196, 3])

    new_dl_dw = np.dot(new_X, new_dl_dy)
    dl_dw = np.reshape(new_dl_dw, [3, 3, 1, 3])

    for i in range(0, C2):
        dl_db[i, 0] = np.sum(np.sum(dl_dy[:,:,i]))

    return dl_dw, dl_db


def pool2x2(x):
    
    """
    Agrs:
        x: a general tensor and matrix
    Returns:
        y: output of the 2 × 2 max-pooling operation with stride 2
    Descriptions:
        Perform max-pooling operation
    """

    H, W, C = x.shape
    pool_stride = 2
    y = np.zeros([H // pool_stride, W // pool_stride, C])

    for i in range(0, H, pool_stride):
        for j in range(0, W, pool_stride):
            for k in range(0, C):
                pool_batch = [
                    x[i,j,k],
                    x[i+pool_stride-1,j,k],
                    x[i,j+pool_stride-1,k],
                    x[i+pool_stride-1,j+pool_stride-1,k]]
                y[(i+pool_stride-1)//2,(j+pool_stride-1)//2,k] = np.max(pool_batch)        
    
    return y


def pool2x2_backward(dl_dy, x, y):
    
    """
    Args: 
        dl_dy: is the loss derivative with respect to the output y
        x: a general tensor and matrix
        y: output of the 2 × 2 max-pooling operation with stride 2
    Returns:
        dl_dx: the loss derivative with respect to the input x
    """

    H, W, C = x.shape
    dl_dx = np.zeros([H, W, C])
    pool_stride = 2

    for i in range(0, H, pool_stride):
        for j in range(0, W, pool_stride):
            for k in range(0, C):
                idx = (i+pool_stride-1) // 2
                jdx = (j+pool_stride-1) // 2
                if(x[i,j,k] == y[idx,jdx,k]):
                    dl_dx[i,j,k] = dl_dy[idx,jdx,k]
                elif(x[idx,j,k] == y[idx,jdx,k]):
                    dl_dx[idx,j,k] = dl_dy[idx,jdx,k]
                elif(x[i,jdx,k] == y[idx,jdx,k]):
                    dl_dx[i,jdx,k] = dl_dy[idx,jdx,k]
                else:
                    dl_dx[idx,jdx,k] = dl_dy[idx,jdx,k]
    
    return dl_dx


def flattening(x):
    
    """
    Args:
        x: a tensor 
    Returns:
        y: the vectorized tensor
    """
    
    z = x.flatten()
    y = np.zeros([len(z), 1])
    for i in range(0, len(z)):
        y[i, 0] = z[i]
    
    return y


def flattening_backward(dl_dy, x, y):
    
    """
    Args: 
        dl_dy: the loss derivative with respect to the output y
        x: a tensor
        y: the vectorized tensor
    Returns:
        dl_dx: the loss derivative with respect to the input x
    """

    dl_dx = np.reshape(dl_dy, x.shape)
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    
    """
    Args:
        mini_batch_x:
        mini_batch_y: 
    Returns:
        w: the trained weights of single-layer perceptron
        b: the trained bias of single-layer perceptron
    Descriptions:
        functions to train a single-layer perceptron ...
        ... using a stochastic gradient descent method ...
        ... including following functions:
                    fc,
                    fc_backward,
                    loss_euclidean
    """

    w = np.random.normal(0, 1, [10, 196])
    b = np.random.normal(0, 1, [10, 1])

    batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.01
    decay_rate = 0.5
    decay_interval = 1000
    nIters = 10000

    k = 0
    loss_iter = 0
    for iter in range(0, nIters):
        if(np.remainder(iter, decay_interval) == 0):
            learning_rate = decay_rate * learning_rate
        dl_dW = 0
        dl_dB = 0

        for i in range(0, batch_size):
            x = mini_batch_x[k][:,i:i+1]
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_euclidean(y_tilde, mini_batch_y[k][:,i:i+1])
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            dl_dW = dl_dW + dl_dw
            dl_dB = dl_dB + dl_db 
        
        k = k+1
        if(k == batches):
            k = 0
        
        w = w - (learning_rate * dl_dW) / batch_size
        b = b - (learning_rate * dl_dB) / batch_size
    
    return w, b


def train_slp(mini_batch_x, mini_batch_y):

    """
    Args:
        mini_batch_x:
        mini_batch_y: 
    Returns:
        w: the trained weights of single-layer perceptron
        b: the trained bias of single-layer perceptron
    Descriptions:
        functions to train a single-layer perceptron with ...
        ... soft-max cross-entropy using a stochastic ...
        ... gradient descent method including following functions:
                    fc,
                    fc_backward,
                    loss_cross_entropy_softmax
    """

    w = np.random.normal(0, 1, [10, 196])
    b = np.random.normal(0, 1, [10, 1])

    batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.2
    decay_rate = 0.9
    decay_interval = 1000
    nIters = 10000

    k = 0
    loss_iter = 0
    for iter in range(0, nIters):
        if(np.remainder(iter, decay_interval) == 0):
            learning_rate = decay_rate * learning_rate
        dl_dW = 0
        dl_dB = 0

        for i in range(0, batch_size):
            x = mini_batch_x[k][:,i:i+1]
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, mini_batch_y[k][:,i:i+1])
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            dl_dW = dl_dW + dl_dw
            dl_dB = dl_dB + dl_db 
        
        k = k+1
        if(k == batches):
            k = 0
        
        w = w - (learning_rate * dl_dW) / batch_size
        b = b - (learning_rate * dl_dB) / batch_size
    
    return w, b


def train_mlp(mini_batch_x, mini_batch_y):
    
    """
    Args:
        mini_batch_x:
        mini_batch_y: 
    Returns:
        w1, w2: the trained weights of multi-layer perceptron
        b1, b2: the trained bias of multi-layer perceptron
    Descriptions:
        functions to train a multi-layer perceptron using a ... 
        ... stochastic gradient descent method including ...
        ... following functions:
                    fc,
                    fc_backward,
                    relu,
                    relu_backward,
                    loss_cross_entropy_softmax
    """

    w1 = np.random.uniform(0, 1, [30, 196])
    b1 = np.random.uniform(0, 1, [30, 1])
    w2 = np.random.uniform(0, 1, [10, 30])
    b2 = np.random.uniform(0, 1, [10, 1])

    batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.05
    decay_rate = 0.75
    decay_interval = 1000
    nIters = 10000

    k = 0
    for iter in range(0, nIters):
        if(np.remainder(iter, decay_interval) == 0):
            learning_rate = decay_rate * learning_rate
        
        dl_dW1 = 0
        dl_dB1 = 0
        dl_dW2 = 0
        dl_dB2 = 0

        for i in range(0, batch_size):
            x = mini_batch_x[k][:,i:i+1]
            a1 = fc(x, w1, b1)
            f1 = relu(a1)
            a2 = fc(f1, w2, b2)

            l, dl_dy = loss_cross_entropy_softmax(a2, mini_batch_y[k][:,i:i+1])
            dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dy, f1, w2, b2, a2)
            dl_dy1 = relu_backward(dl_dx2, a1, f1)
            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy1, x, w1, b1, a1)

            dl_dW1 = dl_dW1 + dl_dw1
            dl_dB1 = dl_dB1 + dl_db1
            dl_dW2 = dl_dW2 + dl_dw2
            dl_dB2 = dl_dB2 + dl_db2

        k = k + 1
        if(k == batches):
            k = 0
        
        w1 = w1 - (learning_rate * dl_dW1) / batch_size
        b1 = b1 - (learning_rate * dl_dB1) / batch_size
        w2 = w2 - (learning_rate * dl_dW2) / batch_size
        b2 = b2 - (learning_rate * dl_dB2) / batch_size
    
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    
    """
    Args:
        mini_batch_x:
        mini_batch_y: 
    Returns:
        w_conv, w_fc: the trained weights of CNN
        b_conv, b_fc: the trained bias of CNN
    Descriptions:
        functions to train a CNN using a stochastic gradient ...
        ... descent method including following functions:
                    conv,
                    coonv_backward,
                    pool2x2,
                    pool2x2_backward,
                    flattening,
                    flattening_backward,
                    fc,
                    fc_backward,
                    relu,
                    relu_backward,
                    loss_cross_entropy_softmax
    """

    w_conv = np.random.normal(0, 1, [3, 3, 1, 3])
    b_conv = np.random.normal(0, 1, [3, 1])
    w_fc = np.random.normal(0, 1, [10, 147])
    b_fc = np.random.normal(0, 1, [10, 1])

    batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]
    learning_rate = 0.1
    decay_rate = 0.8
    decay_interval = 1000
    nIters = 10000

    k = 0
    for iter in range(0, nIters):
        if(np.remainder(iter, decay_interval) == 0):
            learning_rate = decay_rate * learning_rate
        
        dldW_conv = 0
        dldB_conv = 0
        dldW_fc = 0
        dldB_fc = 0

        for i in range(0, batch_size):
            x = mini_batch_x[k][:,i:i+1]
            X = np.reshape(x, [14, 14])
            X_conv = conv(X, w_conv, b_conv)
            f = relu(X_conv)
            f_pooled = pool2x2(f)
            f_flattened = flattening(f_pooled)
            fc_layer = fc(f_flattened, w_fc, b_fc)
            l, dldy_fc_layer = loss_cross_entropy_softmax(fc_layer, mini_batch_y[k][:,i:i+1])

            dldf_flattened, dldw_fc, dldb_fc = fc_backward(dldy_fc_layer, f_flattened, w_fc, b_fc, fc_layer)
            dldf_pooled = flattening_backward(dldf_flattened, f_pooled, f_flattened)
            dldf = pool2x2_backward(dldf_pooled, f, f_pooled)
            dldf_conv_X = relu_backward(dldf, X_conv, f)
            dldw_conv, dldb_conv = conv_backward(dldf_conv_X, X, w_conv, b_conv, X_conv)
            
            dldW_conv = dldW_conv + dldw_conv
            dldB_conv = dldB_conv + dldb_conv
            dldW_fc = dldW_fc + dldw_fc
            dldB_fc = dldB_fc + dldb_fc

        k = k + 1
        if(k == batches):
            k = 0
        
        w_conv = w_conv - (learning_rate * dldW_conv) / batch_size
        b_conv = b_conv - (learning_rate * dldB_conv) / batch_size
        w_fc = w_fc - (learning_rate * dldW_fc) / batch_size
        b_fc = b_fc - (learning_rate * dldB_fc) / batch_size
    
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    #main.main_slp_linear()
    #main.main_slp()
    #main.main_mlp()
    main.main_cnn()



