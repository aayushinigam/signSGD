"""
FashionMNIST_signSGD - signSGD on Neural Nets
dataset - MNIST Fashion

"""
# importing libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle    
import os
from mpi4py import MPI
import functools
import time


comm = MPI.COMM_WORLD   # creating a communicator
size = comm.Get_size()  # getting no. of processes in this communicator
rank = comm.Get_rank()  # getting calling process rank in this communicator

# loading mnist Fashion dataset
(X_train, label_train), (X_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()

# obtaining required parameters
X_test_images = X_test
X_train_images = X_train
m_train = X_train.shape[0]
m_test = X_test.shape[0]
n_x = X_train.shape[1] * X_train.shape[2]

#Network structure
n_classes = 10
n = [n_x, 512, 256, 128, n_classes]
L = len(n)

# utility function to create one-hot encoding from labels and normalizing the data
def process_data(X, label):
    m = X.shape[0]
    assert(m == label.shape[0])
    X = X.reshape(m, -1).T / 255
    Y = np.zeros((m, 10))
    Y[np.arange(m), label] = 1
    Y = Y.T
    return X, Y

X_train, Y_train = process_data(X_train, label_train)
X_test, Y_test = process_data(X_test, label_test)

# The ReLu function
def relu(x):
    return np.maximum(0, x)

# The softmax function
def softmax(x):
    t = np.exp(x - np.max(x, axis = 0).reshape((1, x.shape[1])))
    return t / np.sum(t, axis = 0, keepdims = True)

# Implementation of forward propagation
def forward_prop(X, W, b, dropout_rate):
    m = X.shape[1]
    Z = [None]
    A = [X]
    D = [None]
    for l in range(1, L):
        Z.append(np.dot(W[l], A[l - 1]) + b[l])
        if (l == L - 1): A.append(softmax(Z[l]))
        else: 
            A.append(relu(Z[l]))  
            D.append(np.random.rand(A[l].shape[0], A[l].shape[1]) < dropout_rate)
            assert(D[l].shape == A[l].shape)
            A[l] = A[l] * D[l] / dropout_rate

        assert(Z[l].shape == (n[l], m))
        assert(A[l].shape == Z[l].shape)
    return Z, A, D

# Utility function, used in calculation cross_entropy_loss
def num_stable_prob(x, epsilon = 1e-18):
    x = np.maximum(x, epsilon)
    x = np.minimum(x, 1. - epsilon)
    return x

# Function to calculate cross_entropy_loss
def cross_entropy_loss(Yhat, Y, lbd, W):
    m = Y.shape[1]
    assert(m == Yhat.shape[1])
    num_stable_prob(Yhat)
    res = -np.squeeze(np.sum(Y * np.log(Yhat))) / m
    assert(res.shape == ())
    for l in range(1, L): res += lbd * np.sum(np.square(W[l])) / m / 2.
    return res

def relu_der(x):
    return np.int64(x > 0)

# Implementation for backward propagation
def backward_prop(X, Y, W, b, Z, A, D, dropout_rate, lbd):
    dZ = [None] * L
    dW = [None] * L
    db = [None] * L
    m = Y.shape[1]
    assert(X.shape[1] == m)
    for l in reversed(range(1, L)):
        if (l == L - 1): dZ[l] = A[l] - Y
        else:
             dA_l = np.dot(W[l + 1].T, dZ[l + 1])
             dA_l = dA_l * D[l] / dropout_rate 
             dZ[l] = dA_l * relu_der(Z[l])
        
        dW[l] = np.dot(dZ[l], A[l - 1].T) / m + (lbd * W[l]) / m
        db[l] = np.sum(dZ[l], axis = 1, keepdims = True) / m
        assert(dZ[l].shape == Z[l].shape)
        assert(dW[l].shape == W[l].shape)
        assert(db[l].shape == b[l].shape)
    return dW, db

# Making batches out of total training data
def split_batches(X, Y, batch_size):
    m = X.shape[1]
    assert(m == Y.shape[1])
    perm = list(np.random.permutation(m))

    shuffled_X = X[:, perm]
    shuffled_Y = Y[:, perm].reshape((n_classes, m))
    assert(shuffled_X.shape == X.shape)
    assert(shuffled_Y.shape == Y.shape)
    n_batches = m // batch_size
    batches = []
    for i in range(0, n_batches):
        batch_X = shuffled_X[:, i * batch_size : (i + 1) * batch_size]
        batch_Y = shuffled_Y[:, i * batch_size : (i + 1) * batch_size]
        batches.append((batch_X, batch_Y))
    if (m % batch_size != 0):
        batch_X = shuffled_X[:, batch_size * n_batches : m]
        batch_Y = shuffled_Y[:, batch_size * n_batches : m]
        batches.append((batch_X, batch_Y))
    return batches


# Updating weights in case of signSGD implementation
def update_para_signSGD(W, b, dW, db, alpha):
    W -= alpha * dW
    b -= alpha * db
    return W, b

# Updating weights in case of normal SGD implementation  
def update_para_SGD(W, b, dW, db, alpha):
    for l in range(1, L):
        W[l] -= alpha * dW[l]
        b[l] -= alpha * db[l]
    return W, b

# Implemenation for compressing the data
def compress(mat):
    m, n = mat.shape
    lst = []
    for i in range(0, m):
        res = ''
        for j in range(0,n):
            if(mat[i][j] == 1):
                res = res + '1'
            elif(mat[i][j] == 0):
                res = res + '0'
            else:
                res = res + '2'
        lst.append(res)
    return lst 

# Impementation for decompressing the data
def decompress(comp_mat, signs_mat):
    m,n = signs_mat.shape
    decomp_mat = np.zeros((m,n), dtype='i')
    i=0
    for words in comp_mat:
        j=0
        for c in words:
            if(c=='1'):
                decomp_mat[i][j] = 1
            elif(c=='0'):
                decomp_mat[i][j] = 0
            else:
                decomp_mat[i][j] = -1
            j += 1
        i += 1
    return decomp_mat

# Impelmentation for gradient descent - the main code which calls all other functions
def gradient_descent(W, b ,signSGD,lbd,dropout_rate,  n_epochs = 1, batch_size = 32,  learning_rate = .002):
    # Initializing buffers
    sendbuf_X = sendbuf_Y = None
    sendbuf_W = None
    recvbuf_W1 = None
    recvbuf_b1 = None
    recvbuf_W2 = None
    recvbuf_b2 = None
    recvbuf_W3 = None
    recvbuf_b3 = None
    recvbuf_W4 = None
    recvbuf_b4 = None
    # comp_s_W1 = comp_s_b1 = comp_s_W2 = comp_s_b2 = comp_s_W3 = comp_s_b3 = comp_s_W4 = comp_s_b4 = None
    batches = None
    for epoch_num in range(n_epochs):
        batches = split_batches(X_train, Y_train, batch_size)
        n_batches = len(batches)
        cost_list = []
        X = y = None
        for batch_idx in range(n_batches):
            X_cur, Y_cur = batches[batch_idx]

            # Parameter node splitting the training data 
            if(rank == 0):
                sendbuf_X = np.asarray(np.split(X_cur.T,size), dtype=float)
                sendbuf_Y = np.asarray(np.split(Y_cur.T,size), dtype=float)
            comm.Barrier()
            recvbuf_all_X = np.empty((X_cur.shape[1]//size,X_cur.shape[0]), dtype=float)
            recvbuf_all_Y = np.empty((Y_cur.shape[1]//size, Y_cur.shape[0]), dtype=float)
            
            # Parameter node scattering the data to all the worker nodes
            comm.Barrier()
            comm.Scatter(sendbuf_X, recvbuf_all_X, root=0)
            comm.Barrier()
            comm.Scatter(sendbuf_Y, recvbuf_all_Y, root=0)
            comm.Barrier()

            Z, A, D = forward_prop(recvbuf_all_X.T, W, b, dropout_rate)
            comm.Barrier()
            cost = cross_entropy_loss(A[L - 1], recvbuf_all_Y.T, lbd, W)
            cost_list.append(cost)
            iter_idx = epoch_num * n_batches + batch_idx + 1
            print("Cost after " + str(iter_idx) + " iterations: " + str(cost) + '.rank:',rank)
            comm.Barrier()
            dW, db = backward_prop(recvbuf_all_X.T, recvbuf_all_Y.T, W, b, Z, A, D, dropout_rate, lbd)
            comm.Barrier()

            # If normal SGD, updating weights
            if(not signSGD) : 
                comm.Barrier()
                update_para_SGD(W, b, dW, db, learning_rate)
                comm.Barrier()          
                if(iter_idx == 1000) :
                   return cost_list, W, b
                continue

            # For signSGD, calculating signs of the gradient
            for l in range(1,L):
                if(l==1):
                    signs_W1 = np.sign(dW[l])
                    signs_b1 = np.sign(db[l])
                if(l==2):
                    signs_W2 = np.sign(dW[l])
                    signs_b2 = np.sign(db[l])
                if(l==3):
                    signs_W3 = np.sign(dW[l])
                    signs_b3 = np.sign(db[l])
                if(l==4):
                    signs_W4 = np.sign(dW[l])
                    signs_b4 = np.sign(db[l])

            # Parameter node creating buffers for receiving signs from all the worker nodes    
            if(rank == 0):
                recvbuf_W1 = np.asarray([np.zeros_like(signs_W1)] * size)
                recvbuf_W2 = np.asarray([np.zeros_like(signs_W2)] * size)
                recvbuf_W3 = np.asarray([np.zeros_like(signs_W3)] * size)
                recvbuf_W4 = np.asarray([np.zeros_like(signs_W4)] * size)
                recvbuf_b1 = np.asarray([np.zeros_like(signs_b1)] * size)
                recvbuf_b2 = np.asarray([np.zeros_like(signs_b2)] * size)
                recvbuf_b3 = np.asarray([np.zeros_like(signs_b3)] * size)
                recvbuf_b4 = np.asarray([np.zeros_like(signs_b4)] * size)

            # Parameter node gathering gradient signs from all worker nodes
            comm.Barrier()
            comm.Gather(signs_W1, recvbuf_W1, root=0)
            comm.Barrier()
            comm.Gather(signs_b1, recvbuf_b1, root=0)
            comm.Barrier()
            
            comm.Gather(signs_W2, recvbuf_W2, root=0)
            comm.Barrier()
            comm.Gather(signs_b2, recvbuf_b2, root=0)
            comm.Barrier()

            comm.Gather(signs_W3, recvbuf_W3, root=0)
            comm.Barrier()
            comm.Gather(signs_b3, recvbuf_b3, root=0)
            comm.Barrier()

            comm.Gather(signs_W4, recvbuf_W4, root=0)
            comm.Barrier()
            comm.Gather(signs_b4, recvbuf_b4, root=0)
            comm.Barrier()

            # Parameter node performing majority voting from the received signs
            if(rank==0):
                s_W1 = np.sign(functools.reduce(np.add, recvbuf_W1)/size)
                s_b1 = np.sign(functools.reduce(np.add, recvbuf_b1)/size)
                s_W2 = np.sign(functools.reduce(np.add, recvbuf_W2)/size)
                s_b2 = np.sign(functools.reduce(np.add, recvbuf_b2)/size)
                s_W3 = np.sign(functools.reduce(np.add, recvbuf_W3)/size)
                s_b3 = np.sign(functools.reduce(np.add, recvbuf_b3)/size)
                s_W4 = np.sign(functools.reduce(np.add, recvbuf_W4)/size)
                s_b4 = np.sign(functools.reduce(np.add, recvbuf_b4)/size)
                
                # Compressing the resultant before broadcasting
                comp_s_W1 = np.asarray(compress(s_W1))
                comp_s_b1 = np.asarray(compress(s_b1))
                comp_s_W2 = np.asarray(compress(s_W2))
                comp_s_b2 = np.asarray(compress(s_b2))
                comp_s_W3 = np.asarray(compress(s_W3))
                comp_s_b3 = np.asarray(compress(s_b3))
                comp_s_W4 = np.asarray(compress(s_W4))
                comp_s_b4 = np.asarray(compress(s_b4))
                
            else:
                comp_s_W1 = None
                comp_s_b1 = None
                comp_s_W2 = None
                comp_s_b2 = None
                comp_s_W3 = None
                comp_s_b3 = None
                comp_s_W4 = None
                comp_s_b4 = None
            
            # Parameter node broadcasting the calculated signs to all the worker nodes
            comm.Barrier()
            comp_s_W1 = comm.bcast(comp_s_W1, root=0)
            comm.Barrier()
            comp_s_b1 = comm.bcast(comp_s_b1, root=0)
            comm.Barrier()
            comp_s_W2 = comm.bcast(comp_s_W2, root=0)
            comm.Barrier()
            comp_s_b2 = comm.bcast(comp_s_b2, root=0)
            comm.Barrier()
            comp_s_W3 = comm.bcast(comp_s_W3, root=0)
            comm.Barrier()
            comp_s_b3 = comm.bcast(comp_s_b3, root=0)
            comm.Barrier()
            comp_s_W4 = comm.bcast(comp_s_W4, root=0)
            comm.Barrier()
            comp_s_b4 = comm.bcast(comp_s_b4, root=0)
            comm.Barrier()
            
            # Worker nodes de-compressing the signs
            decomp_recvS_W1 = decompress(comp_s_W1, signs_W1)
            decomp_recvS_b1 = decompress(comp_s_b1, signs_b1)
            decomp_recvS_W2 = decompress(comp_s_W2, signs_W2)
            decomp_recvS_b2 = decompress(comp_s_b2, signs_b2)
            decomp_recvS_W3 = decompress(comp_s_W3, signs_W3)
            decomp_recvS_b3 = decompress(comp_s_b3, signs_b3)
            decomp_recvS_W4 = decompress(comp_s_W4, signs_W4)
            decomp_recvS_b4 = decompress(comp_s_b4, signs_b4)
            
            cur_learning_rate = learning_rate / math.sqrt(epoch_num + 1) 

            # Updating weights (for signSGD)
            if (signSGD) :
                W[1], b[1] = update_para_signSGD(W[1],b[1],decomp_recvS_W1, decomp_recvS_b1,learning_rate)
                W[2], b[2] = update_para_signSGD(W[2],b[2],decomp_recvS_W2, decomp_recvS_b2,learning_rate)
                W[3], b[3] = update_para_signSGD(W[3],b[3],decomp_recvS_W3, decomp_recvS_b3,learning_rate)
                W[4], b[4] = update_para_signSGD(W[4],b[4],decomp_recvS_W4, decomp_recvS_b4,learning_rate)
                
            if(iter_idx == 500) :
                return cost_list, W, b

# Testing the trained model with test data
def test_model(X, Y, W, b, batch_size = 2**8):
    m = X.shape[1]
    assert(m == Y.shape[1])
    batches = split_batches(X, Y, batch_size) 
    acc = 0
    for batch_idx in range(len(batches)):
        X_cur, Y_cur = batches[batch_idx]
        m_cur = X_cur.shape[1]
        assert(m_cur == Y_cur.shape[1])

        Z_cur, A_cur, D_cur = forward_prop(X_cur, W, b, 1.)
        pred = np.argmax(A_cur[L - 1], axis = 0).reshape((m_cur, 1))
        label = np.argmax(Y_cur, axis = 0).reshape((m_cur, 1))
        acc += np.sum(pred == label)
    return acc / m

# Initializing the weights
def load_cache():
    W = [None]
    b = [None]  
    np.random.seed(0)
    for l in range(1, L):
        W.append(np.random.randn(n[l], n[l - 1]) * np.sqrt(2. / n[l - 1]))
        b.append(np.zeros((n[l], 1)))
    return W, b


##### TRAINING SIGN-SGD 
begin1 = time.time()
W_signSGD, b_signSGD = load_cache()
cost_list_signSGD, W1, b1 = gradient_descent(W_signSGD, b_signSGD, dropout_rate=.8, lbd =.03, learning_rate=0.0008, signSGD = True)
if(rank==0):
    signSGD_acc = test_model(X_test,Y_test, W1, b1)
    end1 = time.time()
   
####TRAINING SGD : 
begin2 = time.time()
W_SGD, b_SGD = load_cache()
cost_list_SGD, W2, b2 = gradient_descent(W_SGD, b_SGD, dropout_rate=.7, lbd =.03, learning_rate=0.03, signSGD = False)
if(rank==0):
    sgd_acc = test_model(X_test, Y_test, W2, b2)
    end2 = time.time()
 
if(rank==0):
    x = np.arange(1,501)    # if running normal SGD, make it 1001
    plt.plot(x,cost_list_signSGD, label = "signSGD")
    plt.plot(x,cost_list_SGD, label = "SGD")
    # plt.plot(x,cost_list_SGD, label= "SGD")
    plt.title("Iterations VS Cost")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig("mygraph1.png")
    plt.show()

    print("Accuracy of signSGD model : ", signSGD_acc)
    print(f"Time taken for signSGD: {end1 - begin1}")
    print("Accuracy of normal SGD model : ", sgd_acc)
    print(f"Time taken for normal SGD: {end2 - begin2}")

