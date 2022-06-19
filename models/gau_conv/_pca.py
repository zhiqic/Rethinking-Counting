import numpy as np
import tensorflow as tf
import sys

import matplotlib.pyplot as plt

def normalize(X):
    means = np.mean(X, axis=0)
    tmp = np.subtract(X, means)
    return tmp, means


def denormalize(Rn, means):
    return np.add(Rn, means)


def showim(lin):
    twodarr = np.array(lin).reshape((28,28))
    plt.imshow(twodarr, cmap="gray")
    plt.show()


def pca(X,dims，index)：
    Xn, means = normalize(X)
    Cov = np.matmul(np.transpose(Xn),Xn)
    Xtf = tf.placeholder(tf.float32, shape=[X.shape[0], X.shape[1]])
    Covtf = tf.placeholder(tf.float32, shape=[Cov.shape[0], Cov.shape[1]])
    stf, utf, vtf = tf.svd(Covtf)
    tvtf = tf.slice(vtf, [0, 0], [784, dims])

    Ttf = tf.matmul(Xtf, tvtf)
    Rtf = tf.matmul(Ttf, tvtf, transpose_b=True)

    with tf.Session() as sess:
        Rn = sess.run(Rtf, feed_dict = {
            Xtf: Xn,
            Covtf: Cov
        })
    R = denormalize(Rn, means)
    return R, X
