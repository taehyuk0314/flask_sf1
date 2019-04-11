import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

class GradientDescent:
    @staticmethod
    def execute():
        X = [1., 2., 3.]
        Y = [1., 2., 3.]
        m = n_sample = len(X)
        W = tf.placeholder(tf.float32)
        hypothesis = tf.multiply(X, W)
        cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m
        W_val = []
        cost_val = []

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(-30, 50):
                W_val.append(i * 0.1)
                cost_val.append(sess.run(cost, {W: i * 0.1}))
            plt.plot(W_val, cost_val, 'ro')
            plt.ylabel('COST')
            plt.xlabel('W')
            plt.savefig('../data/GradientDescent.svg')
            print("경사하강법 실행중...")
            return "경사하강법"