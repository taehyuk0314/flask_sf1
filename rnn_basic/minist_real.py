import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
#**********
#mining
#**********
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)
#**********
#modeling
#**********
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
#**********
#learning
#**********
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 100
total_epoch = 30
total_batch = int(mnist.train.num_examples/batch_size)
for epoch in range(total_epoch):
   total_cost = 0
   for i in range(total_batch):
       batch_xs, batch_ys = mnist.train.next_batch(batch_size)
       _, cost_val = sess.run([optimizer, cost], {X:batch_xs, Y:batch_ys, keep_prob:0.8})
       total_cost += cost_val
   print('Epoch : ', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost/total_batch))
print('최적화 완료...훈련종료')

# 결과 확인
labels = sess.run(model, {X: mnist.test.images, Y: mnist.test.labels, keep_prob:1})
fig = plt.figure()
for i in range(10):
   subplot = fig.add_subplot(2, 5, i+1)
   subplot.set_xticks([])
   subplot.set_yticks([])
   subplot.set_title('%d'%np.argmax(labels[i]))
   subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap= plt.cm.gray_r)
plt.show()