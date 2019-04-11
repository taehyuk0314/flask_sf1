import tensorflow as tf

var = tf.Variable([5.0], dtype= tf.float32)
con = tf.constant([10.0], dtype= tf.float32)
session = tf.Session()
init = tf.global_variables_initializer()

session.run(init)

print(session.run(var * con))
print('-------------')
session.run(var.assign([10.0]))
print(session.run(var))

p1 = tf.placeholder(dtype=tf.float32)
p2 = tf.placeholder(dtype=tf.float32)

t1 = p1 * 3
t2 = p1 * p2

session = tf.Session()
print(session.run(t1, feed_dict={p1: [4.0]}))
print(session.run(t2, feed_dict={p1: 4.0, p2: [2.0,5.0]})) # feed_dict 디폴트 값으로 생략가능