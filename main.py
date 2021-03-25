import tensorflow as tf
import numpy as np

variance_epsilon=0.01
input1 = np.arange(36).reshape(2,3,2,3)
input1 = tf.constant(input1,dtype=tf.float32)
input1 = tf.convert_to_tensor(input1)

mean,variance = tf.nn.moments(
        input1,
        axes = [0,1,2]
)

offset = tf.zeros((input1.get_shape()[-1]),dtype = tf.float32)
scale = tf.ones((input1.get_shape()[-1]),dtype = tf.float32)

output = tf.nn.batch_normalization(input1,mean,variance,offset,scale,variance_epsilon)

with tf.Session() as sess:
    print(sess.run(mean))
    print(sess.run(variance))
    print(sess.run(input1))
    print(sess.run(output))
