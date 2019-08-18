import tensorflow as tf
sess = tf.Session()
print(sess.run(tf.nn.softmax([6.0, 7.0, 8.0])))