# import tensorflow as tf 
# a = tf.random_uniform([1])
# b = tf.random_normal([1])

# print("Session 1")
# with tf.Session() as sess1:
#   print(sess1.run(a))  # generates 'A1'
#   print(sess1.run(a))  # generates 'A2'
#   print(sess1.run(b))  # generates 'B1'
#   print(sess1.run(b))  # generates 'B2'

# print("Session 2")
# with tf.Session() as sess2:
#   print(sess2.run(a))  # generates 'A3'
#   print(sess2.run(a))  # generates 'A4'
#   print(sess2.run(b))  # generates 'B3'
#   print(sess2.run(b))  # generates 'B4'

import tensorflow as tf
#// 把 seed 设置为 1
w1 = tf.random_normal([2,3], stddev=1, seed=1)
w2 = tf.random_normal([3,1], stddev=1, seed=1)

sess =  tf.Session()
sess.run(w1.global_variables_initializer())
sess.run(w2.global_variables_initializer())
print(sess.run(w1))
print(sess.run(w2))
sess.close()