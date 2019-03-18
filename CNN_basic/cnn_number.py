import tensorflow as tf
import timeit 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape , stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape =shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')
# strides ==> [1, paddinx on x for one step, ~ y ~, 1]

# define pooling, use the max value pooling, the core function's size is 2x2
# so the ksize = [1,2,2,1] strides = [1,2,2,1]
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# define thr placeholder
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

# dropout's placeholder
keep_prob = tf.placeholder(tf.float32)

'''
change xs' shape to [-1,28,28,1], -1 represents doesn't concern about the dimension,
and because the imput image doesn't contain color, we choose the channel to be 1.
'''
x_image = tf.reshape(xs,[-1,28,28,1])

# construct convolution layer 
#  # first for weights, the patch's size is 5x5, and don't contain color, 
# the channel input is 1 and output is 32
W_conv1 = weight_variable([5,5,1,32])
# the biase, it's length is 32
b_conv1 = bias_variable([32])

'''
define the 1st convolution layer(1st hidden layer), 
and deal it with non_linear treatment.
because of same padding the output size keep the same(28x28x32).
'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# pooling it, and the output become (14,14,32)
h_pool1  = max_pool_2x2(h_conv1)


#define the 2nd convolution layer,
#the input is 32, output is 64.
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# define the hidden convolution layer 2
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
# define the 2nd pooling, and the output is from 14x14x64 to 7x7x64
h_pool2 = max_pool_2x2(h_conv2)


# define the full connected layer
# first use reshape way to let the h_pool2's 3d_dimension to 1d
# [n_samples,7,7,64] --> [n_samples,7x7x64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# expand the size to 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# concern the overfitting problem, dropout it
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# construct the last layer, the input is 1024, and output is 10(0~9 numbers)
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


### choose the optimizer way
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(ys * tf.log(prediction),
    reduction_indices = [1])
)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

start = timeit.default_timer()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0: 
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]
        ))

stop = timeit.default_timer()
print('time elapse={0}'.format(stop-start)) 
#end
