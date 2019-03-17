import tensorflow as tf 
import numpy as np 

# create date
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 +0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-5.0,5.0)) #1-d range from -1.0 to 1.0
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1) #0.5 is the learning rate
train = optimizer.minimize(loss) 

init = tf.initialize_all_variables()   #initialize all variables
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)             ######### Activate the init, this is very important ############

for step in range(1001):
    sess.run(train)
    if step % 50 == 0:
        print(step,sess.run(Weights),sess.run(biases),sess.run(loss))
       # print('loss is: ',sess.run(loss))

 