#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

# parameters =============================
# generate data
data_center=[[2,2],[2,-2],[-2,2],[-2,-2]] # data distribtion centers
train_num=1000                     # number of data around a data center
test_num=100
noise_std=1.0                    # noise of data around a center
# hidden layer
layer_nodes=[10]
#act_func=tf.nn.relu
# train
batch_size=50
step=500
step_show=10
learning_rate=0.5
# generate data =============================
np.random.seed(1)
tot_class=len(data_center)

# create empty numpy array, so we can use vstack and hstack
x_train=np.array([]).reshape(0,2)
x_test=np.array([]).reshape(0,2)
y_train=np.array([])
y_test=np.array([])

# create traning data and validation dataset around each data_center
for n, dc in enumerate(data_center):
        x_train=np.vstack((x_train,np.random.normal(np.tile(dc,(train_num,1)),noise_std)))
        y_train=np.hstack((y_train,np.repeat(n,train_num)))
        x_test=np.vstack((x_test,np.random.normal(np.tile(dc,(test_num,1)),noise_std)))
        y_test=np.hstack((y_test,np.repeat(n,test_num)))
        print(n,dc)

# computation graph =========================
#print("x_train",x_train)
#print("y_train",y_train)
#print(y_train)
#print(np.shape(y_train))
plt.figure(0,figsize=(18, 6))
plt.subplot(1,3,1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=100, lw=0, cmap='tab10')
plt.show()
#end