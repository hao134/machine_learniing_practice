import numpy as np 
np.random.seed(1337)       # for reprofucibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import timeit

#training X shape (60000,28x28), Y shape(60000, ). test X shape(10000, 28x28), Y shape(10000,)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
#print(X_test.shape)
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
# X_train = X_train.reshape(X_train.shape[0], -1) / 255  # normalize
# X_test  = X_test.reshape(X_test.shape[0], -1) /255     # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()

# Conv layer 1 output shape(32, 28, 28)
model.add(Convolution2D(
    batch_input_shape = (None, 1, 28, 28),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    data_format = 'channels_first' #The ordering of the dimensions in the inputs.
    # channels_first corresponds to inputs with shape  (batch, features, steps)
))
model.add(Activation('relu'))

#Pooling layer 1 (max pooling) output shape (32,14,14)
model.add(MaxPooling2D(
    pool_size=2,
    strides= 2,
    padding='same',
    data_format='channels_first'
))

# Conv layer 2 output shape(64,14,14)
model.add(Convolution2D(64,5,strides=1, padding = 'same',data_format = 'channels_first'))
model.add(Activation('relu'))

# Pooling layer 2(max pooling) output shape(64, 7, 7)
model.add(MaxPooling2D(2,2,'same',data_format = 'channels_first'))

# Fully connected layer 1 input shape (64*7*7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#Fully connected layer 2 to shape(10) for 10 classes
model.add(Dense(10),)
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr = 1e-4)

# We add metrices to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start = timeit.default_timer()

print('Training ---------')
model.fit(X_train, y_train, epochs=1, batch_size=100,)

print('\nTraining ---------')
loss, accuracy = model.evaluate(X_test, y_test)

stop = timeit.default_timer()
print('time elapse={0}'.format(stop-start)) 

print('\n test loss', loss)

print('\n test accuracy: ',accuracy)

    
    