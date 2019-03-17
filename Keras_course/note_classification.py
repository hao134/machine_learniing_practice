import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
#from keras.optimizers import RMSprop
import matplotlib.pyplot as plt 

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape(10,000, )
(X_train, y_train), (X_test, y_test)  = mnist.load_data() # X_train==> shape(60000,28,28) 
# data pre-processing
#print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # normalize
X_test  = X_test.reshape(X_test.shape[0], -1) /255     # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# build neural net
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# difine optimizer
#rmsprop = RMSprop(lr=0.001,rho = 0.9,epsilon = 1e-08,decay=0.0)

# we get metrices to get more results you want to see
model.compile(
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics = ['accuracy'])

# training
print("Training ===========")
#model.fit(X_train, y_train, epochs = 10,batch_size= 200)

train_history = model.fit(x=X_train,y=y_train,validation_split=0.2, 
                        epochs=10, batch_size=200,verbose=2)

# save train result, to load it, use: model = load_model('model.h5')
#model.save('model.h5')

# plot train results ===============================
def show_train_history(train_history,train_item,valid_item):
    plt.plot(train_history.history[train_item])
    plt.plot(train_history.history[valid_item])
    plt.title('Train History')
    plt.ylabel(train_item)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')


# test the model
print("\nTesting ==========")
# Evaluating the model with the metrices we defined earlier
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss', loss)
print('test accuracy', accuracy)

