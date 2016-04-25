import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import tasks
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

#Parameters
task='one-ahead' #generative
batch_size=16
epochs=20
lstm_size=32
samples_plot=10
samples_train=1000
samples_test=100
optimizer='adam' #rmsprop, sgd, adam
final_activation='hard_sigmoid'
loss='mse' #mse, categorical_crossentropy


print "Generating Data..."
in_timesteps=20
in_pixels=20
inc_res, res = tasks.generate_predict_sequence(in_timesteps, in_pixels) #from NTM-pattern-completion
predict_train, predict_test = np.array(inc_res), np.array(res) 
timesteps=in_timesteps/2 #output of generate_predict_secquence is 0.75*in_timesteps (last 0.25 repeated)
pixels=in_pixels+2 #output of generate_predict_secquence is in_pixels+2
t_back=in_timesteps/4

# cut the image in semi-redundant sequences of length 't_back' to use as training data
# then use the next entry as the correct output
X_train=np.zeros((samples_train,t_back,pixels), dtype=np.int32)
X_test=np.zeros((samples_test,t_back,pixels), dtype=np.int32)
y_train=np.zeros((samples_train,pixels), dtype=np.int32)
y_test=np.zeros((samples_test,pixels), dtype=np.int32)

for i in range(samples_train):
    start=np.random.randint(timesteps-1)
    X_train[i]=predict_train[start: start+t_back]
    y_train[i]=predict_train[start+t_back+1]
for i in range(samples_test):
    start=np.random.randint(timesteps-1) #for whole dataset
    X_test[i]=predict_train[start: start+t_back]
    y_test[i]=predict_train[start+t_back+1]


# print X_train.shape,y_train.shape
# print X_test.shape,y_test.shape
# print (t_back, pixels)

print "Building LSTM..."
model = Sequential()
model.add(LSTM(lstm_size, input_shape=(t_back, pixels)))
model.add(Dropout(0.2))
model.add(Dense(pixels))
model.add(Dropout(0.2))
model.add(Activation(final_activation))
model.compile(loss=loss, optimizer=optimizer)

print 'Training...'
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, y_test))

print 'Evaluating...'
evaluations = model.evaluate(X_test, y_test, batch_size=batch_size)
predictions = model.predict(X_test,batch_size=batch_size)
print 'Performance: %s=' %loss, evaluations
# print('predictions:', predictions)
# print('correct:', y_test)
# print predictions.shape, y_test.shape

figure, (ax1, ax2) = plt.subplots(1, 2)
# sns.set(style='darkgrid',context='paper')
ax1.imshow(predictions[:samples_plot].T, cmap='gray', interpolation='none')
ax2.imshow(y_test[:samples_plot].T, cmap='gray', interpolation='none')
ax1.set(xlabel='sample',ylabel='value',title='predictions')
ax2.set(xlabel='sample',ylabel='value',title='correct')
plt.show()