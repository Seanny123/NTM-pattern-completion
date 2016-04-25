import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import tasks
import ipdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

#Parameters
t_forward=5 #generative
batch_size=16
epochs=100
lstm_size=64
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
y_ahead=np.zeros((samples_test,t_forward,pixels), dtype=np.int32)

for i in range(samples_train):
    start=np.random.randint(timesteps-1)
    X_train[i]=predict_train[start: start+t_back]
    y_train[i]=predict_train[start+t_back+1]
for i in range(samples_test):
    start=np.random.randint(timesteps-1) #for whole dataset
    X_test[i]=predict_train[start: start+t_back]
    y_test[i]=predict_train[start+t_back+1]
    # y_ahead[i]=predict_train[np.mod(start+t_back+t_forward,timesteps+t_back)]
    y_ahead[i]=predict_train.take(range(start+t_back+1,start+t_back+1+t_forward),axis=0,mode='wrap')

# print X_train.shape,y_train.shape,y_ahead.shape
# print X_test.shape,y_test.shape,y_ahead.shape

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

def predict_generative(model, x_test, y_ahead, batch_size, t_forward):
    predict=np.zeros((samples_test,t_forward,pixels))
    #to generate next predict, take an decreasing number of timesteps from xtest data
    #and append i predictions timesteps onto the end.
    for i in range(t_forward):
        #from x_test: all sample points, t_back-i timesteps forward
        #from predict: i timesteps forward
        new_xtest=np.concatenate((x_test[:,i:], predict[:,:i]),axis=1)
        predict[:,i]=model.predict(new_xtest, batch_size=batch_size)
    evaluate=model.evaluate(new_xtest,y_ahead[:,-1])
    return evaluate, predict 

print 'Evaluating...'
evaluations = model.evaluate(X_test, y_test, batch_size=batch_size)
predictions = model.predict(X_test,batch_size=batch_size)
evaluate_ahead, predict_forward = predict_generative(model,X_test,y_ahead,batch_size,t_forward)
print 'Evaluation on "%s" loss function:' %loss
print 'One ahead: %s' %evaluations
print '%s ahead: %s' %(t_forward, evaluate_ahead)

figure, (ax1, ax2) = plt.subplots(1, 2)
# sns.set(style='darkgrid',context='paper')
# ax1.imshow(predictions[:samples_plot].T, cmap='gray', interpolation='none')
# ax2.imshow(y_test[:samples_plot].T, cmap='gray', interpolation='none')
# plot_correct=np.concatenate((X_test[0,:t_back],y_ahead[0,:t_forward]),axis=0).T
# plot_predict=np.concatenate((X_test[0,:t_back],predict_forward[0,:t_forward]),axis=0).T
plot_predict=predict_forward[0,:t_forward].T
plot_correct=y_ahead[0,:t_forward].T
ax1.imshow(plot_predict, cmap='gray', interpolation='none')
ax2.imshow(plot_correct, cmap='gray', interpolation='none')
ax1.set(xlabel='time',ylabel='',title='predictions',xticks=[],yticks=[])
ax2.set(xlabel='time',ylabel='',title='correct',xticks=[],yticks=[])
# plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
plt.show()