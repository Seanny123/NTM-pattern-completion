import numpy as np
import ipdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from helper import *

#Parameters
filename='plotter'

root=os.getcwd()
os.chdir(root+'/data/')
addon=str(np.random.randint(0,100000))
fname=filename+addon

print "Loading Data..."
path='/home/pduggins/git/NTM-pattern-completion/data/'
filename='lstm_main56623'
params=pd.read_json(path+filename+'_params.json')
dataframe=pd.read_pickle(path+filename+'_data.pkl')

print 'Plotting...'
sns.set(context='talk',style='white')
figure2, (ax3, ax4) = plt.subplots(1, 2)
ax3.imshow(predict[1,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax4.imshow(y_test[1,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax3.set(xlabel='time',ylabel='',title='predictions',xticks=[],yticks=[])
ax4.set(xlabel='time',ylabel='',title='correct',xticks=[],yticks=[])
figure2.savefig(fname+'_y.png')