# Tensorflow script for toy beamforming-integrated neural network
# associated with the paper "Beamforming-Integrated Neural Networks
# for Ultrasound Imaging " in Ultrasonics
# ver. 1.0  (Oct 2024 -- by Di Xiao & Alfred Yu)

import tensorflow as tf
from scipy.io import loadmat
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D,  Lambda
from tensorflow.keras.models import Model



###### Loading sparse matrix and params from Matlab ############
print('load sparse matrix from matlab')
matlab_load = loadmat('sp_matlab.mat')
rows = np.squeeze(matlab_load['rows'].astype(int))-1  # python 0-index vs matlab 1-index
cols = np.squeeze(matlab_load['cols'].astype(int))-1 # python 0-index vs matlab 1-index
rowscols = tuple(zip(rows.tolist(),cols.tolist()))

del rows,cols # save a bit of memory

vals = np.squeeze(matlab_load['vals'])
vals = vals.tolist()
Nz = int(matlab_load['Nz'])
Nx = int(matlab_load['Nx'])
Ns = int(matlab_load['Ns'])
Nc = int(matlab_load['Nc'])

del matlab_load # save a bit of memory

# setting the size of the beamforming matrix correctly
nrows = Nz*Nx
ncols = Ns*Nc

# Generating sparse matrix in Tensorflow
sp_tf = tf.sparse.SparseTensor(indices=rowscols, values = vals, dense_shape = [nrows,ncols])
del vals, rowscols # save a bit of memory

batch_size = 4 # controls the batch size during training/testing, can increase depending on GPU
kernel_in = 1  # controls how many data channels are going into the beamformer

########### Defining beamforming operation #################
# defining the sparse matrix multiplication operation in tensorflow but accounting for batch size and number of data channels
# relies on the vectorize-multiply-unvectorize paradigm, using permute and reshape appropriately
@tf.function
def tf_bmfrm_batch(x):
    tf_vector = tf.reshape(tf.transpose(x,perm=[0, 3, 2, 1]),(batch_size,kernel_in,ncols))
    tf_vector = tf.reshape(tf.transpose(tf_vector,perm=[2, 0, 1]),(ncols,kernel_in*batch_size))
    tf_bmode_vec = tf.sparse.sparse_dense_matmul(sp_tf, tf_vector)
    tf_bmode_vec = tf.reshape(tf_bmode_vec,(nrows,batch_size,kernel_in))
    tf_bmode = tf.reshape(tf.transpose(tf_bmode_vec,perm=[1,2,0]),(batch_size,kernel_in,Nx,Nz))
    tf_img = tf.transpose(tf_bmode,perm=[0,3,2,1])
    return tf_img


###### Defining and compiling toy beamforming rf network #############
print('defining and compiling network')
inputs_rf = Input(shape=(Ns,Nc,1))
hidden_layer = Conv2D(1, (3, 3), activation='linear', padding='same')(inputs_rf)
bmfrm = Lambda(tf_bmfrm_batch)(hidden_layer)  # Lambda layer to turn function in tf layer
out_layer = Conv2D(1, (3, 3), activation='linear', padding='same')(bmfrm)

toy_binn_network = Model(inputs_rf, out_layer)
toy_binn_network.compile(optimizer='adam', loss='mean_squared_error')
toy_binn_network.summary()


####### Loading training data #################
print('loading training and testing data')
matlab_load = loadmat('Data_Tensor.mat')
norm_factor = 15000.0

rf_train = matlab_load['rf_data']/norm_factor # normalize to 0-1 range roughly
rf_train = np.transpose(rf_train,(2,0,1)) # permute to expected dimension order by tensorflow
rf_train = rf_train[:,:,:,np.newaxis] # adding axis expected by tf

img_train = matlab_load['img_data']/norm_factor # normalize to 0-1 range roughly
img_train = np.transpose(img_train,(2,0,1)) # permute to expected dimension by tensorflow
img_train = img_train[:,:,:,np.newaxis] # adding axis expected by tf

del matlab_load # save a bit of memory

# creating 50-25-25 split for train-val-test as example
rf_val = rf_train[24:36,:,:,:]
img_val = img_train[24:36,:,:,:]

rf_test = rf_train[36:48,:,:,:]
img_test = img_train[36:48,:,:,:]

rf_train = rf_train[0:24,:,:,:]
img_train = img_train[0:24,:,:,:]



######### Running and training network to see result ###############
img_untrained = toy_binn_network.predict(rf_test,batch_size=batch_size)

toy_binn_network.fit(x=rf_train,y=img_train,validation_data=(rf_val,img_val),batch_size=batch_size,epochs=5000)

img_trained = toy_binn_network.predict(rf_test,batch_size=batch_size)




########## Visualizing results from toy binn ##################
weights_prebf,bias_prebf = toy_binn_network.layers[1].get_weights()
weights_postbf,bias_postbf  = toy_binn_network.layers[3].get_weights()

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(weights_prebf),cmap='gray')
plt.colorbar()
plt.xticks(ticks=[])
plt.yticks(ticks=[])
plt.title('Pre-BF: b=' + '{:.2e}'.format(bias_prebf[0]))

plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(weights_postbf),cmap='gray')
plt.colorbar()
plt.xticks(ticks=[])
plt.yticks(ticks=[])
plt.title('Post-BF: b=' + '{:.2e}'.format(bias_postbf[0]))

fig.suptitle('Kernel Weights After Training')
plt.show()


def vis_bmode(img,dyn_range):
    img = hilbert(img)
    img = 20*np.log10(np.abs(img)+1e-10)
    img_max = np.amax(img)
    
    plt.imshow(img,vmin=img_max-dyn_range,vmax=img_max,cmap='gray',aspect=Nx/Nz)

test_idx = 1
fig2 = plt.figure()
plt.subplot(1, 3, 1)
vis_bmode(np.squeeze(img_test[test_idx,:,:,:]),40)
plt.title('Matlab Image')
plt.xticks(ticks=[])
plt.yticks(ticks=[])

plt.subplot(1, 3, 2)
vis_bmode(np.squeeze(img_untrained[test_idx,:,:,:]),40)
plt.title('Untrained Image')
plt.xticks(ticks=[])
plt.yticks(ticks=[])

plt.subplot(1, 3, 3)
vis_bmode(np.squeeze(img_trained[test_idx,:,:,:]),40)
plt.title('Trained Image')
plt.xticks(ticks=[])
plt.yticks(ticks=[])

fig2.suptitle('Image Results from Toy BINN')
plt.show()