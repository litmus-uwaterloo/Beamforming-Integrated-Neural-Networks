# PyTorch script for toy beamforming-integrated neural network
# associated with the paper "Beamforming-Integrated Neural Networks
# for Ultrasound Imaging " in Ultrasonics
# ver. 1.0  (Oct 2024 -- by Di Xiao & Alfred Yu)

import torch
from scipy.io import loadmat
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

cuda = torch.device('cuda')

###### Loading sparse matrix and params from Matlab ############
print('load sparse matrix from matlab')
matlab_load = loadmat('sp_matlab.mat')
rows = np.squeeze(matlab_load['rows'].astype(int))-1  # python 0-index vs matlab 1-index
cols = np.squeeze(matlab_load['cols'].astype(int))-1 # python 0-index vs matlab 1-index
rowscols = torch.tensor([rows.tolist(),cols.tolist()])

del rows,cols # save a bit of memory

vals = torch.tensor(np.squeeze(matlab_load['vals']).tolist())
Nz = int(matlab_load['Nz'])
Nx = int(matlab_load['Nx'])
Ns = int(matlab_load['Ns'])
Nc = int(matlab_load['Nc'])

del matlab_load # save a bit of memory

# setting the size of the beamforming matrix correctly
nrows = Nz*Nx
ncols = Ns*Nc

# Generating sparse matrix in PyTorch
sp_torch = torch.sparse_coo_tensor(indices=rowscols,
                                values = vals,
                                size = [nrows,ncols],
                                dtype=torch.float32)
sp_torch_csr = sp_torch.to_sparse_csr().to(cuda)
del vals, rowscols # save a bit of memory

batch_size = 4 # controls the batch size during training/testing, can increase depending on GPU
kernel_in = 1 # controls how many data channels are going into the beamformer

########### Defining beamforming operation #################
# defining the sparse matrix multiplication operation in pytorch but accounting for batch size
# relies on the vectorize-multiply-unvectorize paradigm, using permute and reshape appropriately
class SparseBmfrm(nn.Module):
    def __init__(self, csr_mat,batch_size,kernel_in):
        super().__init__()
        self.csr_mat = csr_mat
        self.batch_size = batch_size
        self.kernel_in = kernel_in

    def forward(self, x):
        batch_rf_vec = torch.t(torch.reshape(torch.permute(x,(0, 1, 3, 2)),(self.batch_size*kernel_in,Ns*Nc)))
        batch_img_vec = self.csr_mat.matmul(batch_rf_vec)
        return torch.permute(torch.reshape(torch.t(batch_img_vec),(self.batch_size,kernel_in,Nx,Nz)),(0,1,3,2))



###### Defining and compiling toy beamforming rf network #############
class ToyBmfrmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding='same')
        self.conv2 = nn.Conv2d(1, 1, 3, padding='same')
        self.bmfrm = SparseBmfrm(sp_torch_csr,batch_size,kernel_in)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bmfrm(x)
        return self.conv2(x)
    
toy_binn_network = ToyBmfrmModel()
toy_binn_network.to(cuda)

criterion = nn.MSELoss()
optimizer = optim.Adam(toy_binn_network.parameters())


####### Loading training data #################
print('loading training and testing data')
matlab_load = loadmat('Data_Tensor.mat')
norm_factor = 15000.0

rf_train = matlab_load['rf_data']/norm_factor # normalize to 0-1 range roughly
rf_train = np.transpose(rf_train,(2,0,1)) # permute to expected dimension order by torch
rf_train = rf_train[:,np.newaxis,:,:] # adding axis expected by torch

img_train = matlab_load['img_data']/norm_factor # normalize to 0-1 range roughly
img_train = np.transpose(img_train,(2,0,1)) # permute to expected dimension by torch
img_train = img_train[:,np.newaxis,:,:] # adding axis expected by torch

del matlab_load # save a bit of memory

# creating 75-25 split for train-test as example
rf_test = rf_train[36:48,:,:,:]
img_test = img_train[36:48,:,:,:]
rf_test_torch = torch.tensor(rf_test,dtype=torch.float32).to(cuda)

rf_train = rf_train[0:36,:,:,:]
img_train = img_train[0:36,:,:,:]

rf_tensor_train = torch.tensor(rf_train,dtype=torch.float32).to(cuda)
img_tensor_train = torch.tensor(img_train,dtype=torch.float32).to(cuda)
training_data = torch.utils.data.TensorDataset(rf_tensor_train,img_tensor_train)
trainloader = torch.utils.data.DataLoader(training_data,batch_size=batch_size)

######### Running and training network to see result ###############
num_test = 12
img_untrained = np.zeros((num_test,1,Nz,Nx),dtype=np.float32)
img_trained = np.zeros((num_test,1,Nz,Nx),dtype=np.float32)

# running untrained network on test data
for i in range(12//batch_size):
    simple_output = toy_binn_network(rf_test_torch[i*batch_size:(i+1)*batch_size,:,:,:])
    img_untrained[i*batch_size:(i+1)*batch_size,:,:,:] = simple_output.cpu().detach().numpy()

## Training the network for epochs
print('Start Training')
for epoch in range(10000):
    running_loss = 0.0
    
    for i,data in enumerate(trainloader,0):
        # get inputs
        input_rf,output_img = data
        
        # zero parameter gradients
        optimizer.zero_grad()
        
        outputs = toy_binn_network(input_rf)
        loss = criterion(outputs,output_img)
        loss.backward()
        optimizer.step()
        
        #print statistics
        running_loss += loss.item()

    print(f'Epoch {epoch + 1} loss: {running_loss:.4e}')

print('Finished Training')

# running trained network on test data
for i in range(12//batch_size):
    simple_output = toy_binn_network(rf_test_torch[i*batch_size:(i+1)*batch_size,:,:,:])
    img_trained[i*batch_size:(i+1)*batch_size,:,:,:] = simple_output.cpu().detach().numpy()

########## Visualizing results from toy binn ##################
weights_prebf = toy_binn_network.conv1.weight.cpu().detach().numpy()
bias_prebf = toy_binn_network.conv1.bias.cpu().detach().numpy()
weights_postbf = toy_binn_network.conv2.weight.cpu().detach().numpy()
bias_postbf = toy_binn_network.conv2.bias.cpu().detach().numpy()

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



