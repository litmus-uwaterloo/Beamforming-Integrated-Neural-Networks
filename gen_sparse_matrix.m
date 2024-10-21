% Matlab script for generating sparse matrix for toy beamforming-integrated neural network
% associated with the paper "Beamforming-Integrated Neural Networks for Ultrasound Imaging " in Ultrasonics
% ver. 1.0  (Oct 2024 -- by Di Xiao & Alfred Yu)

%% Loading an initial set of data to extract and set parameters
load('Data\1.mat','rf_filt');
Nc = size(rf_filt,2); % number of channels
Ns = size(rf_filt,1); % number of samples per channel

pitch = 0.3048e-3; % pitch of linear array
pos_trans = pitch*linspace(-(Nc-1)/2,(Nc-1)/2,Nc); % predefined transducer element positions

Nz = 2048; % setting axial pixels
Nx = 256; % setting lateral pixels
pos_z = linspace(5e-3, 35e-3, Nz); % axial pixel positions
pos_x = linspace(-15e-3, 15e-3, Nx); % lateral pixel positions

ang = -1; % steering angle (deg)
fs = 40e6; % DAQ sampling rate
sos = 1540; % m/s
rx_delay = -4.1e-6; % time delay to time zero
fnum = 1.4; % f-number

%% Generating sparse beamforming matrix
sp_mat = gen_mat(pos_trans,pos_z,pos_x,Nz,Nx,Ns,Nc,fs,sos,rx_delay,ang,fnum);

%% Testing sparse beamforming matrix on full data set, construct dataset for training
Ndata = 48;

% initializing data set for training
rf_data = zeros(Ns,Nc,Ndata);
img_data = zeros(Nz,Nx,Ndata);

% looping through the provided data set and beamforming the data
figure
for data_idx = 1:Ndata
    disp(['Beamforming and displaying data index: ' num2str(data_idx)])
    load(['Data\' num2str(data_idx) '.mat'],'rf_filt')
    rf_data(:,:,data_idx) = rf_filt;
    img = reshape(sp_mat*rf_filt(:),[Nz Nx]); % key SMB step vectorize-multiply-unvectorize
    img_data(:,:,data_idx) = img;

    vis_bmode(img,pos_z,pos_x,40)
    title(num2str(data_idx))
    drawnow
    pause(0.25)
end

% saving the data in float32 for easier use in ML frameworks
rf_data = single(rf_data);
img_data = single(img_data);
save('Data_Tensor.mat','rf_data','img_data')

%% Saving sparse matrix for loading into Tensorflow/PyTorch
[rows, cols, vals] = find(sp_mat);
vals = single(vals); % preconverting to float32 for easier use in ML frameworks
save('sp_matlab.mat','rows','cols','vals','Nz','Nx','Ns','Nc')

%% Helper functions
function vis_bmode(img,pos_z,pos_x,dyn_range)
    img = hilbert(img); % hilbert after beamforming only good if axial pixel positions are dense
    img = 20*log10(abs(img) + 1e-10); % in case of 0 values
    img_max = max(img(:));

    % simple display for log-envelope b-mode images
    imagesc(pos_x,pos_z,img,[img_max-dyn_range img_max])
    axis tight
    colormap gray
end

function sp_mat = gen_mat(pos_trans,pos_z,pos_x,Nz,Nx,Ns,Nc,fs,sos,rx_delay,ang,fnum)
    % switching the first firing element for +ve vs -ve angles
    if ang<0  
        wave_source = pos_trans(end);
    else
        wave_source = pos_trans(1);
    end
    
    % preallocating arrays for construction of sparse beamforming matrix
    s_row = zeros(1,2*Nz*Nc);
    s_col = zeros(1,2*Nz*Nc);
    s_val = zeros(1,2*Nz*Nc);
    count = 1;
    
    % looping through all pixel positions
    f = waitbar(0, 'Generating sparse beamforming matrix');
    for z = 1:Nz
        waitbar(z/Nz,f);
        a = pos_z(z)/(2*fnum); % setting aperture
        for x = 1:Nx
            tx_d = pos_z(z)*cosd(ang) + (pos_x(x)-wave_source)*sind(ang); % tx geometry
            rx_d = sqrt(pos_z(z)^2 + (pos_x(x) - pos_trans).^2); % rx geometry
            total_time = rx_delay + (tx_d + rx_d)/sos; % time of flight calculation for all rx channels
            best_samp = max(min(fs*(total_time),Ns-1),1); % light error handling and conversion to samples
    
            % selecting best sample with interpolation
            s_bot = floor(best_samp);
            s_interp = best_samp-s_bot;
            
            % loop through rx channels to insert interpolation weights into sparse matrix
            for c = 1:Nc
                s_row(count) = z+Nz*(x-1);
                s_col(count) = s_bot(c) + Ns*(c-1);
                if abs(pos_trans(c)-pos_x(x)) < a
                    s_val(count) = 1-s_interp(c);
                end
                count = count + 1;
    
                s_row(count) = z+Nz*(x-1);
                s_col(count) = s_bot(c) + 1 + Ns*(c-1);
                if abs(pos_trans(c)-pos_x(x)) < a
                    s_val(count) = s_interp(c);
                end
                count = count + 1;
            end
        end
    end
    close(f)
    
    % forming sparse matrix based on data size and pixel positions
    sp_mat = sparse(s_row,s_col,s_val,Nz*Nx,Ns*Nc);
end
