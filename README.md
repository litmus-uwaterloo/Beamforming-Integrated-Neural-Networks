# Beamforming Integrated Neural Networks

This repository contains simplified sample code for a beamforming-integrated neural network.

# Citation

If you use any code or data from this repository, please cite the associated [paper](https://doi.org/10.1016/j.ultras.2024.107474) from _Ultrasonics_
> D. Xiao, A. C. H. Yu, "Beamforming-integrated neural networks for ultrasound imaging", _Ultrasonics_, vol. 145, pp. 107474, Jan. 2025, doi: 10.1016/j.ultras.2024.107474.

# Code Sample and Data
NOTE AS OF OCT 21, 2024: We apologize for the delay in uploading the sample code. Initial code has now been uploaded.

The repository contains one [TensorFlow script](/example_tensorflow.py) and one [PyTorch script](/example_pytorch.py) containing the entirety of each example, alongside [shared data](/Data) to run sample training and inference steps. Please <ins>first run</ins> [gen_sparse_matrix.m](/gen_sparse_matrix.m) to generate both the sparse beamforming matrix and the training data. This script gives a simple example of how to generate a sparse matrix according to the desired beamforming parameters. Then run either script for a demonstration of the toy beamforming-integrated neural network architecture found in the corresponding paper.

These scripts have been tested using Tensorflow ver. 2.10.1, PyTorch ver. 2.0.0, and CUDA v11.2.

If any problems arise with the code or if you have any questions, please feel free to raise an issue in the Github or [email](mailto:di.xiao@uwaterloo.ca) me.
