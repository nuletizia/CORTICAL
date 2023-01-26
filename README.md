# CORTICAL
This repository contains the official Keras implementation of cooperative channel capacity learning (CORTICAL)

If you used the repository for your experiments, please cite the paper.


The paper presents a cooperative framework (CORTICAL) to both estimate the channel capacity and sample from the capacity-achieving distribution using a combined generator/discriminator model. The official implementation is now available.

<img src="https://github.com/nuletizia/CORTICAL/blob/main/cortical_teaser.png" width=600>
<em>Example of capacity learning in the case of an AWGN channel (d=2) under peak-power constraint (P=10)</em>


<h2> CORTICAL training commands</h2>

If you want to train your own CORTICAL model and compare its performance with our results

> python CORTICAL.py 

A variety of input arguments, e.g., type of channel or type of power constraint, is offered. Please check the arguments of CORTICAL.py for more details. Use the following command to include them

> python CORTICAL.py --batch_size 512 --epochs 500 --test_size 10000 --dim 1 --channel 'AWGN' --power_constraint 'PP'

To modify the value of the power constraint, manually modify the functions defined outside the CORTICAL class.

Output is a series of .mat files. Every 1000 epochs a batch of generated input channels samples is saved. When the execution terminates, estimates of the channel capacity and samples from the optimal (if well trained) input distribution are provided. 

The code has been tested on Python 3.6 with Tensorflow 1.15.2 and Keras 2.2.4. Please adjust libraries and dependencies based on your system.

<h2> CORTICAL training evolution</h2>

The following gifs show how CORTICAL learns the capacity-achieving distribution over time for different type of channels and power constraints.

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/scalar_gaussian_channel_max_1.gif">

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/scalar_gaussian_channel_max_4.gif">

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/scalar_gaussian_channel_max_8.gif">

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/scalar_gaussian_channel_max_16.gif">

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/scalar_gaussian_channel_max_20.gif">

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/2d_gaussian_channel_max_10.gif">

<img src="https://github.com/nuletizia/CORTICAL/blob/main/gifs/scalar_cauchy_channel_max_log4.gif">

