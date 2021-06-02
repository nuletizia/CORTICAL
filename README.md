# CORTICAL
Keras implementation of cooperative capacity learning (CORTICAL)

If you used the repository for your experiments, please cite the paper.

<img src="https://github.com/nuletizia/CORTICAL/blob/master/teaser.jpg" width=700>

The paper presents two new discriminative mutual information estimators, i-DIME and d-DIME. 
On top of DIME, a cooperative framework (CORTICAL) is discussed to estimate the channel capacity using a combined generator/discriminator model. The official implementations are now available.


-- DIME --

If you want to train and test your own i-DIME model and compare its performance with our results for the 2-d Gaussian case

> python iDIME.py --batch_size 512 --epochs 5000 --test_size 10000

Output is a .mat file containing the $\hat{i}_{iDIME}$ estimator, see the paper for more details.

If you want to train and test your own d-DIME model and compare its performance with our results for the 2-d Gaussian case

> python dDIME.py --batch_size 512 --epochs 5000 --test_size 10000 --alpha 0.1

Output is a .mat file containing both the $\hat{i}_{iDIME}$ and the $\tilde{i}_{iDIME}$ estimators, see the paper for more details.


-- CORTICAL --


If you want to train your own CORTICAL model and compare its performance with our results

> python CORTICAL.py --batch_size 512 --epochs 500 --test_size 10000 --alpha 0.1

Output is a .mat file containing both the $\hat{i}_{iDIME}$ and the $\tilde{i}_{iDIME}$ estimators and the channel input-output samples, see the paper for more details.

To analyize the discrete input cases, modify the variable "noise_real" according to your input distribution and modify the "latent_dim" and "data_dim" variables. In the discrete case, they are related with the code rate.
