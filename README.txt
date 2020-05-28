Code for applying SVIGL to variational inference for Poisson-Gaussian denoising and optical flow estimation.

# Content

This software package contains code to apply Stochastic Variational Inference with Gradient Linearization (SVIGL) to the tasks of Poisson-Gaussian denoising and optical flow estimation as described in the CVPR18 paper:

Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.

# Usage

Run `test_svigl_poisson_gaussian_denoising.m` to see SVIGL in action for denoising. Upon completion you will see the original image, the noisy image, as well as the mean and standard deviation of the inferred variational approximation to the posterior.
With `test_svigl_optical_flow.m` you can see a sample of optical flow refinement. Please note that the images are cropped and the hyperparameters are adjusted for faster processing.


# Contact

For questions and suggestions please contact Tobias Plötz (tobias.ploetz (at) visinf.tu-darmstadt.de) or Anne Wannenwetsch (anne.wannenwetsch (at) visinf.tu-darmstadt.de).
