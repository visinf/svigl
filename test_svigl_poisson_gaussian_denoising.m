% Example code for applying SVIGL to Poisson-Gaussian denoising 
%
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

addpath(genpath('.'));

I = rgb2gray(im2double(imread('peppers.png')));

%%
% Energy parameters
energy_opt = pgdenoising_energyopt();

% Optimization parameters
svigl_opt = struct;
svigl_opt.niter = 40;
svigl_opt.nsamples = 20;

% Create noisy image
sigma_noise = sqrt(energy_opt.sigma_a*I + energy_opt.sigma_b);
Inoisy = I + randn(size(I)).*sigma_noise;
Inoisy = clipImage(Inoisy,0,1);

% Run SVIGL
[Imu, Isigma] = pgdenoising.svigl(Inoisy, Inoisy, 0.1*Inoisy+energy_opt.sigma_b, energy_opt, svigl_opt);

%% Plot results
figure(1), clf;
subplot(2,2,1), imshow(I), title('Original');
subplot(2,2,2), imshow(Inoisy), title('Noisy');
subplot(2,2,3), imshow(Imu), title('Mu');
subplot(2,2,4), imagesc(log(Isigma)), title('Log Sigma'), axis image, axis off, colorbar;