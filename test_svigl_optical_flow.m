% Example code for applying SVIGL to Poisson-Gaussian denoising
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Pl??tz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

addpath(genpath('.'))

% Load sample data
im1 = double(imread('./sample_data/frame1.png'));
im2 = double(imread('./sample_data/frame2.png'));
load('./sample_data/mu_init.mat');
load('./sample_data/ground_truth.mat')

%%
% Energy parameters
energy_opt = opticalflow_energyopt();

% Optimization parameters
% Parameters adjusted for faster processing & smaller inputs
svigl_opt = struct;
svigl_opt.niter = 2;
svigl_opt.nsamples = 50;

% Create initialization sigma
sigma_init = 1e-3 * ones(size(mu_init));

% Pre-process images
im1 = imgaussfilt(im1, 1.1, 'FilterSize', 2*(floor(3*1.1)+1)+1);
im2 = imgaussfilt(im2, 1.1, 'FilterSize', 2*(floor(3*1.1)+1)+1);

% Crop inputs for faster processing
mu_init = mu_init(51:350,201:500,:);
sigma_init = sigma_init(51:350,201:500,:);
im1 = im1(51:350,201:500,:);
im2 = im2(51:350,201:500,:);
ground_truth = ground_truth(51:350,201:500,:);
[n,m,~] = size(mu_init);

% Run SVIGL
[mu, sigma] = flow.svigl(im1, im2, mu_init, sigma_init, energy_opt, svigl_opt);

%% Plot results
flow_ground_truth = flowToColor(ground_truth);
flow_init = flowToColor(mu_init);
flow_svigl = flowToColor(mu);
figure(1), clf;
subplot(2,2,1), imshow(flow_ground_truth), title('Ground truth');
subplot(2,2,2), imshow(flow_init), title('Flow init');
subplot(2,2,3), imshow(flow_svigl), title('Mu');
subplot(2,2,4), imagesc(sum(log(sigma.^2),3)), title('Log Sigma'), axis image, axis off, colorbar;
