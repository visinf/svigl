% Computes the value of the EpicFlow optical flow energy (Eq. 15)
%
% Inputs
% im1, im2:     Input images
% clean:        Flow at which to evaluate energy
% energy_opt:   Parameters of energy function
%
% Outputs
% E:            Energy value
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ E ] = energy(im1, im2, clean, energy_opt)

lambdaD = getOpt(energy_opt, 'lambdaD');
lambdaS = getOpt(energy_opt, 'lambdaS');
kappa = getOpt(energy_opt, 'kappa');
epsilon_data_term = getOpt(energy_opt, 'epsilon_data_term');
filters = getOpt(energy_opt, 'filters');
phi_D = getOpt(energy_opt, 'phi_D');
phi_S = getOpt(energy_opt, 'phi_S');

% Get image derivatives wrt current flow estimate
[~,~,~,Ixx,Ixy,Iyx,Iyy,Ixt,Iyt] = flow.calculate_image_derivatives(im1, im2, clean);

% Calculate data term
Ixx2 = Ixx.^2;
Ixy2 = Ixy.^2;
Iyx2 = Iyx.^2;
Iyy2 = Iyy.^2;
Ixt2 = Ixt.^2;
Iyt2 = Iyt.^2;
normalization_x = Ixx2 + Ixy2 + epsilon_data_term;
normalization_y = Iyx2 + Iyy2 + epsilon_data_term;
data_term = sqrt(sum(Ixt2./normalization_x + Iyt2./normalization_y,3));
data_term = lambdaD * phi_D(data_term);

% Compute spatially varying weight lambdaS
lambdaS = lambdaS * flow.calculate_spatial_lambdaS(im1,kappa);
lambdaS1 = 0.5 * imfilter(lambdaS, abs(filters{1}), 'conv', 'replicate');
lambdaS2 = 0.5 * imfilter(lambdaS, abs(filters{2}), 'conv', 'replicate');

% Calculate smoothness term
smoothness_term11 = imfilter(clean(:,:,1), filters{1}, 'conv', 'replicate');
smoothness_term12 = imfilter(clean(:,:,2), filters{1}, 'conv', 'replicate');
smoothness_term = sqrt(smoothness_term11.^2 + smoothness_term12.^2);
smoothness_term1 = phi_S(smoothness_term);

smoothness_term21 = imfilter(clean(:,:,1), filters{2}, 'conv', 'replicate');
smoothness_term22 = imfilter(clean(:,:,2), filters{2}, 'conv', 'replicate');
smoothness_term = sqrt(smoothness_term21.^2 + smoothness_term22.^2);
smoothness_term2 = phi_S(smoothness_term);

smoothness_term = zeros(size(clean,1),size(clean,2));
smoothness_term(1:end-1,:) = smoothness_term(1:end-1,:) + lambdaS1(1:end-1,:) .* smoothness_term1(1:end-1,:);
smoothness_term(:,1:end-1) = smoothness_term(:,1:end-1) + lambdaS2(:,1:end-1) .* smoothness_term2(:,1:end-1);

% Combine energy terms
E = sum(sum(data_term)) + sum(sum(smoothness_term));

end
