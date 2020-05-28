% Computes the linearized gradient of the EpicFlow optical flow energy
% wrt. the flow
%
% Inputs
% im1, im2:     Input images
% current_flow: Current optical flow estimate
% cleansamples: Estimates of the flow at which to evaluate energy
% energy_opt:   Parameters of energy function
%
% Outputs
% Ac, bc:       Parameterization of the linearized gradient (constant for
%               all samples)
% Ax, bx:       Parameterization of the linearized gradient (varying for
%               each sample)
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ Ac, Ax, bc, bx ] = compute_linearized_gradient_energy(im1, im2, current_flow, cleansamples, energy_opt)

lambdaD = getOpt(energy_opt, 'lambdaD');
lambdaS = getOpt(energy_opt, 'lambdaS');
kappa = getOpt(energy_opt, 'kappa');
epsilon_data_term = getOpt(energy_opt, 'epsilon_data_term');
filters = getOpt(energy_opt, 'filters');
dphi_D = getOpt(energy_opt, 'dphi_D');
dphi_S = getOpt(energy_opt, 'dphi_S');

[n,m,~,nsamples] = size(cleansamples);
npixels = n * m;

% Get image derivatives wrt current flow estimate
[~,~,~,Ixx,Ixy,Iyx,Iyy,Ixt,Iyt] = flow.calculate_image_derivatives(im1,im2,current_flow);

% Calculate coefficients for the data term gradient (x-direction)
Ixx2 = Ixx.^2;
IxxIxy = Ixx .* Ixy;
Ixy2 = Ixy.^2;
IxxIxt = Ixx .* Ixt;
IxyIxt = Ixy .* Ixt;
Ixt2 = Ixt .* Ixt;
normalization = Ixx2 + Ixy2 + epsilon_data_term;
Ixx2 = sum(Ixx2./normalization,3);
IxxIxy = sum(IxxIxy./normalization,3);
Ixy2 = sum(Ixy2./normalization,3);
IxxIxt = sum(IxxIxt./normalization,3);
IxyIxt = sum(IxyIxt./normalization,3);
Ixt2 = sum(Ixt2./normalization,3);

% Calculate coefficients for the data term gradient (y-direction)
Iyx2 = Iyx.^2;
IyxIyy = Iyx .* Iyy;
Iyy2 = Iyy.^2;
IyxIyt = Iyx .* Iyt;
IyyIyt = Iyy .* Iyt;
Iyt2 = Iyt .* Iyt;
normalization = Iyx2 + Iyy2 + epsilon_data_term;
Iyx2 = sum(Iyx2./normalization,3);
IyxIyy = sum(IyxIyy./normalization,3);
Iyy2 = sum(Iyy2./normalization,3);
IyxIyt = sum(IyxIyt./normalization,3);
IyyIyt = sum(IyyIyt./normalization,3);
Iyt2 = sum(Iyt2./normalization,3);

% Combine data term gradient coefficients for both directions
K11 = Ixx2 + Iyx2;
K12 = IxxIxy + IyxIyy;
K13 = IxxIxt + IyxIyt;
K22 = Ixy2 + Iyy2;
K23 = IxyIxt + IyyIyt;
K33 = Ixt2 + Iyt2;

% Compute spatially varying weight lambdaS
lambdaS = lambdaS * flow.calculate_spatial_lambdaS(im1, kappa);
lambdaS1 = 0.5 * imfilter(lambdaS,abs(filters{1}), 'conv', 'replicate');
lambdaS2 = 0.5 * imfilter(lambdaS,abs(filters{2}), 'conv', 'replicate');

% Prepare spatial filter matrices
filter_matrix1 = make_convn_mat(filters{1}, [n m], 'valid', 'same');
filter_matrix2 = make_convn_mat(filters{2}, [n m], 'valid', 'same');

% Initialize matrix components
Ac = sparse(2*n*m, 2*n*m);
bc = sparse(2*n*m, 1);

Ax = cell(nsamples,1);
bx = cell(nsamples,1);

for i = 1:nsamples
    % Compute flow offsets
    du = cleansamples(:,:,1,i) - current_flow(:,:,1);
    dv = cleansamples(:,:,2,i) - current_flow(:,:,2);

    % Compute data gradient
    data_term = sqrt(K33 + 2 * K13 .* du + 2 * K23 .* dv + K11 .* du.^2 + 2 * K12 .* du .* dv + K22 .* dv.^2);
    [~, data_term] = dphi_D(data_term);
    data_term = lambdaD * data_term;
    duu = spdiags(data_term(:).*K11(:), 0, npixels, npixels);
    dvv = spdiags(data_term(:).*K22(:), 0, npixels, npixels);
    duv = spdiags(data_term(:).*K12(:), 0, npixels, npixels);

    % Compute smoothness gradient
    smoothness_term11 = imfilter(current_flow(:,:,1) + du,filters{1}, 'conv', 'replicate');
    smoothness_term12 = imfilter(current_flow(:,:,2) + dv,filters{1}, 'conv', 'replicate');
    smoothness_term = sqrt(smoothness_term11.^2 + smoothness_term12.^2);
    [~, smoothness_term] = dphi_S(smoothness_term);
    smoothness_term1 = zeros(n,m);
    smoothness_term1(1:end-1,:) = smoothness_term(1:end-1,:) .* lambdaS1(1:end-1,:);

    smoothness_term21 = imfilter(current_flow(:,:,1) + du, filters{2}, 'conv', 'replicate');
    smoothness_term22 = imfilter(current_flow(:,:,2) + dv, filters{2}, 'conv', 'replicate');
    smoothness_term = sqrt(smoothness_term21.^2 + smoothness_term22.^2);
    [~, smoothness_term] = dphi_S(smoothness_term);
    smoothness_term2 = zeros(n,m);
    smoothness_term2(:,1:end-1) = smoothness_term(:,1:end-1) .* lambdaS2(:,1:end-1);

    A_smoothness = filter_matrix1'*spdiags(smoothness_term1(:), 0, npixels, npixels)*filter_matrix1 + filter_matrix2'*spdiags(smoothness_term2(:), 0, npixels, npixels)*filter_matrix2;

    % Combine data and smoothness gradient terms
    Ax{i} = [duu duv; duv dvv] + [A_smoothness, sparse(npixels, npixels); sparse(npixels, npixels), A_smoothness];
    bx{i} = [data_term(:).*K13(:); data_term(:).*K23(:)] - [duu duv; duv dvv] * current_flow(:);

end


end
