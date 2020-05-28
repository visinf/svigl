% Computes the linearized gradient of the Poisson-Gaussian denoising energy
% wrt. the clean image
% 
% Inputs
% noisy:        Noisy input image
% cleansamples: Estimates of the clean image at which to evaluate energy
% energy_opt:   Parameters of energy function
%
% Outputs
% Ac, bc:       Parameterization of the linearized gradient (constant for
%               all samples)
% Ax, bx:       Parameterization of the linearized gradient (varying for
%               each sample)
% 
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ Ac, Ax, bc, bx ] = compute_linearized_gradient_energy( noisy, cleansamples, energy_opt)

Js = getOpt(energy_opt, 'Js');
dphi_f = getOpt(energy_opt, 'dphi_f');
lambdaD = getOpt(energy_opt, 'lambdaD');
lambdaS = getOpt(energy_opt, 'lambdaS');
sigma_a = getOpt(energy_opt, 'sigma_a');
sigma_b = getOpt(energy_opt, 'sigma_b');

x = cleansamples;
y = noisy;

[n,m, nsamples] = size(x);
nfilter = numel(Js);

% Precompute filter responses
F = cell(nfilter,1);
Jx = cell(nfilter,1);
for j=1:nfilter
    F{j} = make_convn_mat(Js{j}, [n m], 'valid');
    Jx{j} = convn(x, Js{j}, 'valid');
end

Ac = sparse(n*m, n*m);
bc = 0;

Ax = cell(nsamples,1);
bx = cell(nsamples,1);
for i=1:nsamples
    x_i = flatMat(x(:,:,i));
    
    % Calculate noise standard deviation / variance (Eq. 16)
    sigma_noise = sqrt(max(sigma_a*x_i,0)+sigma_b);
    sigma_noise_sq = sigma_noise.^2;
    
    % Eq. 35 (supplemental)
    Ax{i} = lambdaD.*(1./sigma_noise_sq-0.5*sigma_a*x_i./sigma_noise_sq.^2+sigma_a*y(:)./sigma_noise_sq.^2);
    Ax{i} = spdiags(Ax{i}(:), 0, numel(Ax{i}), numel(Ax{i}));
    
    % Eq. 36 (supplemental)
    bx{i} = -lambdaD.*(0.5*sigma_a*y(:).^2./sigma_noise_sq.^2 + y(:)./sigma_noise_sq);
    
    % Eq. 30a - 30c (supplemental)
    for j=1:nfilter
        [~,phiPr] = dphi_f{j}(Jx{j}(:,:,i));
        sf = size(phiPr);
        Ax{i} = Ax{i} + lambdaS * F{j}' * spdiags(phiPr(:),0,prod(sf),prod(sf)) * F{j};
    end
end


end

