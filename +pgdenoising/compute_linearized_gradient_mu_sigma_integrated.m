% Computes the linearized gradient of the KL divergence wrt. mu and sigma
% in a more memory efficient manner
% 
% Inputs
% noisy:        Noisy input image
% mu:           Current value for mu
% sigma:        Current value for sigma
% zsamples:     Samples of standard normal distribution to be used in MC
%               approximation of the gradient of the KL divergence
% energy_opt:   Parameters of energy function
%
% Outputs
% A, b:         Parameterization of the linearized gradient
% 
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ A, b ] = compute_linearized_gradient_mu_sigma_integrated( noisy, mu, sigma, zsamples, energy_opt)
%COMPUTE_LINEARIZED_GRADIENT_MU_SIGMA Summary of this function goes here
%   Detailed explanation goes here

[n,m, nsamples] = size(zsamples);

cleansamples = bsxfun(@plus, bsxfun(@times, zsamples, sigma), mu);

Js = getOpt(energy_opt, 'Js');
dphi_f = getOpt(energy_opt, 'dphi_f');
lambdaD = getOpt(energy_opt, 'lambdaD');
lambdaS = getOpt(energy_opt, 'lambdaS');
sigma_a = getOpt(energy_opt, 'sigma_a');
sigma_b = getOpt(energy_opt, 'sigma_b');
precomp = getOpt(energy_opt, 'precomp', []);

x = cleansamples;
y = noisy;

F = getOpt(precomp.data, 'F', []);


[n,m, nsamples] = size(x);
nfilter = numel(Js);

if isempty(F)
    F = cell(nfilter,1);
    for j=1:nfilter
        F{j} = make_convn_mat(Js{j}, [n m], 'valid');
    end
    precomp.data.F = F;
end
Jx = cell(nfilter,1);
for j=1:nfilter
    Jx{j} = convn(x, Js{j}, 'valid');
end

Ac = sparse(n*m, n*m);
bc = 0;

A_mu_mu = sparse(n*m, n*m);
A_mu_sigma = sparse(n*m, n*m);
A_sigma_mu = sparse(n*m, n*m);
A_sigma_sigma = spdiags(2./(sigma(:).^2),0,n*m,n*m);

b_mu = bc;
b_sigma = -3./(sigma(:));

z = cell(nsamples,1);
for i=1:nsamples
    z{i} = spdiags(flatMat(zsamples(:,:,i)),0,n*m,n*m);
end


for i=1:nsamples
    x_i = flatMat(x(:,:,i));
    sigma_noise = sqrt(max(sigma_a*x_i,0)+sigma_b);
    sigma_noise_sq = sigma_noise.^2;
    
    Ax = lambdaD.*(1./sigma_noise_sq-0.5*sigma_a*x_i./sigma_noise_sq.^2+sigma_a*y(:)./sigma_noise_sq.^2);
    Ax = spdiags(Ax(:), 0, numel(Ax), numel(Ax));
    bx = -lambdaD.*(0.5*sigma_a*y(:).^2./sigma_noise_sq.^2 + y(:)./sigma_noise_sq);
    for j=1:nfilter
        [~,phiPr] = dphi_f{j}(Jx{j}(:,:,i));
        sf = size(phiPr);
        Ax = Ax + lambdaS * F{j}' * spdiags(phiPr(:),0,prod(sf),prod(sf)) * F{j};
    end
    
    A_mu_mu = A_mu_mu + 1/nsamples * (Ac+Ax);
    A_mu_sigma = A_mu_sigma + 1/nsamples * (Ac+Ax) * z{i};
    A_sigma_mu = A_sigma_mu + 1/nsamples * z{i} * (Ac+Ax);
    A_sigma_sigma = A_sigma_sigma + 1/nsamples *  z{i} * (Ac+Ax) * z{i};
    
    b_mu = b_mu + 1/nsamples * bx;
    b_sigma = b_sigma + 1/nsamples * flatMat(zsamples(:,:,i)) .* (bc + bx);
end
clear Ac

A = [A_mu_mu, A_mu_sigma;
    A_sigma_mu, A_sigma_sigma];

clear A_mu_mu A_mu_sigma A_sigma_mu A_sigma_sigma
b = [b_mu;
    b_sigma];

end

