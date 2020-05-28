% Computes the linearized gradient of the KL divergence wrt. mu and sigma
%
% Inputs
% im1, im2:     Input images
% mu:           Current value for mu
% sigma:        Current value for sigma
% zsamples:     Samples of standard normal distribution to be used in MC
%               approximation of the gradient of the KL divergence
% energy_opt:   Parameters of energy function
%
% Outputs
% A, b:         Parameterization of the linearized gradient
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ A, b ] = compute_linearized_gradient_mu_sigma(im1, im2, mu, sigma, zsamples, energy_opt)

[n,m,~,nsamples] = size(zsamples);

% Transform samples of N(0,I) to samples of q
cleansamples = bsxfun(@plus, bsxfun(@times, zsamples, sigma), mu);

% Get linearized energy gradient wrt. samples
[Ac, Ax, bc, bx] = flow.compute_linearized_gradient_energy(im1, im2, mu, cleansamples, energy_opt);

A_mu_mu = sparse(2*n*m, 2*n*m);
A_mu_sigma = sparse(2*n*m, 2*n*m);
A_sigma_mu = sparse(2*n*m, 2*n*m);
A_sigma_sigma = spdiags(2./(sigma(:).^2),0,2*n*m,2*n*m); % Due to linearization of entropy gradient (Eq. 11c)

b_mu = bc;
b_sigma = -3./(sigma(:)); % Due to linearization of entropy gradient (Eq. 11c)

z = cell(nsamples,1);
for i=1:nsamples
    z{i} = spdiags(flatMat(zsamples(:,:,:,i)),0,2*n*m,2*n*m);
end

% Lift linearization of energy gradient to linearization of KL gradient
for i=1:nsamples
    A_mu_mu = A_mu_mu + 1/nsamples * (Ac+Ax{i}); % Eq. 9c
    A_mu_sigma = A_mu_sigma + 1/nsamples * (Ac+Ax{i}) * z{i}; % Eq. 9c
    A_sigma_mu = A_sigma_mu + 1/nsamples * z{i} * (Ac+Ax{i}); % Eq. 11c
    A_sigma_sigma = A_sigma_sigma + 1/nsamples *  z{i} * (Ac+Ax{i}) * z{i}; % Eq. 11c

    b_mu = b_mu + 1/nsamples * bx{i}; % Eq. 9c
    b_sigma = b_sigma + 1/nsamples * flatMat(zsamples(:,:,:,i)) .* (bc + bx{i}); % Eq. 11c
end

% Eq. 13
A = [A_mu_mu, A_mu_sigma;
    A_sigma_mu, A_sigma_sigma];
b = [b_mu;
    b_sigma];

end
