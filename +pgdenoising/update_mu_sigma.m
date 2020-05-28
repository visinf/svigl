% Run one update step for mu and sigma
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
% mu_new:       New value for mu
% sigma_new:    New value for sigma
% 
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ mu_new, sigma_new ] = update_mu_sigma( noisy, mu, sigma, zsamples, energy_opt)

sor_iter = getOpt(energy_opt, 'sor_iter', 100);

[n,m, ~] = size(zsamples);

% Get linearized gradient wrt. mu and sigma
% [A, b] = pgdenoising.compute_linearized_gradient_mu_sigma( noisy, mu, sigma, zsamples, energy_opt);
% This is an alternative implementation that is more memory efficient
[A, b] = pgdenoising.compute_linearized_gradient_mu_sigma_integrated( noisy, mu, sigma, zsamples, energy_opt);

% Solve for the new iterate of mu and sigma by solving the linear system of
% equations (Eq. 12)
% theta_new = A\(-b);
theta_new = sor(A', -b, 1.95, sor_iter, 1e-5, [mu(:); sigma(:)]);

mu_new = reshape(full(theta_new(1:n*m)), n,m);
sigma_new = sigma + 1*(reshape(full(theta_new(n*m+1:end)), n,m)-sigma);

% Fix to avoid negative sigmas
sigma_new = abs(sigma_new);

end

