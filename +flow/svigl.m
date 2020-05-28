% General code that implements SVIGL optimization
%
% Inputs
% im1, im2:     Input images
% mu_init:      Initial value for mu
% sigma_init:   Initial value for sigma
% energy_opt:   Parameters of energy function
% opt:          Hyperparameters of optimization
%
% Outputs
% mu:           Mean of variational approximation
% sigma:        Marginal standard deviations of variational approximation
% fval:         Monte Carlo estimate of (unnormalized) KL divergence
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ mu, sigma, fval] = svigl(im1, im2, mu_init, sigma_init, energy_opt, opt)

if ~exist('opt', 'var')
    opt = struct;
end

% Parse optimization parameters
niter = getOpt(opt, 'niter', 20);
nsamples = getOpt(opt, 'nsamples', 50);
fixed_zsamples = getOpt(opt, 'zsamples', []);

mu = mu_init;
sigma = sigma_init;

[n,m,~] = size(mu);

for i=1:niter
    % Sample from N(0,I)
    if isempty(fixed_zsamples)
        zsamples = randn(n,m,2,nsamples);
    else
        zsamples = fixed_zsamples;
    end

    % Run one update step
    [mu, sigma] = flow.update_mu_sigma(im1, im2, mu, sigma, zsamples, energy_opt);

    % Evaluate current KL divergence
    fval = flow.sampled_kl(im1, im2, mu, sigma, zsamples, energy_opt);

    fprintf('Iter %d: Function val %d \t Mean sigma %.4f \n', i, fval, mean(sigma(:)));
end

end
