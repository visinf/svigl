% Calculates Monte Carlo estimate of KL divergence KL(q||p)
%
% Inputs
% noisy:        Noisy input image
% mu:           Mean of variational approximation
% sigma:        Marginal standard deviations of variational approximation
% zsamples:     Samples of standard normal distribution to be used in MC
%               approximation
% energy_opt:   Parameters of energy function
%
% Outputs
% kl:            Monte Carlo estimate of (unnormalized) KL divergence
%
% Author: Tobias Pl??tz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Pl??tz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ kl ] = sampled_kl(  noisy, mu, sigma, zsamples, energy_opt)

[n,m,nsamples] = size(zsamples);

% Transform samples of N(0,I) to samples of q
cleansamples = bsxfun(@plus, bsxfun(@times, zsamples, sigma), mu);

% MC estimate of E_q (log p) up to normalization constant
energies = zeros(nsamples,1);
for i=1:nsamples
    energies(i) = pgdenoising.energy(noisy, cleansamples(:,:,i),energy_opt);
end
energy = mean(energies);

% Entropy of Gaussian
entropy = 0.5*(n*m*log(2*pi*exp(1)) + sum(log(sigma(:).^2)));

kl = energy - entropy;

end
