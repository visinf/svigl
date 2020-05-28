% Computes the value of the Poisson-Gaussian denoising energy (Eq. 17)
% 
% Inputs
% noisy:        Noisy input image
% clean:        Clean image at which to evaluate energy
% energy_opt:   Parameters of energy function
%
% Outputs
% E:            Energy value
% 
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ E ] = energy( noisy, clean, energy_opt)

Js = getOpt(energy_opt, 'Js');
phi_f = getOpt(energy_opt, 'phi_f');
sigma_a = getOpt(energy_opt, 'sigma_a');
sigma_b = getOpt(energy_opt, 'sigma_b');
lambdaD = getOpt(energy_opt, 'lambdaD');
lambdaS = getOpt(energy_opt, 'lambdaS');

x = clean;
y = noisy;

% Calculate noise standard deviation / variance (Eq. 16)
sigma_noise = sqrt(max(sigma_a*x,0)+sigma_b);

% Data term
D = 0.5 * lambdaD ./ sigma_noise.^2 .* (x - y).^2;
data_term = sum(D(:));

% Smoothness term
nj = numel(Js);
Jxs = cell(nj,1);
phis = zeros(nj,1);
for j=1:nj
    Jxs{j} = conv2(x, Js{j}, 'valid');
    phis(j) = sum(phi_f{j}(Jxs{j}(:)));
end

smoothness_term = lambdaS * sum(phis);

% Energy value
E = data_term + smoothness_term;

end

