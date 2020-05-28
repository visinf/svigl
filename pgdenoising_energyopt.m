% Sets up parameters of energy for Poisson-Gaussian denoising 
%
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ energy_opt ] = pgdenoising_energyopt()
energy_opt = struct;
energy_opt.precomp = hstruct(struct);

% Temperature parameter
T = 0.3;

% Weights for data term and smoothness term
lambdaD = 1/T;
lambdaS = lambdaD;
sfac = 10^2.112928; % obtained through Bayesian optimization
energy_opt.lambdaS = sfac*lambdaS;
energy_opt.lambdaD = lambdaD;


% Parameters of noise level function
energy_opt.sigma_a = 0.05;
energy_opt.sigma_b = 0.0001;

%% Pairwise models with generalized Charbonnier experts
% Filters
Js = {[1 -1], [1 -1]'}; % 4-connected pw MRF

% Parameters of generalized Charbonnier
gcafac = 0.635723; % obtained through Bayesian optimization
gcas = gcafac*ones(numel(Js),1);
gccs = (1e-5)*ones(numel(Js),1);
nj = numel(Js);

% Set up expert functions as generalized Charbonniers
energy_opt.Js = Js;
energy_opt.phi_f = cell(nj,1);
energy_opt.dphi_f = cell(nj,1);
for j=1:nj
    energy_opt.phi_f{j} = @(x) gcharbonnier(x, gcas(j), gccs(j));
    energy_opt.dphi_f{j} = @(x) grad_gcharbonnier(x, gcas(j), gccs(j));
    energy_opt.d2phi_f{j} = @(x) grad2_gcharbonnier(x, gcas(j), gccs(j));
end


end

