% Sets up parameters of energy for optical flow estimation
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Pl??tz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ energy_opt ] = opticalflow_energyopt()
energy_opt = struct;
energy_opt.precomp = hstruct(struct);

% Temperature parameter
T = 0.0001;

% Weights for data term and smoothness term
lambdaD = 1/T;
lambdaS = lambdaD;
sfac = 0.5; % obtained through Bayesian optimization
energy_opt.lambdaS = sfac*lambdaS;
energy_opt.lambdaD = lambdaD;

% Parameters of optical flow function
energy_opt.kappa = 5;
energy_opt.epsilon_data_term = 0.01;

%% Pairwise models with generalized Charbonnier experts
% Filters
energy_opt.filters = {[1 -1]',[1 -1]}; % 4-connected pairwise MRF

% Parameters of generalized Charbonnier
gca_D = 0.7; % obtained through Bayesian optimization
gcc_D = 1e-3; % obtained through Bayesian optimization
gca_S = 1.0; % obtained through Bayesian optimization
gcc_S = 1e-3; % obtained through Bayesian optimization

% Set up generalized Charbonnier functions
energy_opt.phi_D = @(x)gcharbonnier(x,gca_D,gcc_D);
energy_opt.dphi_D = @(x)grad_gcharbonnier(x,gca_D,gcc_D);
energy_opt.phi_S = @(x)gcharbonnier(x,gca_S,gcc_S);
energy_opt.dphi_S = @(x)grad_gcharbonnier(x,gca_S,gcc_S);

end

