% Calculates spatially dependent lambdaS for EpicFlow energy.
%
% Inputs
% im:       Reference image
% kappa:    Scale factor
%
% Outputs
% lambda:   Spatially varying lambda parameter
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ lambda ] = calculate_spatial_lambdaS(im,kappa)
    derivative_Filter = [1 -8 0 8 -1]/12;
    luminance = (0.299 * im(:,:,1) + 0.587 * im(:,:,2) + 0.114 * im(:,:,3))/255;
    Ix = imfilter(luminance, derivative_Filter,  'conv', 'replicate', 'same');
    Iy = imfilter(luminance, derivative_Filter',  'conv', 'replicate', 'same');
    lambda = exp(- kappa * sqrt(Ix .* Ix + Iy .* Iy));
end
