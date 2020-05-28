% Gradient of generalized Charbonnier
%
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ df, dfgl ] = grad_gcharbonnier( x,a,c )

if a == 0
    dfgl = 2./(x.^2 + 2 * c^2);
    df = x.*dfgl;
else
    za = max(1, 2-a);

    dfgl = 1./(c) .* ( (x./c).^2 ./ za +1) .^ (a./2 -1);
    df = x.*dfgl;
end
end

