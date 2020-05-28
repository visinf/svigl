% Evaluates generalized Charbonnier function
%
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.
function [ f ] = gcharbonnier( x, a, c )

if a == 0
    f = log(0.5 * (x./c).^2 + 1);
else
    za = max(1, 2-a);

    f = za .*(c) ./ a * (((x./c).^2 / za +1).^(a./2) -1);
end

end

