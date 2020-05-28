% Utility function for parsing options
%
% Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [ val, wasset ] = getOpt( options, field, default )

if isfield(options, field)
    val = options.(field);
    wasset = true;
else
    val = default;
    wasset = false;
end

end

