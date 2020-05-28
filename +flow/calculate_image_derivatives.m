% Calculates image derivatives wrt current flow estimate for EpicFlow energy.
%
% Inputs
% im1, im2:     Input images
% flow:      	Current flow estimate
%
% Outputs
% I_:           Image derivatives
%
% Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
%
% This file is part of the implementation as described in the CVPR 2018 paper:
% Tobias Plötz, Anne S. Wannenwetsch, and Stefan Roth, Stochastic variational inference with gradient linearization.
% Please see the file LICENSE.txt for the license governing this code.

function [It,Ix,Iy,Ixx,Ixy,Iyx,Iyy,Ixt,Iyt] = calculate_image_derivatives(im1,im2,flow)

    % Warp second image wrt current flow
    [im2_warped, out_of_boundary] = warp_image(im2, flow);

    % Compute temporal difference between first and second image
    It = im2_warped - im1;

    % Calculate first and second order derivatives on averaged image
    averaged_im = 0.5 * im1 + 0.5 * im2_warped;
    derivative_filter = [1 -8 0 8 -1]/12;
    Ix = imfilter(averaged_im,derivative_filter, 'replicate');
    Iy = imfilter(averaged_im,derivative_filter', 'replicate');
    Ixx = imfilter(Ix,derivative_filter, 'replicate');
    Ixy = imfilter(Ix,derivative_filter', 'replicate');
    Iyx = Ixy;
    Iyy = imfilter(Iy,derivative_filter', 'replicate');
    Ixt = imfilter(It,derivative_filter, 'replicate');
    Iyt = imfilter(It,derivative_filter', 'replicate');

    % Set derivatives of all out of boundary pixels to zero
    It(out_of_boundary) = 0;
    Ix(out_of_boundary) = 0;
    Iy(out_of_boundary) = 0;
    Ixx(out_of_boundary) = 0;
    Ixy(out_of_boundary) = 0;
    Iyx(out_of_boundary) = 0;
    Iyy(out_of_boundary) = 0;
    Ixt(out_of_boundary) = 0;
    Iyt(out_of_boundary) = 0;
end

% Warp image according to flow
function [im_warped, out_of_boundary] = warp_image(im, flow)
    % Compute sample locations
    imDim = size(im);
    [x, y] = meshgrid(1:imDim(2), 1:imDim(1));
    xx = x + flow(:,:,1);
    yy = y + flow(:,:,2);

    % Determine out of boundary pixels
    out_of_boundary = (xx<1) | (yy<1) | (xx>imDim(2)) | (yy>imDim(1));
    out_of_boundary = repmat(out_of_boundary, [1,1,3]);

    % Determine bilinear coefficients and surrounding pixel locations
    x1 = floor(xx);
    y1 = floor(yy);
    dx = xx-x1;
    dy = yy-y1;

    x2 = crop_flow(x1+1,imDim(2));
    x1 = crop_flow(x1, imDim(2));
    y2 = crop_flow(y1+1, imDim(1));
    y1 = crop_flow(y1, imDim(1));
    idx0 = sub2ind(imDim, y, x);
    idx1 = sub2ind(imDim, y1, x1);
    idx2 = sub2ind(imDim, y1, x2);
    idx3 = sub2ind(imDim, y2, x1);
    idx4 = sub2ind(imDim, y2, x2);

    % Perform bilinear interpolation
    im_warped = zeros(imDim);
    for i = 1:imDim(3)
        current_channel = im(:,:,i);
        current_channel(idx0) = current_channel(idx1).*(1-dx).*(1-dy) + current_channel(idx2).*dx.*(1-dy) + current_channel(idx3).*(1-dx).*dy + current_channel(idx4).*dx.*dy;
        im_warped(:,:,i)=current_channel;
    end
end

function a = crop_flow(a,b)
    a(a<1) = 1;
    a(a>b) = b;
end
