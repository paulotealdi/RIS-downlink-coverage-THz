% Poisson Point Process
% Arguments x0, y0, radius, lambda
% x0 and y0 -> center of disk
% radius -> radius of disk
% lambda -> intensity of PPP

function [x,y] = poisson_pp(x0, y0, radius, lambda)
    area_cons = pi*radius^2;
    lambda_dash = lambda.*area_cons;
    n_point = poissrnd(lambda_dash);
    distance = radius*sqrt(rand(1,n_point)); 
    angle = 2*pi*rand(1,n_point);
    [x,y] = pol2cart(angle,distance);
    x = x + x0;
    y = y + y0;
end