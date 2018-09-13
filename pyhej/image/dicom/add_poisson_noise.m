% https://github.com/xinario/SAGAN/tree/master/poisson_noise_simulation
% add_poisson_noise -- add poisson noise to a ct slice.
%   Args:
%   @ac        : attenuation coefficients of a single ct slice
%   @N         : x-ray source influx
%   Author     : Xin Yi (xiy525@mail.usask.ca)
%   Date       : 03/22/2017
function ac_noise = add_poisson_noise(ac, N)
dtheta = 0.3;
dsensor = 0.1;
D = 500;

sinogram_in = fanbeam(ac, D, 'FanSensorSpacing', dsensor, 'FanRotationIncrement', dtheta);

% small number >0 that reflects the smallest possible detected photon count
epsilon = 5;

% to detector count coefficients unit is cm-1
sinogramCT = N * exp(-sinogram_in*0.0625);

% add poison noise
sinogramCT_noise = poissrnd(sinogramCT);

sinogram_out = -log(sinogramCT_noise/N)/0.0625;

idx = isinf(sinogram_out);
sinogram_out(idx) = -log(epsilon/N)/0.0625;

ac_noise = ifanbeam(sinogram_out, D, 'FanSensorSpacing', dsensor, 'OutputSize', max(size(ac)));