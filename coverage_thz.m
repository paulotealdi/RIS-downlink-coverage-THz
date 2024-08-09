clc; clear;

c   = physconst('LightSpeed');
P_A = 1e-3;                     % mW
N_A = 10;                       % Antennas
f   = 0.3e12;                   % 0.3 THz
k_a = 0.075;
LX  = c/(2*f); LY = LX;
R_t = sqrt(140);
h_A = 3;                        % APs height
h_U = 1;                        % User antenna height
h_R = 0.75*h_A;                 % RIS height
RIS_loc = [1;1;h_R];            % 3D RIS location
v_0 = sqrt(2);                  % UE -> RIS 2D distance
r_B = 0.22;                     % Blockage radius
h_B = 1.63;                     % Blockage height
lambda_B = 2;                   % PPP intensity for blockages
lambda_A = 1;                   % PPP intensity for APs
G_A = db2mag(30); G_U = G_A;    % Antenna gain
tau = db2mag(2);
N   = 1e13;                     % RIS number of elements
n_realizations = 1e6;           % Number of Monte-Carlo realizations

[theta_R, rho_R] = cart2pol(RIS_loc(1), RIS_loc(2));
[x_m,y_m] = poisson_pp(0, 0, R_t, lambda_A);
[theta_m,r_m] = cart2pol(x_m,y_m);
phi_m = theta_m - theta_R;

z_m = sqrt(r_m.^2 + v_0.^2 - 2.*r_m.*v_0.*cos(phi_m));

distances_UA = sqrt(x_m.^2 + y_m.^2);
[min, I] = min(distances_UA);
x_0 = x_m(I);
y_0 = y_m(I);
phi_0 = phi_m(I);
[theta_0,r_0] = cart2pol(x_0,y_0);

beta_D = 2*lambda_B*r_B*abs(h_B/h_A);
beta_R = 2*lambda_B*r_B*abs(h_B/h_R);
PD_LOS = @(r) exp(-beta_D*r);           % AP-UE LoS probability
PR_LOS = @(r) exp(-beta_R*r);           % RIS-UE LoS probability

% Pathloss of the AP-UE link
PLD = @(r) G_U*G_A*c.^2./(4*pi*f).^2 .* exp(-k_a.*sqrt(r.^2+h_A.^2)) .* 1./(r.^2+h_A.^2);   

% Pathloss of the AP-RIS-UE link
PLR = @(z) (G_U*G_A*(LX*LY).^2 ./ ((4*pi).^2 .* (z.^2 + (h_A-h_R).^2) .* (v_0.^2+h_R.^2))) .* exp(-k_a.*sqrt(z.^2+(h_A-h_R).^2)) .* exp(-k_a*sqrt(v_0.^2+h_R.^2)) .* (h_A-h_R).^2 ./ (z.^2 + (h_A-h_R).^2);

% User Association Probability
A_D = @(r) exp(-beta_D.*r).*(1-exp(-beta_R.*v_0));
A_R = @(r) (1-exp(-beta_R.*r)).*exp(-beta_R.*v_0);
A_C = @(r) exp(-beta_D.*r).*exp(-beta_R.*v_0);

k_D = 1/(2*N_A.^2);      % |f_j|.^2 = 1

mu_B = pi/2;
sig_B2 = 2.^2 * (1-pi.^2/16);
k_I = mu_B/(2*N_A*sig_B2);
k_R = @(z) 1./(2*N.^2.*(P_A.*PLR(z).*N_A).^2);
k_C = @(r,z) 1./((sqrt(2).*(N.*PLR(z) + PLD(r)).*N_A).^2);
Q = @(x) exp(-k_I.*x./(1+2.*x))./sqrt(1+2.*x);

integ_func_L_ID = @(r,s) (1 - k_D./(k_D+s.*P_A.*PLD(r))).*PD_LOS(r).*r;
integ_func_L_IR = @(r,phi,s) (1 - Q(s.*P_A.*PLR(sqrt(r.^2 + v_0.^2 - 2.*r.*v_0.*cos(phi))))).*r;

L_ID_param = tau .* k_D ./ (P_A .* PLD(r_0));

L_ID = @(s) exp(-2*pi*lambda_A*integral(@(r) integ_func_L_ID(r,s),r_0,R_t,'ArrayValued',true));
L_IR = @(s) exp(-2*pi*lambda_A*integral2(@(r,phi) integ_func_L_IR(r,phi,s),r_0,R_t,0,pi));

z_0 = @(r,phi) sqrt(r.^2+v_0.^2-2.*r.*v_0.*cos(phi));

PD = @(r) L_ID(tau .* k_D ./ (P_A .* PLD(r)));
PR = @(r,phi) L_ID(tau.*k_R(z_0(r,phi))).*L_IR(tau.*k_R(z_0(r,phi)));
PC = @(r,phi) L_ID(tau.*k_C(r,z_0(r,phi))).*L_IR(tau.*k_C(r,z_0(r,phi)));

f = @(r) 2.*pi.*lambda_A.*r.*exp(-lambda_A.*pi.*r.^2);
integ_func_P_cov = @(r,phi) f(r).*(A_D(r).*PD(r) + A_R(r).*PR(r,phi) + A_C(r).*PC(r,phi));

P_cov = (1/pi)*integral2(@(a,b) integ_func_P_cov(a,b),0,R_t,0,pi);


