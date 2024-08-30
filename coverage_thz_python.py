# 3.12.3 ('base': conda)

import scipy as sp
import matplotlib.pyplot as plt
import torch
from torchquad import Simpson, set_up_backend
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_up_backend("torch", data_type="float32")

c = sp.constants.c
p_A = 1*10**(-3)                    
n_A = 10                       
f   = 0.3*10**12                   
k_a = 0.075
lX  = c/(2*f); lY = lX
R_t = torch.sqrt(torch.tensor([140]))
h_A = 3                        
h_U = 1                        
h_R = 0.75*h_A                 
RIS_loc = torch.tensor([1, 1, h_R])            
v_0 = torch.sqrt(torch.tensor([2]))                  
r_B = 0.22                     
h_B = 1.63                     
lambda_B = 2              
lambda_A = 1              
g_A = 10**(30/10); g_U = g_A
tau = 10**(2/10)
n   = 1e13            
n_realizations = 1e6  

def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.arctan(y/x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * torch.cos(phi[0])
    y = rho * torch.sin(phi[0])
    return(x, y)

def poisson_pp(xx0, yy0, radius, lambda0):
    lambda0 = torch.tensor([lambda0])
    areaTotal = torch.pi*radius**2
    numbPoints = torch.poisson(lambda0*areaTotal)
    phi=2*torch.pi*torch.rand(1,int(numbPoints.item()))
    rho=radius*torch.sqrt(torch.rand(1,int(numbPoints.item())))
    xx, yy = pol2cart(rho,phi)
    xx=xx+xx0; yy=yy+yy0
    return (xx,yy)

def z(r,phi):
    return torch.sqrt(r**2+v_0**2-2*r*v_0*torch.cos(phi))

xm, ym = poisson_pp(0, 0, R_t, lambda_A)
rho_R, phi_R = cart2pol(torch.tensor([RIS_loc[0]]), torch.tensor([RIS_loc[1]]))
rho_m, theta_m = cart2pol(xm, ym)
phi_m = theta_m - phi_R

def calc(lambda_B=lambda_B, lambda_A=lambda_A, v_0=v_0, h_R=h_R, tau=tau): 
    h_B_hat = h_B - h_U
    h_R_hat = h_R - h_U
    h_A_hat = h_A - h_U

    distances_UA = torch.sqrt(xm**2 + ym**2)
    index_min = torch.argmin(distances_UA)
    x0 = xm[0][index_min.item()].item()
    y0 = ym[0][index_min.item()].item()
    r_0, theta_0 = cart2pol(torch.tensor([x0]), torch.tensor([y0]))
    r_0,theta_0 = r_0.item(), theta_0.item()
    beta_D = torch.tensor([2*lambda_B*r_B*abs(h_B_hat/h_A_hat)])
    beta_R = torch.tensor([2*lambda_B*r_B*abs(h_B_hat/h_R_hat)])

    pdLOS = lambda r : torch.exp(-beta_D*r)
    prLOS = torch.exp(-1*beta_R*v_0)

    def pld(r):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        return (g_U*g_A*c**2/(4*torch.pi*f)**2)*torch.exp(-k_a*torch.sqrt(r**2+h_A_hat**2))/(r**2+h_A_hat**2)

    def plr(z):
        if not torch.is_tensor(z):
            z = torch.tensor([z])
        return (g_U*g_A*(lX*lY)**2 / ((4*torch.pi)**2 * (z**2 + (h_A_hat-h_R_hat)**2) * (v_0**2+(h_R_hat)**2))) * torch.exp(-1*k_a*torch.sqrt(z**2+(h_A_hat-h_R_hat)**2)) * torch.exp(-1*k_a*torch.sqrt(v_0**2+(h_R_hat)**2)) * (h_A_hat-h_R_hat)**2/(z**2+(h_A_hat-h_R_hat)**2)

    aD = lambda r : torch.exp(-beta_D*r)*(1-torch.exp(-beta_R*v_0))
    aR = lambda r : (1-torch.exp(-1*beta_D*r))*torch.exp(-1*beta_R*v_0)
    aC = lambda r : torch.exp(-1*beta_D*r)*torch.exp(-1*beta_R*v_0)

    k_D = 1/(2*n_A**2)

    mu_B = torch.pi/2
    sig_B2 = 2**2 * (1-torch.pi**2/16)
    k_I = mu_B/(2*n_A*sig_B2)
    k_R = lambda z : 1/(2*n**2*(p_A*plr(z)*n_A)**2)
    k_C = lambda r,z : 1/(torch.sqrt(torch.tensor([2]))*(n*plr(z) + pld(r))*n_A)**2
    Q = lambda x : torch.exp(-1*k_I*x/(1+2*x))/torch.sqrt(1+2*x)

    integ_func_L_ID = lambda r,s : (1-k_D/(k_D + s*p_A*pld(r)))*pdLOS(r)*r
    integ_func_L_IR = lambda r,phi,s : (1-Q(s*p_A*plr(z(r,phi))))*r

    simp = Simpson()

    def lID(s,N=501):
        integral = simp.integrate(lambda r : integ_func_L_ID(r,s),dim=1,N=N,integration_domain=[[r_0,R_t]])
        return torch.exp(-2*torch.pi*lambda_A*integral)

    def lIR(s,N=351**2):
        integration_domain = [[r_0,R_t],[0,torch.pi]]
        integrand = lambda d: integ_func_L_IR(d[:,0],d[:,1],s)
        return torch.exp(-2*torch.pi*lambda_A*simp.integrate(integrand,dim=2,N=N,integration_domain=integration_domain))

    def pD(r):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        lID_arg = tau*k_D/(p_A*pld(r))
        return lID(lID_arg) 
    
    def pR(r,phi):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        if not torch.is_tensor(phi):
            phi = torch.tensor([phi])
        return lID(tau*k_R(z(r,phi)))*lIR(tau*k_R(z(r,phi)))

    def pC(r,phi):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        if not torch.is_tensor(phi):
            phi = torch.tensor([phi])
        return lID(tau*k_C(r,z(r,phi)))*lIR(tau*k_C(r,z(r,phi)))

    g = lambda r : 2*torch.pi*lambda_A*r*torch.exp(-lambda_A*torch.pi*r**2)
    integ_func_P_cov = lambda r,phi,active_D,active_R,active_C : g(r)*(active_D*aD(r)*pD(r) + active_R*aR(r)*pR(r,phi) + active_C*aC(r)*pC(r,phi))

    def solve(active_D,active_R,active_C,N=351**2):
        integration_domain = [[0,R_t],[0,torch.pi]]
        integrand = lambda d : integ_func_P_cov(d[:,0], d[:,1],active_D,active_R,active_C)
        return (1/torch.pi)*simp.integrate(integrand,dim=2,N=N,integration_domain=integration_domain)

    P_covD = solve(1,0,0)
    P_covR = solve(0,1,0)
    P_covC = solve(0,0,1)
    P_covOverall = solve(1,1,1)
    return (P_covD, P_covR, P_covC, P_covOverall)

# FIGURA 2(b) #######################################################################
b_lambdas = torch.arange(0.5,20,0.5)
results = torch.tensor([calc(torch.tensor([l])) for l in b_lambdas])
results_D = results[:,0]
results_R = results[:,1]
results_C = results[:,2]
results_overall = results[:,3]
plt.plot(b_lambdas.cpu(), results_D.cpu())
plt.plot(b_lambdas.cpu(), results_R.cpu())
plt.plot(b_lambdas.cpu(), results_C.cpu())
plt.plot(b_lambdas.cpu(), results_overall.cpu())
plt.legend(["Direct","RIS","Composite","Overall"])
plt.grid()
plt.show()

# FIGURA 3(a) #######################################################################
# v0s = torch.arange(0.25,6,0.2)
# results8 = torch.tensor([calc(v_0=torch.tensor([l]),h_R=0.8*h_A) for l in v0s])
# results7 = torch.tensor([calc(v_0=torch.tensor([l]),h_R=0.7*h_A) for l in v0s])
# results6 = torch.tensor([calc(v_0=torch.tensor([l]),h_R=0.6*h_A) for l in v0s])
# results_overall8 = results8[:,3]
# results_overall7 = results7[:,3]
# results_overall6 = results6[:,3]
# plt.plot(v0s.cpu(), results_overall8.cpu())
# plt.plot(v0s.cpu(), results_overall7.cpu())
# plt.plot(v0s.cpu(), results_overall6.cpu())
# plt.show()

# FIGURA 3(b) #######################################################################
# lAs = torch.arange(1,10,0.5)
# results2 = torch.tensor([calc(lambda_A=torch.tensor([l]),tau=10**(2/10)) for l in lAs])
# results4 = torch.tensor([calc(lambda_A=torch.tensor([l]),tau=10**(4/10)) for l in lAs])
# results6 = torch.tensor([calc(lambda_A=torch.tensor([l]),tau=10**(6/10)) for l in lAs])
# results_overall2 = results2[:,8]
# results_overall4 = results4[:,8]
# results_overall6 = results6[:,8]
# plt.plot(lAs.cpu(), results_overall2.cpu())
# plt.plot(lAs.cpu(), results_overall4.cpu())
# plt.plot(lAs.cpu(), results_overall6.cpu())
# plt.show()

# results_D = results[:,0]
# beta_D_values = results[:,4]
# pdLOS_values = results[:,5]
# pD_values = results[:,6]

# # Generating the plots
# plt.figure(figsize=(14, 10))

# plt.subplot(2, 2, 1)
# plt.plot(b_lambdas.cpu(), results_D.cpu(), label='P_covD')
# plt.grid()
# plt.title('P_covD vs. Lambda_B')
# plt.xlabel('Lambda_B')
# plt.ylabel('P_covD')

# plt.subplot(2, 2, 2)
# plt.plot(b_lambdas.cpu(), beta_D_values.cpu(), label='Beta_D', color='orange')
# plt.grid()
# plt.title('Beta_D vs. Lambda_B')
# plt.xlabel('Lambda_B')
# plt.ylabel('Beta_D')

# plt.subplot(2, 2, 3)
# plt.plot(b_lambdas.cpu(), pdLOS_values.cpu(), label='pdLOS(r_0)', color='green')
# plt.grid()
# plt.title('pdLOS(r_0) vs. Lambda_B')
# plt.xlabel('Lambda_B')
# plt.ylabel('pdLOS(r_0)')

# plt.subplot(2, 2, 4)
# plt.plot(b_lambdas.cpu(), pD_values.cpu(), label='pD(r_0)', color='red')
# plt.grid()
# plt.title('pD(r_0) vs. Lambda_B')
# plt.xlabel('Lambda_B')
# plt.ylabel('pD(r_0)')

# plt.tight_layout()
# plt.show()