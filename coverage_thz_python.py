# 3.12.3 ('base': conda)

import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
import torch
from torchquad import Simpson, set_up_backend
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_up_backend("torch", data_type="float32")

c = sp.constants.c
p_A = 1e-3
n_A = 10
f   = 0.3e12
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

# xm, ym = poisson_pp(0, 0, R_t, lambda_A) # To generate a poisson point process realization

def hvs(x,r):
    x = x - r
    if not torch.is_tensor(x):
        x = torch.tensor([x])
    return torch.heaviside(x,torch.tensor([1.]))

def calc(lambda_B=lambda_B, lambda_A=lambda_A, v_0=v_0, h_R=h_R, tau=tau, n=n, h_B=h_B, activeRIS=False): 
    h_B_hat = h_B - h_U
    h_R_hat = h_R - h_U
    h_A_hat = h_A - h_U

    beta_D = torch.tensor([2*lambda_B*r_B*abs(h_B_hat/h_A_hat)])
    beta_R = torch.tensor([2*lambda_B*r_B*abs(h_B_hat/h_R_hat)])

    pdLOS = lambda r : torch.exp(-beta_D*r)
    prLOS = torch.exp(-beta_R*v_0)

    def z(r,phi):
        return torch.sqrt(r**2+v_0**2-2*r*v_0*torch.cos(phi))

    def pld(r):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        return ((g_U*g_A*c**2)/(4*torch.pi*f)**2)*torch.exp(-k_a*torch.sqrt(r**2+h_A_hat**2))/(r**2+h_A_hat**2)

    def plr(z):
        if not torch.is_tensor(z):
            z = torch.tensor([z])
        if activeRIS:
            return ((10**(30/10))*g_U*g_A*(lX*lY)**2 / ((4*torch.pi)**2 * (z**2 + (h_A_hat-h_R_hat)**2) * (v_0**2+(h_R_hat)**2))) * torch.exp(-k_a*torch.sqrt(z**2+(h_A_hat-h_R_hat)**2)) * torch.exp(-k_a*torch.sqrt(v_0**2+(h_R_hat)**2)) * (h_A_hat-h_R_hat)**2/(z**2+(h_A_hat-h_R_hat)**2)
        return ((g_U*g_A*(lX*lY)**2) / ((4*torch.pi)**2 * (z**2 + (h_A_hat-h_R_hat)**2) * (v_0**2+(h_R_hat)**2))) * torch.exp(-k_a*torch.sqrt(z**2+(h_A_hat-h_R_hat)**2)) * torch.exp(-k_a*torch.sqrt(v_0**2+(h_R_hat)**2)) * (h_A_hat-h_R_hat)**2/(z**2+(h_A_hat-h_R_hat)**2)

    aD = lambda r : torch.exp(-beta_D*r)*(1-torch.exp(-beta_R*v_0))
    aR = lambda r : (1-torch.exp(-1*beta_D*r))*torch.exp(-1*beta_R*v_0)
    aC = lambda r : torch.exp(-1*beta_D*r)*torch.exp(-1*beta_R*v_0)

    k_D = 1/(2*n_A**2)

    mu_B = torch.pi/2
    sig_B2 = 4*(1-(torch.pi**2)/16)
    k_I = mu_B/(2*n_A*sig_B2)
    k_R = lambda z : 1/(2*(n**2)*(p_A*plr(z)*n_A)**2)
    k_C = lambda r,z : 1/(torch.sqrt(torch.tensor([2]))*(n*plr(z) + pld(r))*n_A)**2
    Q = lambda x : torch.exp(-1*k_I*x/(1+2*x))/torch.sqrt(1+2*x)

    integ_func_L_ID = lambda r,s : (1-k_D/(k_D + s*p_A*pld(r)))*pdLOS(r)*r
    integ_func_L_IR = lambda r,phi,s : (1-Q(s*p_A*plr(z(r,phi))))*r

    simp = Simpson()

    def lID(s,r0,N=701):
        integration_domain = [[0, R_t]] # heaviside "hvs(r)" takes care of the integration domain
        integral = simp.integrate(lambda r : integ_func_L_ID(r,s)*hvs(r0,r),dim=1,N=N,integration_domain=integration_domain)
        return torch.exp(-2*torch.pi*lambda_A*integral)
        
    def lIR(s,r0,N=101**2):
        integration_domain = [[0,R_t],[0,torch.pi]] # heaviside "hvs(r)" takes care of the integration domain
        integrand = lambda d: integ_func_L_IR(d[:,0],d[:,1],s)*hvs(r0,d[:,0])
        return torch.exp(-2*torch.pi*lambda_A*simp.integrate(integrand,dim=2,N=N,integration_domain=integration_domain))

    def pD(r):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        lID_arg = tau*k_D/(p_A*pld(r))
        return lID(lID_arg,r) 
    
    def pR(r,phi):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        if not torch.is_tensor(phi):
            phi = torch.tensor([phi])
        return lID(tau*k_R(z(r,phi)),r)*lIR(tau*k_R(z(r,phi)),r)

    def pC(r,phi):
        if not torch.is_tensor(r):
            r = torch.tensor([r])
        if not torch.is_tensor(phi):
            phi = torch.tensor([phi])
        return lID(tau*k_C(r,z(r,phi)),r)*lIR(tau*k_C(r,z(r,phi)),r)

    g = lambda r : 2*torch.pi*lambda_A*r*torch.exp(-lambda_A*torch.pi*r**2)
    integ_func_P_cov = lambda r,phi,active_D,active_R,active_C : g(r)*(active_D*aD(r)*pD(r) + active_R*aR(r)*pR(r,phi) + active_C*aC(r)*pC(r,phi))

    def solve(active_D,active_R,active_C,N=101**2):
        integration_domain = [[0,R_t],[0,torch.pi]]
        integrand = lambda d : integ_func_P_cov(d[:,0], d[:,1],active_D,active_R,active_C)
        return (1/torch.pi)*simp.integrate(integrand,dim=2,N=N,integration_domain=integration_domain)

    P_covD = solve(1,0,0)
    P_covR = solve(0,1,0)
    P_covC = solve(0,0,1)
    P_covOverall = solve(1,1,1)
    return (P_covD, P_covR, P_covC, P_covOverall)

# FIGURA 2(b) #######################################################################
b_lambdas = torch.arange(1,20,0.5)
results = torch.tensor([calc(torch.tensor([l])) for l in b_lambdas])
results_D = results[:,0]
results_R = results[:,1]
results_C = results[:,2]
results_overall = results[:,3]
plt.plot(b_lambdas.cpu(), results_D.cpu(), color="blue")
plt.plot(b_lambdas.cpu(), results_R.cpu(), color="red")
plt.plot(b_lambdas.cpu(), results_C.cpu(), color="orange")
plt.plot(b_lambdas.cpu(), results_overall.cpu(), color="purple")
plt.legend(["Direct","RIS","Composite","Overall"])
plt.xlabel("Densidade de bloqueios λB (bloqueios/m2)")
plt.ylabel("Probabilidades de cobertura")
plt.margins(0)
plt.grid()
plt.show()

# FIGURA 3(a) #######################################################################
# v0s = torch.arange(0.25,6.5,0.5)
# results8 = torch.tensor([calc(v_0=torch.tensor([l]),h_R=0.8*h_A) for l in v0s])
# results7 = torch.tensor([calc(v_0=torch.tensor([l]),h_R=0.7*h_A) for l in v0s])
# results6 = torch.tensor([calc(v_0=torch.tensor([l]),h_R=0.6*h_A) for l in v0s])
# results_overall8 = results8[:,3]
# results_overall7 = results7[:,3]
# results_overall6 = results6[:,3]
# # plt.subplot(2,2,2)
# plt.plot(v0s.cpu(), results_overall8.cpu(), color="blue")
# plt.plot(v0s.cpu(), results_overall7.cpu(), color="red")
# plt.plot(v0s.cpu(), results_overall6.cpu(), color="orange")
# plt.legend(["hR = 0.8 hA","hR = 0.7 hA","hR = 0.6 hA"])
# plt.xlabel("Distância v0 UE-RIS (m)")
# plt.ylabel("Probabilidades de cobertura")
# plt.margins(0)
# plt.grid()
# plt.show()

# FIGURA 3(b) #######################################################################
# lAs = torch.arange(0.5,11,1)
# results2 = torch.tensor([calc(lambda_A=torch.tensor([l]),tau=10**(2/10)) for l in lAs])
# results4 = torch.tensor([calc(lambda_A=torch.tensor([l]),tau=10**(4/10)) for l in lAs])
# results6 = torch.tensor([calc(lambda_A=torch.tensor([l]),tau=10**(6/10)) for l in lAs])
# results_overall2 = results2[:,3]
# results_overall4 = results4[:,3]
# results_overall6 = results6[:,3]
# # plt.subplot(2,2,3)
# plt.plot(lAs.cpu(), results_overall2.cpu(), color="blue")
# plt.plot(lAs.cpu(), results_overall4.cpu(), color="red")
# plt.plot(lAs.cpu(), results_overall6.cpu(), color="orange")
# plt.legend(["τ = 2 dB","τ = 4 dB","τ = 6 dB"])
# plt.xlabel("Densidade de APs λA (APs/m2)")
# plt.ylabel("Probabilidades de cobertura")
# plt.grid()
# plt.show()

# FIGURA 3(c) #######################################################################
# nRIS = torch.logspace(2,15,40)
# results1 = torch.tensor([calc(n=n_elm,h_B=1.1,activeRIS=True) for n_elm in nRIS])
# results2 = torch.tensor([calc(n=n_elm,h_B=1.6,activeRIS=True) for n_elm in nRIS])
# results3 = torch.tensor([calc(n=n_elm,h_B=1.1) for n_elm in nRIS])
# results4 = torch.tensor([calc(n=n_elm,h_B=1.6) for n_elm in nRIS])
# results5 = torch.tensor([calc(n=n_elm,h_B=1.1) for n_elm in nRIS])
# results6 = torch.tensor([calc(n=n_elm,h_B=1.6) for n_elm in nRIS])
# results_overall1 = results1[:,3]
# results_overall2 = results2[:,3]
# results_overall3 = results3[:,3]
# results_overall4 = results4[:,3]
# results_overall5 = results5[:,0]
# results_overall6 = results6[:,0]
# # plt.subplot(2,2,3)
# plt.plot(nRIS.cpu(), results_overall1.cpu(), color="blue", linestyle='--', marker='o')
# plt.plot(nRIS.cpu(), results_overall2.cpu(), color="blue")
# plt.plot(nRIS.cpu(), results_overall3.cpu(), color="red", linestyle='--', marker='o')
# plt.plot(nRIS.cpu(), results_overall4.cpu(), color="red")
# plt.plot(nRIS.cpu(), results_overall5.cpu(), color="orange", linestyle='--', marker='o')
# plt.plot(nRIS.cpu(), results_overall6.cpu(), color="orange")
# plt.legend(["Active hB = 1.1 m","Active hB = 1.6 m","Passive hB = 1.1 m","Passive hB = 1.6 m","No RIS hB = 1.1 m","No RIS hB = 1.6 m"])
# plt.xlabel("Número de elementos da RIS")
# plt.ylabel("Probabilidades de cobertura")
# plt.xscale('log')
# plt.grid()
# plt.show()