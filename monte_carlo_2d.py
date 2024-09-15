import torch
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = sp.constants.c
n_A = 10                       
f   = 0.3e12
kA = 0.075
R_t = torch.sqrt(torch.tensor([140]))
h_A = 3                        
h_U = 1                        
h_R = 0.75*h_A
ris_loc = torch.tensor([1, 1, h_R])
v_0 = torch.sqrt(torch.tensor([2]))                  
r_B = 0.22                     
h_B = 1.63                     
lambda_B = 2              
lambda_A = 1
gU = 10**(30/10)
gA = gU
tau = 10**(2/10)
n   = 1e13            
n_realizations = 450
lX  = c/(2*f); lY = lX

def cart2pol(x,y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.arctan(y/x)
    return(rho, phi)

def pol2cart(rho,phi):
    x = rho * torch.cos(phi[0])
    y = rho * torch.sin(phi[0])
    return(x, y)

def poisson_pp(xx0,yy0,radius,lambda0):
    lambda0 = torch.tensor([lambda0])
    areaTotal = torch.pi*radius**2
    numbPoints = torch.poisson(lambda0*areaTotal)
    phi=2*torch.pi*torch.rand(1,int(numbPoints.item()))
    rho=radius*torch.sqrt(torch.rand(1,int(numbPoints.item())))
    xx, yy = pol2cart(rho,phi)
    xx=xx+xx0; yy=yy+yy0
    return (xx,yy)

def vec_proj(x,y):
    # vector projection of y in x
    return x*torch.dot(x,y)/torch.dot(x,x)

def verify_horizontal_blockage(vecA, vecB):
    # verify if object located at vecA is horizontally LoS blocked by blockage at vecB 
    vecA_norm = torch.sqrt(torch.dot(vecA,vecA))
    vecB_norm = torch.sqrt(torch.dot(vecB,vecB))
    orth_vect_blockage = vecB - vec_proj(vecA,vecB)
    if torch.sqrt(torch.dot(orth_vect_blockage,orth_vect_blockage)) < r_B and vecB_norm < vecA_norm:
        return True

def verify_vertical_blockage(vecA, hA, vecB, hB):
    # verify if object located at vecA is vertically LoS blocked by blockage at vecB
    if torch.dot(vecB,vecB).item() == 0:
        return True
    if torch.dot(vecA,vecA).item() == 0:
        return False
    if hB/torch.dot(vecB,vecB).item() >= hA/torch.dot(vecA,vecA).item():
        return True
    
def direct_pathloss(distance):
    return gU*gA*c**2/(4*np.pi*f)**2 * np.exp(-kA*distance)/(distance**2)

# def power_calc(xa,ya):
#     # have to implement the rest
#     ap_distance = np.sqrt(xa**2 + ya**2 + (h_A-h_U)**2)
#     return direct_pathloss(ap_distance)

def blockage_verification(xb,yb,vec_2D_RIS,nearest_AP_2D_vector,norm_RIS_vector,norm_nearest_AP_vector):
    blockage_RIS = False
    blockage_direct = False
    
    for i in range(len(xb[0])):
        if blockage_RIS and blockage_direct:
            break
        blockage_vector = torch.tensor([xb[0][i].item(),yb[0][i].item()])
        
        if not blockage_RIS and torch.dot(blockage_vector,blockage_vector).item() <= norm_RIS_vector:
            if verify_horizontal_blockage(vec_2D_RIS,blockage_vector):
                if verify_vertical_blockage(vec_2D_RIS,h_R,blockage_vector,h_B):
                    blockage_RIS = True
        if not blockage_direct and torch.dot(blockage_vector,blockage_vector).item() <= norm_nearest_AP_vector:
            if verify_horizontal_blockage(nearest_AP_2D_vector,blockage_vector):
                if verify_vertical_blockage(nearest_AP_2D_vector,h_A,blockage_vector,h_B):
                    blockage_direct = True
    return (blockage_direct,blockage_RIS)

# def direct_threshold_verification(xb,yb,xa,ya,x0,y0):
#     # True means that SIR is greater than threshold, False otherwise
#     interference_power = []
#     serving_power = power_calc(x0,y0)
#     for i in range(len(xa[0])):
#         ap_vector = torch.tensor([xa[0][i].item(),ya[0][i].item()])
#         # for each AP
#         for j in range(len(xb[0])):
#             # verify if there is any blockage
#             blockage_vector = torch.tensor([xb[0][j].item(),yb[0][j].item()])
#             if not verify_horizontal_blockage(ap_vector,blockage_vector):
#                 if not verify_vertical_blockage(ap_vector,h_A,blockage_vector,h_B):
#                     # AP not blocked
#                     received_power = power_calc(xa[0][i].item(),ya[0][i].item())
#                     interference_power.append(received_power)
#     total_interference_power = sum(interference_power)
#     calc_SIR = serving_power/total_interference_power
#     if calc_SIR >= tau:
#         return True
#     return False

def monte_carlo(lambda_B=lambda_B):
    direct_count = 0
    ris_count = 0
    composite_count = 0
    no_assoc_count = 0

    for _ in range(int(n_realizations)):
        xa, ya = poisson_pp(0,0,R_t,lambda_A)
        xb, yb = poisson_pp(0,0,R_t,lambda_B)

        distances_UA = torch.sqrt(xa**2 + ya**2)
        index_min = torch.argmin(distances_UA)
        x0 = xa[0][index_min.item()].item()
        y0 = ya[0][index_min.item()].item()
        nearest_AP_2D_vector = torch.tensor([x0,y0])

        vec_2D_RIS = torch.tensor([ris_loc[0].item(),ris_loc[1].item()])
        norm_RIS_vector = torch.dot(vec_2D_RIS,vec_2D_RIS).item()
        norm_nearest_AP_vector = torch.dot(nearest_AP_2D_vector,nearest_AP_2D_vector).item()

        blockage_direct,blockage_RIS = blockage_verification(xb,yb,vec_2D_RIS,nearest_AP_2D_vector,norm_RIS_vector,norm_nearest_AP_vector)

        if blockage_direct and not blockage_RIS:
            ris_count += 1
        if blockage_RIS and not blockage_direct:
            direct_count += 1
        if not blockage_direct and not blockage_RIS:
            composite_count += 1
        if blockage_direct and blockage_RIS:
            no_assoc_count += 1

    direct_coverage_probability = direct_count/n_realizations
    ris_coverage_probability = ris_count/n_realizations
    composite_coverage_probability = composite_count/n_realizations
    no_assoc_probability = no_assoc_count/n_realizations

    return (direct_coverage_probability,ris_coverage_probability,composite_coverage_probability,no_assoc_probability)

b_lambdas = torch.arange(0.5,10,0.5)

dir_cov_probs = []
ris_cov_probs = []
comp_cov_probs = []
no_assoc_probs = []

for lb in b_lambdas:
    dir_cov_prob,ris_cov_prob,comp_cov_prob,no_assoc_prob = monte_carlo(lambda_B=lb)
    dir_cov_probs.append(dir_cov_prob)
    ris_cov_probs.append(ris_cov_prob)
    comp_cov_probs.append(comp_cov_prob)
    no_assoc_probs.append(no_assoc_prob)

plt.plot(b_lambdas,dir_cov_probs,color="blue")
plt.plot(b_lambdas,ris_cov_probs,color="red")
plt.plot(b_lambdas,comp_cov_probs,color="orange")
plt.plot(b_lambdas,no_assoc_probs,'--',color="purple")
plt.legend(["Direct","RIS","Composite","No Association"])
plt.grid()
plt.xlabel("Densidade de bloqueios λB (bloqueios/m2)")
plt.ylabel("Probabilidades de associação")
plt.margins(0)
plt.show()