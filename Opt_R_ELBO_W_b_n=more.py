import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import pdb


##########################################################################
def analytical_RELBO(K,xi, cov_xz, cov_zx, mu_z, cov_z, W, b, U, c, alpha):

    #mu_zx = np.dot(W, xi) + b

    mu_zx = np.dot(W, xi).reshape((K,)) + b.reshape((K,))

    a = xi - c
    B = (1 - alpha) * np.linalg.inv(cov_xz)
    D = (1 - alpha) * np.linalg.inv(cov_z)
    d = mu_z
    f = mu_zx
    E = alpha * np.linalg.inv(cov_zx)

    A = np.dot(np.dot(U.T, B), U) + D + E
    v = np.dot(np.dot(U.T, B), a) + np.dot(D, d) + np.dot(E, f)
    w = np.dot(np.dot(a.T, B), a) + np.dot(np.dot(d.T, D), d) + np.dot(np.dot(f.T, E), f)

    t = np.linalg.solve(A, v)
    elem1 = w - np.dot(v.T, t)

    log_det_A = np.log(np.linalg.det(A))
    log_det_cov_z = np.log(np.linalg.det(cov_z))
    log_det_cov_xz = np.log(np.linalg.det(cov_xz))
    log_det_cov_zx = np.log(np.linalg.det(cov_zx))

    R = -N * np.log(2 * np.pi) - log_det_A / (1 - alpha) - log_det_cov_z - log_det_cov_xz - alpha * log_det_cov_zx / (
        1 - alpha) - elem1 / (1 - alpha)
    R = 0.5 * R
    #pdb.set_trace()
    return R

def Max_R_analytical_W_b(var, sign, K,xi, cov_xz, cov_zx, mu_z, cov_z, U, c, alpha):
    W = var[0:K**2]
    W=W.reshape((K,K))
    b = var[K**2:K+K**2]

    #print("\n pdb in Max_R_analytical_W_b \n")
    #pdb.set_trace()

    mu_zx = np.dot(W, xi) + b

    a = xi - c
    B = (1 - alpha) * np.linalg.inv(cov_xz)
    D = (1 - alpha) * np.linalg.inv(cov_z)
    d = mu_z
    f = mu_zx
    E = alpha * np.linalg.inv(cov_zx)

    A = np.dot(np.dot(U.T, B), U) + D + E
    v = np.dot(np.dot(U.T, B), a) + np.dot(D, d) + np.dot(E, f)
    w = np.dot(np.dot(a.T, B), a) + np.dot(np.dot(d.T, D), d) + np.dot(np.dot(f.T, E), f)

    t = np.linalg.solve(A, v)
    elem1 = w - np.dot(v.T, t)

    log_det_A = np.log(np.linalg.det(A))
    log_det_cov_z = np.log(np.linalg.det(cov_z))
    log_det_cov_xz = np.log(np.linalg.det(cov_xz))
    log_det_cov_zx = np.log(np.linalg.det(cov_zx))

    R = -N * np.log(2 * np.pi) - log_det_A / (1 - alpha) - log_det_cov_z - log_det_cov_xz - alpha * log_det_cov_zx / (
        1 - alpha) - elem1 / (1 - alpha)
    R = 0.5 * R
    scale = 10 ** (0)
    R = np.array([scale * sign * R])

    #print("\n In Max_R_analytical_W_b R=",R, "\n")

    return R

def Max_mu_R_estimated(var, sign, K,xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha,cons_std_zx):

    #W = var[0:4]
    #W=W.reshape((2,2))
    #b = var[4:6]
    #W = var[0:K ** 2]
    #W = W.reshape((K, K))

    W=var[0:K ** 2].reshape((K, K))
    b = var[K ** 2:K + K ** 2]

    #pdb.set_trace()

    mu_zx = np.dot(W,xi) + b
    mvn_qzx = multivariate_normal(mean=mu_zx, cov=cov_zx)

    Re = np.zeros(shape=(Ntimes, 1))

    for j in range(0, Ntimes, 1):
        zs = mvn_qzx.rvs(Nz)

        mvn_pz = multivariate_normal(mean=mu_z, cov=cov_z)

        qzx = mvn_qzx.pdf(zs)
        pz = mvn_pz.pdf(zs)

        ratio = np.zeros(shape=(Nz,))
        pxz = np.zeros(shape=(Nz,))

        for i in range(0, Nz, 1):
            mu_xz = np.dot(U,zs[i]) + c
            mvn_pxz = multivariate_normal(mean=mu_xz, cov=cov_xz)
            pxz[i] = mvn_pxz.pdf(xi)
            if qzx[i] > 0:
                ratio[i] = (pxz[i] * pz[i]) / qzx[i]
        Re[j] = np.log(np.mean((ratio) ** (1 - alpha))) / (1 - alpha)

    mu_Re = np.array([sign * np.mean(Re)])

    #verify the constraints
    #mu_x = np.dot(U, mu_z) + c
    #M1=np.dot(W,mu_x)+b-4*cons_std_zx
    #print("\n np.sqrt(np.dot(M1.T,M1))=",np.sqrt(np.dot(M1.T,M1)))
    #print("\n np.dot(M1.T,M1)=", np.dot(M1.T, M1))

    M1 = np.dot(W, xi) + b - 4 * cons_std_zx-mu_z
    norm2_elem=np.dot(M1.T,M1)

    #print("\n np.sqrt(norm2_elem)=",np.sqrt(norm2_elem))

    #norm2_mu_z=np.dot(mu_z.T,mu_z)
    #S_M1=np.sum(M1)
    #S_z = np.sum(mu_z)
    #print("\n S_M1-S_z=",S_M1-S_z)

    #print("\n norm2_elem-norm2_mu_z=",norm2_elem-norm2_mu_z)
    #M2 = np.dot(U,M1)
    #print("\n np.sqrt(np.dot(M2.T,M2))=", np.sqrt(np.dot(M2.T, M2)))

    # pdb.set_trace()
    #print("In Max_mu_R_estimated np.max(ratio)=", np.max(ratio))
    #print("In Max_mu_R_estimated mu_Re=", mu_Re)
    return mu_Re

def mu_R_estimated(W,b, xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha):
    mu_zx = np.dot(W,xi) + b
    mvn_qzx = multivariate_normal(mean=mu_zx, cov=cov_zx)

    Re = np.zeros(shape=(Ntimes, 1))

    for j in range(0, Ntimes, 1):
        zs = mvn_qzx.rvs(Nz)

        mvn_pz = multivariate_normal(mean=mu_z, cov=cov_z)

        qzx = mvn_qzx.pdf(zs)
        pz = mvn_pz.pdf(zs)

        ratio = np.zeros(shape=(Nz,))
        pxz = np.zeros(shape=(Nz,))

        for i in range(0, Nz, 1):
            mu_xz = np.dot(U,zs[i]) + c
            mvn_pxz = multivariate_normal(mean=mu_xz, cov=cov_xz)
            pxz[i] = mvn_pxz.pdf(xi)
            if qzx[i] > 0:
                ratio[i] = (pxz[i] * pz[i]) / qzx[i]
        Re[j] = np.log(np.mean((ratio) ** (1 - alpha))) / (1 - alpha)

    mu_Re = np.mean(Re)
    #pdb.set_trace()

    return mu_Re

###########################################################################################################
# Optimize R_ELBO_analytical, w.r.t. W,b, with constraints and with U and c fixed
N = 1#in 10 dim it needs over 800 func evaluations to minimize,Ntimes=20,Nz=50
K = 1
I = np.eye(K)
Ix = np.eye(N)

mu_z = np.zeros(shape=(K,))  # np.array([0,0])
sig_z = 1.0#0.1
cov_z = sig_z * I

U = 1.0*I
c = 0.1*np.ones(shape=(K,))
#c=c.reshape((K,))

z = np.zeros(shape=(K,))
# mu_xz=np.dot(U,z.T)+c -> U, c are the parameters of the output model, p(x|z), which are fixed
sig_xz = 1.0
cov_xz = sig_xz * Ix

sig_zx = 1.0#0.8
cov_zx = sig_zx * I

cons_std_zx=np.sqrt(sig_zx)*np.ones(shape=(K,))

#The mean and the covariance matrix are computed using the formulas for conditional Gaussian distributions, in the Bayesian framework
#Formulas can be found in this book:
#"Pattern recognition and machine learning", Bishop C., 2006, see page 93
mu_x = np.dot(U, mu_z) + c
cov_x = cov_xz + np.dot(np.dot(U, cov_z), U.T)

# mu_x=np.array([5])
# cov_x=np.array([[2.85]])
print("\n mu_x=", mu_x, "cov_x=", cov_x)
# U_best=np.sqrt((cov_x-cov_xz)/cov_z)
# c_best=mu_x-U_best*mu_z

# U=U_best
# c=c_best
#pdb.set_trace()
mvn_px = multivariate_normal(mean=mu_x, cov=cov_x)

# xi=np.array([mu_x-0*np.sqrt(cov_x)])#np.array([4.5089317])#np.array([-1.39172401])

#sample data from p(x); we just want to test the input data equal to the mean of p(x)
xi=np.zeros(shape=(N,))
for i in range(0,N,1):
    xi[i] = mu_x[i] + 0 * float(np.sqrt(cov_x[i,i]))

log_px = mvn_px.logpdf(xi)

print("\n U=", U, "c=", c, "log_px=", log_px)
#pdb.set_trace()

#W,b are the learned parameters of the posterior distribution, p(z|x)
#we use the scypi minimization package, to find the best W,b that minimizes the Renyi ELBO
W = 30*I#30*I
b = 5*np.ones(shape=(K,))#5*np.ones(shape=(K,))

alpha=0.5
R_analytical = analytical_RELBO(K,xi, cov_xz, cov_zx, mu_z, cov_z, W, b, U, c, alpha)

Ntimes=20
Nz=100
mu_Re = mu_R_estimated(W,b, xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha)


print("\n R_analytical=",R_analytical)
print("\n mu_Re=",mu_Re)

#minimize Renyi_ELBO_analytical, to get the best W,b
var0=[W,b]
var_opt_a = minimize(Max_R_analytical_W_b, var0, args=(-1.0,K,xi, cov_xz, cov_zx, mu_z, cov_z, U, c, alpha),
                      method='COBYLA',options={'maxiter':600})
print("\n var_opt_a=", var_opt_a)

Wa=var_opt_a.x[0]
ba=var_opt_a.x[1]


#The Renyi ELBO computed analytically for a simple toy example, similar to the pdfs found in a traditional VAE,
#but lower-dimensional
#The Renyi ELBO is introduced in Liu et al. "Rényi Divergence Variational Inference", 2016 https://arxiv.org/abs/1602.02311
R_analytical = analytical_RELBO(K,xi, cov_xz, cov_zx, mu_z, cov_z, Wa, ba, U, c, alpha)

#The Renyi ELBO estimated with importance sampling
mu_Re = mu_R_estimated(Wa,ba, xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha)

print("\n R_analytical=",R_analytical)
print("\n mu_R_estimated=",mu_Re)


#pdb.set_trace()

#Optimize mu_R_estimated
#W = var[0:K ** 2]
#W = W.reshape((K, K))
#b = var[K ** 2:K + K ** 2]

#W=var[0:K ** 2].reshape((K, K))
#b = var[K ** 2:K + K ** 2]
W0 = 0.5*I
b0 = 1*np.ones(shape=(K,))
#b0=1*cons_std_zx-np.dot(W0,xi)

#Check if the initial condition satisfies the constraint
M1 = np.dot(W0, xi) + b0 - 4 * cons_std_zx-mu_z
norm2_elem=np.dot(M1.T,M1)

#print("\n np.sqrt(norm2_elem)=",np.sqrt(norm2_elem))

#pdb.set_trace()

#Here, minimize the estimated Renyi ELBO, with respect to W and b, with different constraints on the W and b
var0 =np.vstack((W0,b0))
var0=var0.reshape((K+K**2,-1))

eps2=7.2**2

tmp1=eps2-np.dot((np.dot(var0[0:K**2].reshape((K, K)),mu_x)+(var0[K ** 2:K + K ** 2]-4*cons_std_zx)).T,
                 np.dot(var0[0:K**2].reshape((K, K)),mu_x)+(var0[K ** 2:K + K ** 2]-4*cons_std_zx))
#print("\n tmp1=",tmp1)

jac1=-2*(np.dot(var0[0:K**2].reshape((K, K)),mu_x)+(var0[K ** 2:K + K ** 2]-4*cons_std_zx))
jac2=-2*np.dot((np.dot(var0[0:K ** 2].reshape((K, K)),mu_x)+(var0[K ** 2:K + K ** 2]-4*cons_std_zx)),mu_x.T)

#print("\n jac1=",jac1)
#print("\n jac2=",jac2)

#pdb.set_trace()

cons = (
    {'type': 'ineq',
     'fun': lambda var: np.array([eps2-np.dot((np.dot(var[0:K**2].reshape((K, K)),xi)+(var[K ** 2:K + K ** 2]-4*cons_std_zx)).T,np.dot(var[0:K**2].reshape((K, K)),xi)+(var[K ** 2:K + K ** 2]-4*cons_std_zx))]),
     'jac': lambda var: np.array([-2*(np.dot(var[0:K**2].reshape((K, K)),xi)+(var[K ** 2:K + K ** 2]-4*cons_std_zx)),-2*np.dot((np.dot(var[0:K ** 2].reshape((K, K)),xi)+(var[K ** 2:K + K ** 2]-4*cons_std_zx)),xi.T)])})

#print("\n var0=",var0)
#pdb.set_trace()

#minimize without constraints
var_opt_e_0 = minimize(Max_mu_R_estimated, var0, args=(-1.0, K,xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha,cons_std_zx),
                      method='COBYLA',options={'maxiter':600})
print("\n var_opt_e_0=", var_opt_e_0)
#pdb.set_trace()

W_e_0 = var_opt_e_0.x[0:K**2]
W_e_0 = W.reshape((K, K))
b_e_0 = var_opt_e_0.x[K**2:K+K**2]
b_e_0=b_e_0.reshape((K,))

mu_Re_0=mu_R_estimated(W_e_0, b_e_0, xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha)
Ra_0 = analytical_RELBO(K,xi, cov_xz, cov_zx, mu_z, cov_z, W_e_0, b_e_0, U, c, alpha)

#minimize with constraint
var_opt_e_1 = minimize(Max_mu_R_estimated, var0, args=(-1.0, K,xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha,cons_std_zx),
                      constraints=cons,method='COBYLA',options={'maxiter':200})
print("\n var_opt_e_1=", var_opt_e_1)
#pdb.set_trace()

W_e_1 = var_opt_e_1.x[0:K**2]
W_e_1 = W.reshape((K, K))
b_e_1 = var_opt_e_1.x[K**2:K+K**2]
b_e_1=b_e_1.reshape((K,))
#R_analytical = analytical_RELBO(K,xi, cov_xz, cov_zx, mu_z, cov_z, W, b, U, c, alpha)

constr1_e=eps2-np.dot((np.dot(W,xi)+(b-4*cons_std_zx)).T,np.dot(W,xi)+(b-4*cons_std_zx))
print("\n constr1_e=",constr1_e)

mu_Re_1=mu_R_estimated(W_e_1, b_e_1, xi, mu_z, cov_zx, cov_xz, cov_z, U, c, Nz, Ntimes, alpha)
Ra_1 = analytical_RELBO(K,xi, cov_xz, cov_zx, mu_z, cov_z, W_e_1, b_e_1, U, c, alpha)

log_px = mvn_px.logpdf(xi)

#pdb.set_trace()

#print("\n R_analytical=",R_analytical)
#print("\n mu_R_estimated=",-var_opt_e.fun)
print("\n Function value after optimization without constraints: ", -var_opt_e_0.fun)
print("\n Function value after optimization with constraints: ", -var_opt_e_1.fun)

#print("\n mu_Re_0=",mu_Re_0)
#print("\n mu_Re_1=",mu_Re_1)

print("\n np.abs(Ra_0-(-var_opt_e_0.fun))=",np.abs(Ra_0-(-var_opt_e_0.fun)))
#print("\n np.abs(Ra_1-(-var_opt_e_1.fun))=",np.abs(Ra_1-(-var_opt_e_1.fun)))

#print("\n np.abs(Ra_0-mu_Re_0)=",np.abs(Ra_0-mu_Re_0))
#print("\n np.abs(Ra_1-mu_Re_1)=",np.abs(Ra_1-mu_Re_1))

#compare the minimized Renyi ELBO with the true log-likelihood
print("\n log_px=",log_px,"\n")
#print("\n np.abs(log_px-mu_Re_0)=",np.abs(log_px-mu_Re_0),"\n")
#print("\n np.abs(log_px-mu_Re_1)=",np.abs(log_px-mu_Re_1),"\n")
#######################################


