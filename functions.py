#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:48:09 2019

@author: marco
"""

import numpy as np
from scipy.special import loggamma
from sklearn.utils import shuffle
import torch, math 
import torch.cuda
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = "cpu"

######################################################################################################
################################# Auxiliary functions : begin ########################################
######################################################################################################

def gauss_kernel(x, sigma):
    sigma_sq = sigma**2
    n = len(x)
    row_norm_sq = .5*np.square(x).sum(axis = 1)
    M = row_norm_sq.repeat(n).reshape(-1,n)
    return np.exp( ( np.matmul(x, x.transpose()) - M - M.transpose() )/sigma_sq ) 
     
def poly_kernel(x,n):
    return np.power((1 + np.matmul(x, x.transpose())),n)

## Function to (directly!) compute the blocks inside G_q^{-1}
def PrecisionMatrix(eta_q, Card_q, K):
    D = len(K)
    A_q = eta_q * K
    # non diagonal block(s)
    ndb = -np.matmul(np.linalg.inv(np.eye(D) + Card_q*A_q), A_q)   ## We can do better, inversion can be computed by hand
    # diagonal block(s)
    db = np.eye(D) + ndb
    return [db, ndb]

## Function to compute the quadratic form associcated with cluster q 
## Provided that PrecisionMatrix is a O(D^2), the following function is O(ND^2) 
def Quadraticq(ClusteredObs_q, SumObsIn_q, eta_q, K):
    Card_q = ClusteredObs_q.shape[0]
    db, ndb = PrecisionMatrix(eta_q, Card_q, K)
    out = np.sum(np.power(ClusteredObs_q, 2)) + np.matmul(np.matmul(SumObsIn_q, ndb),SumObsIn_q.transpose())  
    return out
    
# Let's say that G_q = eta_q*KK_q + I. The following function computes KK_q
def get_KK(cZ, K, cluster):
    store = np.where(cZ == cluster)
    Cq = len(store[0])
    return np.tile(K, (Cq, Cq))

# Function to find the firt occurrence of a row in a marix or npdarray
def FindRowInMat(row, M):
    N,D = M.shape
    for idx in range(N):
        if np.sum(row == M[idx,:])== D:
            return idx
    
##########################
## Homoschedastic model ##
##########################
###############################
## (minus the) loss function ##
###############################

def ExactICL(y, cZ, theta, K, eigvals_):
    N = torch.tensor(y.shape[0], dtype = torch.int32, device = device)
    D = torch.tensor(y.shape[1], dtype = torch.int32, device = device)
    a = torch.exp(theta[0])[0]
    b = torch.exp(theta[1])[0]
    eta = torch.exp(theta[2])
    Q = len(eta)
    alpha_val = torch.exp(theta[3])
    alpha = torch.repeat_interleave(alpha_val, repeats = Q)
    
    eigvals = torch.tensor(eigvals_, dtype = torch.float32, device = device)  
    SumQuadTerms = torch.tensor(0.0, dtype = torch.float32, device = device)
    LogSumDet = torch.tensor(0.0, dtype = torch.float32, device = device)
    SumLgammaCqaq = torch.tensor(0.0, dtype = torch.float32, device = device)
    for q in range(Q):
        KK_ = get_KK(cZ, K, (q+1))
        KK = torch.tensor(KK_, dtype = torch.float32, device = device)
        Cq = len(KK)/D
        SumLgammaCqaq += torch.lgamma(Cq + alpha[q])
        Gq = eta[q]*KK + torch.eye(len(KK), device = device)
        detGq = torch.prod(torch.tensor(1, device = device) + Cq*eta[q]*eigvals)
        pos = np.where(cZ == (q+1))
        val_ = y[pos[0], :].reshape(-1,1)
        val = torch.tensor(val_, dtype = torch.float32, device = device)   
        quadTermq = torch.matmul(val.transpose(1,0), torch.matmul(torch.inverse(Gq), val))[0,0]  # [0,0] because it is seen as a matrix
        LogSumDet += torch.tensor(-0.5, device = device)*torch.log(detGq)
        SumQuadTerms += torch.tensor(0.5, device = device)*quadTermq
    ##########################
    # first term: log p(Y|Z,Q)
    ##########################
    logT1 = torch.tensor(-0.5, device = device)*N*D*torch.log(torch.tensor(2, device = device)*torch.tensor(math.pi, device = device)) + LogSumDet
    logT1 += a*torch.log(b) - torch.lgamma(a) + torch.lgamma(torch.tensor(0.5, device = device)*D*N + a)
    logT1 += -(torch.tensor(0.5, device = device)*N*D)*torch.log(b + SumQuadTerms)
    #########################
    # second term: log p(Z|Q)
    #########################
    logT2 = torch.lgamma(torch.sum(alpha)) - torch.sum(torch.lgamma(alpha)) + SumLgammaCqaq
    logT2 += -torch.lgamma(N + torch.sum(alpha))
    return logT1 + logT2


def FitOneEpoch(y, cZ, theta, K, eigvals, opt):
     opt.zero_grad()
     eicl = -ExactICL(y, cZ, theta, K, eigvals)
     eicl.backward()
     opt.step()
     return eicl, opt
 
#####################
## Autograd M-step ##
#####################
def MStep(y, cZ, theta, K, eigvals, epochs, opt, verbose):
    old_loss = torch.tensor(-1, dtype = torch.float32, device = device) 
    loss, opt = FitOneEpoch(y, cZ, theta, K, eigvals, opt)
    # print(' first loss in MStep: {}'.format(loss))
    epoch = 0
    store_loss = np.zeros(epochs)
    while epoch<epochs :    #torch.abs(old_loss - loss)>0.00000000000000000000001 and epoch<epochs:
        old_loss = loss
        loss, opt = FitOneEpoch(y, cZ, theta, K, eigvals, opt)
        store_loss[epoch] = loss
        if (epoch%10 == 0 and verbose != 0):
            print(' Negative ICL value: {}'.format(loss))
            print(' Epoch number : {}'.format(epoch))
        epoch +=1   
    return store_loss[np.where(store_loss !=0)]

################################
## Greedy Classification step ##
################################
def ClassifStep(y, 
                cZ, #
                Card, #
                K,
                eigvals,
                ClusteredObs, #
                SumObsInClus, #
                Qfs,#
                LogDetGqs, #
                eta,
                a,
                b,
                alpha):
    Q = len(Card)
    N = np.shape(y)[0]
    D = np.shape(y)[1]
    nb_swaps = 0
    if Q>1:
        seq = np.arange(len(y))
        seq = shuffle(seq)
        for idx in range(len(y)):
            obs = seq[idx]
            # the current cluster is denoted by q, the destination cluster by l
            q = cZ[obs] - 1
            if (Card[q]>1):
                delta_ll = np.zeros(Q)
                Card[q] -= 1
                pos = FindRowInMat(y[obs,:], ClusteredObs[q])
                indices = np.arange(ClusteredObs[q].shape[0])
                ClusteredObs[q] = ClusteredObs[q][indices != pos, :]
                SumObsInClus[q,:] = SumObsInClus[q,:] - y[obs,:]
                NewQFq = Quadraticq(ClusteredObs[q], SumObsInClus[q,:], eta[q], K)
                # test swap
                for l in range(Q):
                    Card[l] += 1
                    ClusteredObs[l] = np.vstack((ClusteredObs[l], y[obs,:]))
                    SumObsInClus[l,:] = SumObsInClus[l,:] + y[obs,:]
                    # log-determinant
                    LogDetGqs_mod = LogDetGqs.copy()
                    LogDetGqs_mod[q] = np.sum(np.log(1 + Card[q]*eta[q]*eigvals))
                    LogDetGqs_mod[l] = np.sum(np.log(1 + Card[l]*eta[l]*eigvals))
                    delta_ll[l] = -0.5*np.sum(LogDetGqs_mod)
                    # quadratic form
                    Qfs_mod = Qfs.copy()
                    Qfs_mod[q] = NewQFq #Quadraticq(ClusteredObs[q], SumObsInClus[q,:], eta[q], K)  # da spostare sopra il for...
                    Qfs_mod[l] = Quadraticq(ClusteredObs[l], SumObsInClus[l,:], eta[l], K)
                    delta_ll[l] -= (0.5*D*N + a)*np.log(b + 0.5*np.sum(Qfs_mod))
                    # delta_ll[l] += loggamma(a + 0.5*D*N)
                    # Dirichlet prior
                    delta_ll[l] += np.sum(loggamma(Card + alpha)) 
                    Card[l] -= 1
                    SumObsInClus[l,:] = SumObsInClus[l,:] - y[obs,:]
                    ClusteredObs[l] = ClusteredObs[l][:-1,:]
                # do swap and update
                lmax = delta_ll.argmax()
                # updating cards and cZ
                Card[lmax] += 1
                cZ[obs] = lmax+1
                # --
                # updating vector sums in cluster 
                SumObsInClus[lmax,:] = SumObsInClus[lmax,:] + y[obs,:]
                ClusteredObs[lmax] = np.vstack((ClusteredObs[lmax], y[obs,:]))
                if lmax!=q:                    # updating quadratic forms
                    Qfs[q] = Quadraticq(ClusteredObs[q], SumObsInClus[q,:], eta[q], K)
                    Qfs[lmax] = Quadraticq(ClusteredObs[lmax], SumObsInClus[lmax,:], eta[lmax], K)
                    LogDetGqs[q] = np.sum(np.log(1 + Card[q]*eta[q]*eigvals))
                    LogDetGqs[lmax] = np.sum(np.log(1 + Card[lmax]*eta[lmax]*eigvals))
                    nb_swaps += 1
    return nb_swaps

###########################################################################################################
###################################### Auxiliary functions : end ##########################################                
###########################################################################################################                

def Fit(y,
        times,
        Q,
        ia,
        ib,
        ieta,
        ialpha,
        kernel_type = "polynomial",         
        poly_ord = 3,
        epochs_in = 2, 
        epochs_out = 1000, 
        sigma = 0.2,
        lr = 2e-04,        
        Z = None, 
        verbose = 0,
        doMS = False,
        lbd_pois = None
        ):           
    N = y.shape[0]
    D = y.shape[1]
    if np.sum(Z==None)==1 :
        init_Z = np.random.choice(np.arange(1, Q+1), N, replace = True )
    else:
        init_Z = Z
    cZ = init_Z
    
    # hyper-parameters that are torch variables to optimize
    ## prior on sigma^2
    log_a = Variable(torch.tensor([np.log(ia)], dtype = torch.float32, device = device), requires_grad = True)
    log_b = Variable(torch.tensor([np.log(ib)], dtype = torch.float32, device = device), requires_grad = True)
    ## prior variances on beta
    log_eta = Variable(torch.tensor(np.repeat(np.log(ieta), Q), dtype = torch.float32, device = device), requires_grad = True)
    ## Dirichlet prior
    log_alpha = Variable(torch.tensor([np.log(ialpha)], dtype = torch.float32, device = device), requires_grad = True) # 4.60
    ## all parameters
    theta = [log_a, log_b, log_eta, log_alpha]
    
    ###################
    ## The optimizer ##
    ###################
    lr = lr
    opt = torch.optim.SGD(theta, lr = lr)
        
    ## polynomial kernel
    if kernel_type == "polynomial":
        K = poly_kernel(np.matrix(times).transpose(), poly_ord)
    else:
        K = gauss_kernel(np.matrix(times).transpose(), sigma)
    eigvals = np.linalg.eigvals(K).real
    
    # first M  Step
    # epochs = epochs_out
    # MStep(y, cZ, theta, K, eigvals, epochs, opt)
    
    ## Toward the C step
    ## The first step consists into stocking:    
    # 1. The clustered observations: Cq x D matrices
    ClusteredObs = []
    # 2. The number of obs in each cluseter
    Card = np.zeros(Q, dtype = int)
    # 3. The sum of the observations in each clustes
    SumObsInClus = np.zeros(shape=(Q,D))
    # 4. The quadratic forms for each cluster
    Qfs = np.zeros(Q)
    # 5. The log-determinants of precision matrices
    LogDetGqs = np.zeros(Q)
    # 6. The positions of each observation i inside ClusterObs[Z_i]
    # PosInClust = np.zeros(len(cZ))
    
    eta = torch.exp(log_eta).cpu().data.numpy()
    a = torch.exp(log_a).cpu().data.numpy()
    b = torch.exp(log_b).cpu().data.numpy()
    alpha_val = torch.exp(log_alpha).cpu().data.numpy()
    alpha = np.repeat(alpha_val, Q)
    #alpha_n = alpha.cpu().data.numpy() 
    
    for q in range(Q):
        pos_ = np.where(cZ == (q+1))
        pos = pos_[0]
        #PosInClust[pos] = np.arange(0, len(pos))
        Card[q] = len(pos)
        val = y[pos, :]
        ClusteredObs.append(val)
        SumObsInClus[q,:] = np.sum(val, 0)
        Qfs[q] = Quadraticq(ClusteredObs[q], SumObsInClus[q,:],  eta[q], K)
        LogDetGqs[q] = np.sum(np.log(1 + Card[q]*eta[q]*eigvals))
      
    ###############
    ## Main Loop ## 
    ###############
    loss = 0.0
    # MStep(y, cZ, theta, K, eigvals, epochs_out, opt, verbose)
    nb_swaps = -1
    print(" I am starting with CS-MS steps...")
    while nb_swaps != 0:
        if verbose != 0:
            print(" Classification Step...")
        nb_swaps = ClassifStep(y, cZ, Card, K, eigvals, ClusteredObs, SumObsInClus, Qfs, LogDetGqs, eta, a, b, alpha)
        if doMS == True:
            if verbose != 0:
                print(" Number of swaps: {}".format(nb_swaps))
                print(" Negative ICL value: {}".format(-ExactICL(y, cZ,theta, K, eigvals)))
                print(" Maximization Step...")
            loss = MStep(y, cZ, theta, K, eigvals, epochs_in, opt, verbose)
        eta = torch.exp(log_eta).cpu().data.numpy()
        a = torch.exp(log_a).cpu().data.numpy()
        b = torch.exp(log_b).cpu().data.numpy()
        alpha_val = torch.exp(log_alpha).cpu().data.numpy()
        alpha = np.repeat(alpha_val, Q)
    
    #################    
    ## Last M-Step ##    
    #################
    if doMS == True:
        print(" Last Maximization Step...")
        loss = MStep(y, cZ, theta, K, eigvals, epochs_out, opt, verbose)
    final_icl = ExactICL(y, cZ, theta, K, eigvals)
    if lbd_pois != None:
        final_icl += (Q*torch.log(torch.tensor(lbd_pois)) - torch.lgamma(torch.tensor((Q+1), dtype = torch.float32)) -  torch.tensor(lbd_pois)) #torch.log(torch.exp(torch.tensor(lbd_pois))-torch.tensor(1.0))) 
    print("Final Negative ICL: {}".format(- final_icl))
    eta = torch.exp(log_eta).cpu().data.numpy()
    a = torch.exp(log_a).cpu().data.numpy()
    b = torch.exp(log_b).cpu().data.numpy()
    alpha_val = torch.exp(log_alpha).cpu().data.numpy()
    alpha = np.repeat(alpha_val, Q)    
    return [cZ, eta, a, b, alpha, final_icl, loss]

#####################################################
## Function to create empirical confidence regions ##
## for the posterior trajectory                    ##
#####################################################        

def ConfBounds(mu, Sigma, N = 5000, alpha = 0.95):
    val = np.random.multivariate_normal(mu, Sigma, N).T
    ub = np.quantile(val,  alpha, axis = 1)
    lb = np.quantile(val, (1-alpha), axis = 1)
    return ub,lb


##########################
## Predictive Distributions ##
##############################
def Plot(y, times, poly_ord, cZ, eta, a, b, kernel_type = "polynomial", sigma=0.2, SplitFigures = True, n_traj = 5000, palette = None):
    if palette == None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    else:
        colors = palette 
    top = np.max(y) + 0.2
    bottom = np.min(y) - 0.2
    N,D = y.shape
    Q = len(eta)
    Qfs = np.zeros(Q)
    # Start by computing the max posterior estimate of sigma
    ClusteredObs = []
    if kernel_type == "polynomial":
        K = poly_kernel(np.matrix(times).transpose(), poly_ord)
    else:
        K = gauss_kernel(np.matrix(times).transpose(), 0.2)
    # eigvals = np.linalg.eigvals(K).real
    for q in range(Q):
        pos = np.where(cZ == (q+1))[0]
        val = y[pos, :]
        ClusteredObs.append(val)
        SumObsInClus = np.sum(val, 0)
        Qfs[q] = Quadraticq(ClusteredObs[q], SumObsInClus, eta[q], K)
        #LogDetGqs[q] = np.sum(np.log(1 + Card[q]*eta[q]*eigvals))
    b_bar = b + 0.5*np.sum(Qfs)
    a_bar = 0.5*D*N + a
    Es2 = b_bar/(a_bar-1)  # the mode of the estimated Gamma distribution
    
    # new grid to compute the predictive trajectories
    ti = times[0]
    tf = times[-1:]
    step = 0.05
    times_ = np.arange(ti - step, tf + step, step)
    if kernel_type == "polynomial":
        K_ = poly_kernel(np.vstack((times.reshape(-1,1), times_.reshape(-1,1))), poly_ord)
    else: 
        K_ = gauss_kernel(np.vstack((times.reshape(-1,1), times_.reshape(-1,1))), sigma)
    Kno = K
    Kso = K_[D:, :D]
    Kse = K_[D:, D:]
   # Kne = K_[:D, D:]
    
    mean_traj = np.zeros((Q, len(times_)))
    for q in range(Q):
        card_q = len(ClusteredObs[q])
        Y_q = ClusteredObs[q].reshape(-1,1)
        augmented_Kno = np.tile(Kno, (card_q, card_q))
        augmented_Kso = np.tile(Kso, (1, card_q))
        #augmented_Kne = np.tile(Kne, (card_q, 1))
        mean_traj[q,:] = np.matmul(np.matmul(eta[q]*augmented_Kso, np.linalg.inv(eta[q]*augmented_Kno + Es2*np.eye(len(augmented_Kno)))), Y_q).transpose()
        varcov = (eta[q]*Kse + Es2*np.eye(len(Kse))) - np.matmul(np.matmul(eta[q]*Kso, np.linalg.inv(eta[q]*Kno + Es2*np.eye(len(Kno)))), eta[q]*Kso.transpose()).transpose()
        ub, lb = ConfBounds(mean_traj[q,], varcov, n_traj)
        #stds = np.sqrt(np.diag(varcov))
        #ub = mean_traj[q,:] + 1.96*stds
        #lb = mean_traj[q,:] - 1.96*stds
        if SplitFigures == True:
            plt.figure(" Cluser {}".format(q+1))
            plt.ylim(bottom, top)
            for idx in range(np.min((30,len(ClusteredObs[q])))):
                plt.plot(times, np.array(ClusteredObs[q][idx,:]).reshape(len(times),), color = colors[q], linewidth = 0.5)
                plt.plot(times_, mean_traj[q,:], color = "black")
                plt.plot(times_, ub, '--', color = "black")
                plt.plot(times_, lb, '--', color = "black")
        else:
            plt.subplot(2,np.ceil(Q/2),(q+1))
            plt.xlabel("Time")
            plt.ylabel("Trajectory")
            plt.ylim(bottom, top)
            for idx in range(np.min((30,len(ClusteredObs[q])))):
                plt.plot(times, np.array(ClusteredObs[q][idx,:]).reshape(len(times),), color = colors[q], linewidth = 0.5)
                plt.plot(times_, mean_traj[q,:], color = "black")
                plt.plot(times_, ub, '--', color = "black")
                plt.plot(times_, lb, '--', color = "black")

        


#Plot(y, times, cZ, eta, a, b,  SplitFigures = False)

#####################
## Model Selection ##
#####################

#store_icl = []
#for Q in range(2, 6):
#    print(" Number of clusters: {}".format(Q))
#    cZ, eta, a, b, icl = Fit(y, times, Q, poly_ord = poly_ord)
#    store_icl.append(icl)



