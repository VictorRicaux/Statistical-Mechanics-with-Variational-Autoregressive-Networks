import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from VAN_new import *
import pandas as pd

def get_column(matrix, i):
    return torch.tensor([matrix[j][i] for j in range(len(matrix))])

def energie1D(spin): 
    spin_copie=spin.clone()
    spin_copie[spin_copie==0]=-1
    spin_copie_1 = torch.roll(spin_copie, -1)
    spin_copie_2 = torch.roll(spin_copie, 1)    
    energie=- torch.sum(spin_copie_1*spin_copie+spin_copie_2*spin_copie)
    return energie

def energie2D(lattice):
    energie = 0 
    for i in range(len(lattice)):
        energie+=energie1D(lattice[0])
    for j in range(len(lattice[0])):
        column = get_column(lattice, j)
        energie+=energie1D(column)
    return energie


def log_prob_energie(beta, energie):
    return -beta*energie


def log_prob_target_energie(spins, beta):
    
    log_probs = torch.ones(spins.shape[0]) * np.log(0.001)
    for i in range(len(log_probs)):
        racine=spins[i].shape[0]
        racine=(int(np.sqrt(racine)))
        lattice = spins[i].reshape(racine, racine)
        log_probs[i] = log_prob_energie(beta, energie2D(lattice))
    return log_probs 



taille=100
betas = np.linspace(0.5, 6.5, 11)
for beta in betas: 
    magnetisations_list=[]
    for i in range(15): 
        print(beta, i)
        mymodel1 = VAN(taille)
        losses = train(mymodel1, lambda x:  log_prob_target_energie(x, beta), batch_size=200, n_iter=1000, lr=0.01)
        mysample=mymodel1.sample(1000)
        magnetisations=[]
        for spin in mysample:
            magnetisations.append(torch.mean(spin))
        plt.hist(magnetisations, bins=20, edgecolor='black') 
        plt.xlabel('Magnetisation du spin')
        plt.ylabel('Nombre de spins')
        plt.title('Magnetisation des spins pour beta =' + str(beta))
        plt.savefig('./figures/magnetisation for beta= '+str(beta)+' test n° ' + str(i) + '.png')
        magnetisations_list.append(magnetisations)
    pd.DataFrame(magnetisations_list).to_csv('.magnetisations/magnetisations for beta= '+str(beta)+'.csv')
        
