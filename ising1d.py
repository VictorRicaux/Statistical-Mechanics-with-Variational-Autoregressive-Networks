import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from VAN_new import *

taille=20

y1= torch.tensor([[1.0 for i in range(taille//2)] + [0.0 for i in range(taille//2)]])
y2= torch.tensor([[0.0 for i in range(taille//2)] + [1.0 for i in range(taille//2)]])
y3=torch.tensor([[0.0 for i in range(taille)]])
y4=torch.tensor([[1.0 for i in range(taille)]])
y5=torch.tensor([[1.0, 0.0] *( taille//2)])

def log_prob_target(spins):
    ''''
    Compute the log probability of the target distribution
    spin confirgurations mostly up more likely (* 0.8) than mostly down (* 0.2)
    '''
    spin_size = spins.shape[1]
    log_probs = torch.ones(spins.shape[0]) * np.log(0.2)
    log_probs[spins.mean(-1) > 0.5] = np.log(0.8)
    return log_probs - np.log(2 ** (spin_size -1)) 


def log_prob_target_2(spins):
    """
    log proba de la target distribution. High probability if the forst spin is 1 and low otherwise
    """
    spin_size = spins.shape[1]
    log_probs = torch.ones(spins.shape[0]) * np.log(0.2)
    
    log_probs[torch.tensor([spins[i][0] > 0.5 for i in range(len(spins))] )] = np.log(0.8)
    
    return log_probs - np.log(2 ** (spin_size -1)) 
    


def energie1D(spin): 
    spin_copie=spin.clone()
    spin_copie[spin_copie==0]=-1
    spin_copie_1 = torch.roll(spin_copie, -1)
    spin_copie_2 = torch.roll(spin_copie, 1)

    
    energie=- torch.sum(spin_copie_1*spin_copie+spin_copie_2*spin_copie)
    return energie
    

def log_prob_energie(beta, energie):
    return -beta*energie


def log_prob_target_energie(spins):
    
    log_probs = torch.ones(spins.shape[0]) * np.log(0.01)
    
    for i in range(len(log_probs)):
        log_probs[i] = log_prob_energie(1, energie1D(spins[i]))
    return log_probs 

    
# recuperer la colonne d'une matrice
def get_column(matrix, i):
    return torch.tensor([matrix[j][i] for j in range(len(matrix))])



if __name__=='__main__':
    mymodel1 = VAN(taille)
    losses = train(mymodel1, log_prob_target, batch_size=1000, n_iter=1000, lr=0.01)
    plt.plot(losses)
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('DKL')
    plt.title('Entraînement modèle 1')
    plt.show()
    print('Probability to observe the spins y1, y2, y3, y4, y5 with model 1:')
    print(mymodel1.prob_of_spins(y1))
    print(mymodel1.prob_of_spins(y2))
    print(mymodel1.prob_of_spins(y3))
    print(mymodel1.prob_of_spins(y4))
    print(mymodel1.prob_of_spins(y5))