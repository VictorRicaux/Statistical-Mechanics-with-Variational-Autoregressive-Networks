import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from VAN_new import *
import pandas as pd
import os


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


def sample_magnetisations_without_annealing(betas=[0.05, 0.1, 0.145, 0.19, 0.235, 0.28, 0.325, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], taille=100, n_samples=10):
    '''
    Function that computes a probability distribution of the spins and then samples from it.
    It plots the results and saves both the plots and the data. 
    '''
        
        
    for beta in betas: 
        magnetisations_list=[]

        for i in range(n_samples):
            my_model = VAN(taille)
            losses = train(my_model, lambda x:  log_prob_target_energie(x, beta), batch_size=200, n_iter=1000, lr=0.01)
            mysample=my_model.sample(1000)
            magnetisations=[]
            for spin in mysample:
                magnetisations.append(torch.mean(spin))
            plt.hist(magnetisations, bins=20, edgecolor='black') 
            plt.xlabel('Magnetisation of the spin')
            plt.ylabel('Number of spins')
            plt.title('Magnetisation of the spins for beta =' + str(beta))
            plt.savefig('./figures/magnetisation_for beta= '+str(beta)+ 'test n° ' +str(i) + '.png')
            magnetisations_list.append(magnetisations)
            plt.show()
        pd.DataFrame(magnetisations_list).to_csv('./magnetisations/magnetisations for beta= '+str(beta)+'.csv')



def plot_evolution_of_magnetisation(betas=[0.05, 0.1, 0.145, 0.19, 0.235, 0.28, 0.325, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], taille=100): 
    '''
    Function that plots the previous computation to observe the evolution of the magnetisation with temperature
    '''
    moyennes_de_magnetisations=[]
    for titre in os.listdir('magnetisations'): 
        if 'annealing' not in titre:
            with open ('./magnetisations/' + titre) as f:
                data = pd.read_csv(f)
                data = data.drop(columns=['Unnamed: 0'])
                data = data.to_numpy()
                data_final=np.sum([data[i] for i in range(len(data))])
                data_converted_final=[[float(data[j][i].split('(')[1].split(')')[0]) for i in range(len(data[j]))] for j in range(len(data))]
                magnetisations_moyennes=[np.mean(data_converted_final[i]) for i in range(len(data_converted_final))]
                moyennes_de_magnetisations.append(magnetisations_moyennes)

    x =betas
    y = [abs(np.mean(moyennes_de_magnetisations[i])-0.5) for i in range(len(moyennes_de_magnetisations))]
    variances=[np.var(moyennes_de_magnetisations[i]) for i in range(len(moyennes_de_magnetisations))]
    beta_c= 0.5*np.log(1+np.sqrt(2))
    plt.errorbar(x, y, yerr=variances, fmt='o', color='red', ecolor='black')
    plt.plot(x, y, 'g:')
    cote=int(np.sqrt(taille))

    plt.title('Magnetization of the  '+ str(cote)+' x ' + str(cote) + ' grid as a function of beta, without annealing')
    plt.xlabel('Beta')
    plt.ylabel("Magnetization")
    # ajouter une barre à beta_c: 
    plt.axvline(x=beta_c, color='blue', linestyle='--')

    plt.show()


def sample_magnetisations_with_annealing(betas=[0.05, 0.1, 0.145, 0.19, 0.235, 0.28, 0.325, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], taille=100):
    '''
    Function that computes a probability distribution of the spins and then samples from it, with annealing. 
    It plots the results and saves both the plots and the data. 
    '''
    mymodel1 = VAN(taille)
    racine=int(np.sqrt(taille))
    for beta in betas: 
        print( taille, beta)
        magnetisations_list=[]
        losses = train(mymodel1, lambda x:  log_prob_target_energie(x, beta), batch_size=200, n_iter=1000, lr=0.001)
        mysample=mymodel1.sample(1000)
        magnetisations=[]
        for spin in mysample:
            magnetisations.append(torch.mean(spin))
        plt.hist(magnetisations, bins=20, edgecolor='black') 
        plt.xlabel('Magnetisation of the spin')
        plt.ylabel('Number of spins')
        plt.title('Magnetisation of the spins for beta =' + str(beta) + ' on a ' + str(racine) + 'x' + str(racine) + ' lattice')
        plt.savefig('./figures/magnetisation_test_annealing for beta= '+str(beta)+ ', racine=' +str(racine) + '.png')
        magnetisations_list.append(magnetisations)
        plt.show()
        pd.DataFrame(magnetisations_list).to_csv('./magnetisations/magnetisations_test_annealing for beta= '+str(beta)+ ' and racine = ' + str(racine) + '.csv')




def plot_evolution_of_magnetisation_with_annealing(betas=[0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1], taille=100):
    '''
    Function that plots the previous computation to observe the evolution of the magnetisation with temperature
    Without annealing
    '''
    moyennes_de_magnetisations=[]
    for titre in os.listdir('magnetisations'): 
        if 'annealing'  in titre and 'racine' not in titre:
            print(titre)
            with open ('./magnetisations/' + titre) as f:
                data = pd.read_csv(f)
                data = data.drop(columns=['Unnamed: 0'])
                data = data.to_numpy()
                data_final=np.sum([data[i] for i in range(len(data))])
                data_converted_final=[[float(data[j][i].split('(')[1].split(')')[0]) for i in range(len(data[j]))] for j in range(len(data))]
                magnetisations_moyennes=[np.mean(abs(data_converted_final[i]-0.5*np.ones(len(data_converted_final[i])))) for i in range(len(data_converted_final))]
                moyennes_de_magnetisations.append(magnetisations_moyennes)
                
    x=betas
    y = [(np.mean(moyennes_de_magnetisations[i])) for i in range(len(moyennes_de_magnetisations))]
    variances=[np.var(moyennes_de_magnetisations[i]) for i in range(len(moyennes_de_magnetisations))]
    plt.errorbar(x, y, yerr=variances, fmt='o', color='red', ecolor='black')
    plt.plot(x, y, 'g:')
    cote=int(np.sqrt(taille))

    plt.title('Magnetisation of the  '+ str(cote)+' x ' + str(cote) + ' grid as a function of beta, with annealing')
    plt.xlabel('Beta')
    plt.ylabel("Magnetisation of the grid")
