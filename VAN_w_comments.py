import torch 
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.distributions import Bernoulli

'''
Une amélioratrion probablement nécessaire du code est de "vectoriser" les fonctions. 
Elles doivent manipuler des torch.tensor de dimension (batch_size, spin_size). 
'''

def prob(s_hat, s): # fonction intermdédiaire utilisée dans train
    '''
    p(s ; s_hat): loi de Bernoulli de paramètre s_hat
    '''
    return (s_hat**s)*(1-s_hat)**(1-s)

# class MaskedLinear: Couche de neurones qui 
# permet de mettre un masque pour que les sorties d'indice j ne dépendent 
# que des entrées d'indice i avec i<=j
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask = mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight * self.mask, self.bias)


# Classe VAN : Variable Auto-regressive Network
class VAN(nn.Module):
    def __init__(self, input_size, activation=torch.sigmoid):
        super(VAN, self).__init__() #initialisation obligatoire
        self.input_size = input_size
        self.activation = activation

        # Création de la matrice de masque : que des 0 sur et au dessus de la diagonale et que des 1 dessous
        M = torch.zeros((input_size, input_size), dtype=torch.float)
        for i in range(input_size):
            for j in range(i, input_size):
                M[i][j] = 0.0
        for i in range(1, input_size):
            for j in range(i):
                M[i][j] = 1.0

        self.fc1 = MaskedLinear(input_size, input_size, mask=M) 
        for param in self.parameters():
            param.requires_grad = True
            init.constant_(param, 0)  # Initialize all parameters to 0

    def forward(self, x):
        '''
        Compute the parameters of the Bernoulli distribution for each spin
        '''
        x = self.fc1(x)
        x = self.activation(x)
        # à cette ligne on a multiplié x par la matrice de masque (triangulaire inférieure), puis appliqué la fonction d'activation
        # donc la première coordonnée de x vaut activation(0) =0.5 (normal, s^_1 ne dépend de personne)
        return x
    
    
    def calculer_proba_vec(self, spin): # fonction qui à partir d'un état de spin et d'un modèle renvoie la proba de voir ce spin d'après le modèle (q_theta)
        '''
        Cette fonction aurait sa place dans cette classe. Par contre, utiliser .detach() 
        vous empechera de l'utiliser dans le calcul de l'erreur d'entrainement.
        '''
        probas = self.model(spin)
        for i in range(len(spin)):
            res*=prob(probas[i], spin[i].detach().numpy())
        return res
    


def Kullback_Leibler(q,p): # p et q sont des listes de probas d'observation pour le même s
    # on renormalise sur le support tiré
    '''
    Ici vous avez gardé q/torch.sum(q) en facteur du log. 
    C'est expression est correcte, exacte même, si vous sommez 
    sur tous les 2^{spin_size} états possibles une seule fois.

    Pour des systèmes où spin_size sera grand, il ne sera pas réaliste de faire une 
    somme de cette taille. Il faudra alors utiliser une approximation de Monte Carlo
    qui consiste à faire une moyenne empirique de log(q(s)/p(s) sur un échantillon de s
    tirés selon q(s).

    En pratique si vous ne sommez pas sur tout le support, vous ne pourrez pas normaliser de la sorte. 
    Toute-fois votre distribution q doit être normalisée par definition, vous n'avez donc pas besoin de la faire. 
    Il est aussi inutile de normaliser p, car la normalisation est une constante qui n'aura pas d'impact sur le gradient.
    '''
    return torch.sum((q/torch.sum(q))*torch.log((q/torch.sum(q))/(p/torch.sum(p))))  # c'est une espérance empirique c'est pour ça qu'on n'a pas qi en facteur (Marylou: vous l'écrivez mais vous ne le faites pas :))

    
def train(model, p_obj,  n_iter=100, lr=1e-2, train_size=100):
    # p_obj est la distribution à approximer. 
    # elle prend en input un vecteur de 0 et 1 de taille model.input_size et renvoie la proba de ce vecteur
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_iter):
        optimizer.zero_grad() # What is this step? IMPORTANT LINE, c'est la prof qui l'a dit

        # il faut tirer un train_set grâce au modèle pour s'entraîner dessus
        train_set=torch.zeros((train_size, model.input_size))
        for i in range(train_size):
            # pour tirer un x dans x_train, on tire une bernoulli de paramètre 0.5 : c'est s_1 
            # comment avoir s_2 ? ON fait passer le vecteur [s_1, 0, 0, 0] dans le réseau de neurones
            # en sortie du réseau de neurones, on a y =[ s1 randomn, p(s2|s1), p(s3|s1), p(s4|s1)...]
            # la deuxième coordonnée de y est p(s2|s1), donc on tire une bernoulli de paramètre p(s2|s1)
            # puis on recommence !
            # on fait ça train_size fois
            '''
            Correct! sauf pour le s_1. Le modèle prédit un paramètre de Bernoulli, qui n'est pas nécessairement 0.5.
            '''
            train_set[i][0]=torch.bernoulli(torch.tensor(0.5).detach()) # on met une bernoulli de 0.5 sur la première valeur

            for j in range(1, model.input_size): # on parcourt tout le spin 
                y_pred=model(train_set[i])
                p_j = y_pred[j] # c'est p(s_j|s_{i<j})
                if p_j > 1 or p_j < 0: # c'est pas censé arriver 
                    p_j = torch.sigmoid(p_j)
                    print('Proba impossible ')
                if p_j!=p_j: # test pour ne pas avoir de p_j qui valent nan 
                    p_j=torch.tensor(0.5)
                    print('proba = Nan ! ')
                train_set[i][j] = torch.bernoulli(p_j).item() # on tire une bernoulli de paramètre p(s_j|s_{i<j}) pour la j-ème variable
        
        #retirer les doublons de train_set pour pouvoir calculer DKL sur le support des spins tirés 
        '''
        Ici, vous vous rattrapez en retirant les doublons pour espérer faire in fine la somme exacte. 
        Ce n'est pas sûr que vous ayez vu toutes les configurations permises, si vous choisissez 
        cette stratégie autant les énumérer.  Mais ça ne "scale" pas.
        '''
        train_set = torch.unique(train_set, dim=0)
        y_train=torch.tensor([p_obj(s) for s in train_set], requires_grad=True) # les p(s) sur les s_i tirés
        
        # on a notre train set pour cette époque

        listes_de_probas_conditionelles=model(train_set) # on récupère les probas conditionelles, il faut les multiplier pour avoir les probas tout court
        # print(listes_de_probas_conditionelles)
        q_theta_predit=torch.zeros(len(listes_de_probas_conditionelles))
        for j, proba_conditionelle in enumerate(listes_de_probas_conditionelles):
            res = 1.0
            for i,  proba in  enumerate(proba_conditionelle):
                res *= prob(proba_conditionelle[i], train_set[j][i])
                

            q_theta_predit[j] = res
            # print(res)
        # c'est bon on a les probas, on peut appliquer DKL
    
        loss = Kullback_Leibler(q_theta_predit, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)#conseillé par copilot, évite l'explosion du gradient qui conduit à des nan
        optimizer.step()
        losses.append(loss.item())
        if epoch % (n_iter/10) == 0:
            print(f'Epoch {epoch}: {loss.item()}')
            # for param in model.parameters():
            #     print(param)
           
           
    return losses


