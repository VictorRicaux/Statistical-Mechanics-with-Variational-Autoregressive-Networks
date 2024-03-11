import torch 
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.distributions import Bernoulli


def prob(s_hat, s):
    ''''
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
        M = torch.triu(torch.ones((input_size,input_size), dtype=torch.float),diagonal=1).t()

        self.fc1 = MaskedLinear(input_size, input_size, mask=M) 
        for param in self.parameters():
            param.requires_grad = True
            init.uniform_(param, -1, 1)  # Initialize parameters randomly between -1 and 1

    def forward(self, x):
        '''
        Compute the parameters of the Bernoulli distribution for each spin
        '''
        x = self.fc1(x)
        x = self.activation(x)
        # à cette ligne on a multiplié x par la matrice de masque (triangulaire inférieure), puis appliqué la fonction d'activation
        # donc la première coordonnée de x vaut activation(0) =0.5 (normal, s^_1 ne dépend de personne)
        return x
    
    
    def prob_of_spins(self, spins):
        '''
        Fonction vectorisée sur un batch
        '''
        params_Bernoulli = self(spins)
      
        probs_per_site = params_Bernoulli * spins + (1 - params_Bernoulli) * (1 - spins)
        return probs_per_site.prod(dim=1)
    
    def log_prob_of_spins(self, spins):
        '''
        Fonction vectorisée sur un batch
        '''
        return torch.log(self.prob_of_spins(spins))
    
    def sample(self, n_samples):
        '''
        Output: n_samples spins sampled from the model q_theta(s)
        '''
        spins = - torch.ones((n_samples, self.input_size))
        for spin_site in range(self.input_size):
            params_Bernoulli = self(spins)                
            spins_at_site = Bernoulli(params_Bernoulli[:, spin_site]).sample()
            spins[:, spin_site] = spins_at_site
        return spins


def Kullback_Leibler(model, log_prob_target, spins):
    '''
    Monte Carlo estimation of the Kullback-Leibler divergence.

    model (torch.nn.Module): model for q_theta(s)
    log_prob_target (function): log-probabilities of the target log(p(s))
    spins (torch.tensor): spins sampled from the model q_theta(s)
    '''
    log_prob_model_value = model.log_prob_of_spins(spins)
    log_pobj = log_prob_target(spins)
    return (log_prob_model_value - log_pobj).mean()

    
def train(model, log_prob_target,  n_iter=100, lr=1e-2, batch_size=100, clip_grad=True):
    '''
    Train the model to approximate the target distribution p_obj.

    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    beta=1

    for epoch in range(n_iter):
        optimizer.zero_grad() # What is this step? IMPORTANT LINE, c'est la prof qui l'a dit

        with torch.no_grad():
            sample =  model.sample(batch_size)
            

        assert not sample.requires_grad

        log_prob = model.log_prob_of_spins(sample)
        
        '''
        Pour l'instant, vous pouvez garder beta = 1.
        Mais l'annealing sera surement utile pour les distributions multi-modales.
        '''
        # 0.998**9000 ~ 1e-8
        beta = beta * (1 + 0.998**9000)
        
        
        with torch.no_grad():
            energy = - log_prob_target(sample)
            loss = log_prob + beta * energy
        assert not energy.requires_grad
        assert not loss.requires_grad
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        loss_reinforce.backward()
        losses.append(loss_reinforce.item())
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if epoch % (n_iter/10) == 0:
            print(f'Epoch {epoch}: {loss_reinforce.item()}')
           
           
    return losses


class VAN_2D(nn.Module):
    def __init__(self, input_size, activation=torch.sigmoid):
        super(VAN, self).__init__() #initialisation obligatoire
        self.input_size = input_size
        self.activation = activation

        # Création de la matrice de masque : que des 0 sur et au dessus de la diagonale et que des 1 dessous
        M = torch.triu(torch.ones((input_size,input_size), dtype=torch.float),diagonal=1).t()

        self.fc1 = MaskedLinear(input_size, input_size, mask=M) 
        for param in self.parameters():
            param.requires_grad = True
            init.uniform_(param, -1, 1)  # Initialize parameters randomly between -1 and 1



            


