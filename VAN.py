import torch 
import torch.nn as nn
import numpy as np



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


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # à cette ligne on a multiplié x par la matrice de masque (triangulaire inférieure), puis appliqué la fonction d'activation
        # donc la première coordonnée de x vaut activation(0) (normal, s^_1 ne dépend de personne)
        # il faut donc ajouter 0.5 à la première coordonnée pour montrer qu'elle vaut 0 et 1 avec proba 0.5
        x[0] = 0.5 
        return x



def Kulback_Leibler(q,p): # p et q sont des listes de probas d'observation pour le même s
    # ex : p=[0.5, 0.5, 0] et q=[0.3, 0.3, 0.4]
    for i in range(len(p)):
        if p[i] == 0:
            p[i] = 1e-10
    result = 0 
    for i in range(len(p)):
        result += np.log(q[i]/p[i])
    return result  


def train(model, p_obj,  n_iter=100, lr=1e-2, train_size=100):
    losses = []
    # p_obj est la distribution à approximer. 
    # elle prend en input un vecteur de 0 et 1 de taille model.input_size et renvoie la proba de ce vecteur
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    # à cette étape, on a un train set, on peut entraîner le modèle
    for epoch in range(n_iter):
        optimizer.zero_grad() # What is this step? IMPORTANT LINE


        
        # il faut tirer un train_set grâce au modèle pour s'entraîner dessus
        train_set=torch.zeros((train_size, model.input_size))
        for i in range(train_size):
            # pour tirer un x dans x_train, on tire une bernoulli de paramètre 0.5 : c'est s_1
            # comment avoir s_2 ? ON fait passer le vecteur [s_1, 0, 0, 0] dans le réseau de neurones
            # en sortie du réseau de neurones, on a y =[ s1 randomn, p(s2|s1), p(s3|s1), p(s4|s1)...]
            # la deuxième coordonnée de y est p(s2|s1), donc on tire une bernoulli de paramètre p(s2|s1)
            # puis on recommence !
            # on fait ça train_size fois
            train_set[i][0]=torch.bernoulli(torch.tensor(0.5).detach())

            for j in range(1, model.input_size):
                y_pred=model(train_set[i])
                p_j= y_pred[j] # c'est p(s_j|s_{i<j})
                train_set[i][j] = torch.tensor(float(np.random.binomial(1, p_j.detach().numpy()))) # on tire une bernoulli de paramètre p(s_j|s_{i<j}) pour la j-ème variable
        
        y_train=torch.tensor([p_obj(s) for s in train_set])
        # on a notre train set pour cette époque


        listes_de_probas_conditionelles=model(train_set) # on récupère les probas conditionelles, il faut les multiplier pour avoir les probas tout court
        q_theta_predit=[]
        for proba_conditionelle in listes_de_probas_conditionelles:
            res=1
            for i in proba_conditionelle:
                res*=i
            q_theta_predit.append(res)
        # c'est bon on a les probas, on peut appliquer DKL
        loss = Kulback_Leibler(torch.tensor(q_theta_predit), y_train)
        loss.requires_grad = True   # j'ai du ajouter cette ligne pour que ça marche
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % (n_iter/10) == 0:
            print(f'Epoch {epoch}: {loss.item()}')
    return losses



