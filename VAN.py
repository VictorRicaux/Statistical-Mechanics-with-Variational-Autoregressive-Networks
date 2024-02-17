import torch 
import torch.nn as nn
import numpy as np




class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask = mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight * self.mask, self.bias)
    
class VAN(nn.Module):
    def __init__(self, input_size, activation=torch.sigmoid):
        super(VAN, self).__init__() #initialisation obligatoire
        self.input_size = input_size

        # Création de la matrice de masque : que des 0 sur et au dessus de la diagonale et que des 1 dessous
        M = torch.zeros((input_size, input_size), dtype=torch.int)
        for i in range(input_size):
            for j in range(i, input_size):
                M[i][j] = 0
        for i in range(1, input_size):
            for j in range(i):
                M[i][j] = 1

        self.fc1 = MaskedLinear(input_size, input_size, mask=M)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # à cette ligne on a multiplié x par la matrice de masque (triangulaire inférieure), puis appliqué la fonction d'activation
        # donc la première coordonnée de x vaut activation(0) (normal, s^_1 ne dépend de personne)
        # il faut donc ajouter 0.5 à la première coordonnée pour montrer qu'elle vaut 0 et 1 avec proba 0.5
        x[0] = 0.5
        return x