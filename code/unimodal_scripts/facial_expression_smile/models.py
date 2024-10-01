from torch import nn
import baal.bayesian.dropout as mcdropout
import torch_uncertainty.layers.bayesian as bayesian

'''
ML baselines using pytorch + BAAL Dropout
'''
class ANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.drop1 = mcdropout.Dropout(p = drop_prob)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1,bias=True)
        self.drop2 = mcdropout.Dropout(p = drop_prob)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop1(x1)
        y = self.fc2(x1)
        y = self.drop2(y)
        y = self.sig(y)
        return y


class ShallowANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ShallowANN, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1,bias=True)
        self.drop = mcdropout.Dropout(p = drop_prob)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        y = self.fc(x)
        y = self.drop(y)
        y = self.sig(y)
        return y
 

'''
BNN Variants
'''
class BNN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(BNN, self).__init__()
        self.fc1 = bayesian.BayesLinear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.drop1 = mcdropout.Dropout(p=drop_prob)
        self.fc2 = bayesian.BayesLinear(in_features=self.fc1.out_features, out_features=1, bias=True)
        self.drop2 = mcdropout.Dropout(p=drop_prob)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop1(x1)
        y = self.fc2(x1)
        y = self.drop2(y)
        y = self.sig(y)
        return y

   
class ShallowBNN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ShallowBNN, self).__init__()
        self.fc = bayesian.BayesLinear(in_features=n_features, out_features=1, bias=True)
        self.drop = mcdropout.Dropout(p=drop_prob)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        y = self.drop(y)
        y = self.sig(y)
        return y

