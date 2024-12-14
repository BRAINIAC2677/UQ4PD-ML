from torch import nn
import torch_uncertainty.layers.bayesian as bayesian


class BNN(nn.Module):
    def __init__(self, n_features):
        super(BNN, self).__init__()
        self.fc1 = bayesian.BayesLinear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.fc2 = bayesian.BayesLinear(in_features=self.fc1.out_features, out_features=1, bias=True)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.hidden_activation(self.fc1(x))
        y = self.fc2(x1)
        # y = self.sig(y)
        return y

   
class ShallowBNN(nn.Module):
    def __init__(self, n_features):
        super(ShallowBNN, self).__init__()
        self.fc = bayesian.BayesLinear(in_features=n_features, out_features=1, bias=True)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        # y = self.sig(y)
        return y

