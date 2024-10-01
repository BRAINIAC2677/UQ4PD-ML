import torch
from torch import nn
import baal.bayesian.dropout as mcdropout
from torch_uncertainty.layers.bayesian import BayesLinear

from constants import MODEL_SUBSETS

'''
Unimodal models
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
BNN variants of uni-modal models
'''
class BNN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(BNN, self).__init__()
        self.fc1 = BayesLinear(in_features=n_features, out_features=int(n_features / 2), bias=True)
        self.drop1 = mcdropout.Dropout(p=drop_prob)
        self.fc2 = BayesLinear(in_features=self.fc1.out_features, out_features=1, bias=True)
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
        # Replace nn.Linear with BayesLinear
        self.fc = BayesLinear(in_features=n_features, out_features=1, bias=True)
        self.drop = mcdropout.Dropout(p=drop_prob)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        y = self.drop(y)
        y = self.sig(y)
        return y

'''
Final predictor
Contains two modules:
    1. a custom cross-attention module
    2. prediction network
'''
class CrossAttention(nn.Module):
    def __init__(self, input_dim, query_dim, drop_prob, uncertainty_weight, n_embeddings=3):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.drop_prob = drop_prob
        self.uncertainty_weight = uncertainty_weight
        self.n_embeddings = n_embeddings
        
        self.form_query = torch.nn.Linear(input_dim, query_dim)
        self.form_key = torch.nn.Linear(input_dim, query_dim)
        self.form_value = torch.nn.Linear(input_dim, query_dim)

        self.drop = mcdropout.Dropout(p = drop_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.final_layer = torch.nn.Linear((self.n_embeddings-1) * self.input_dim, self.input_dim)

    def forward(self, features, prediction_variances):
        prediction_variances = torch.stack(prediction_variances).transpose(0,1) #(n, N)
        
        queries = []
        keys = []
        values = []
        for i in range(self.n_embeddings):
            q = self.form_query(features[i])
            q = self.drop(q)

            key = self.form_key(features[i])
            key = self.drop(q)

            val = self.form_value(features[i])
            val = self.drop(val)

            queries.append(q)
            keys.append(key)
            values.append(val)

        queries = torch.stack(queries) #(N, n, d)
        queries = queries.transpose(0,1) #(n, N, d)
        keys = torch.stack(keys) #(N, n, d)
        keys_T = keys.transpose(0,1).transpose(-1,-2) #(n, d, N)
        values = torch.stack(values) #(N, n, d)
        values = values.transpose(0,1) #(n, N, d)

        scores = torch.matmul(queries, keys_T) #(n, N, N)
        #scores = self.softmax(scores) #the mid dimension sums up to 1, e.g., rows of scores[0]

        vars = prediction_variances.repeat(1, self.n_embeddings).reshape(-1, self.n_embeddings, self.n_embeddings) #(n, N, N)
        vars = vars + prediction_variances.unsqueeze(dim=-1) #(n, N, N)
        scores = scores - self.uncertainty_weight*vars #(n, N, N)
        scores = self.softmax(scores)
        
        zs = torch.matmul(scores, values) #(n, N, d)
        z = zs.reshape((-1, self.n_embeddings*self.query_dim)) #(n, N*d)
        return z

class CrossAttentionBNN(nn.Module):
    def __init__(self, input_dim, query_dim, drop_prob, uncertainty_weight, n_embeddings=3):
        super(CrossAttentionBNN, self).__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.drop_prob = drop_prob
        self.uncertainty_weight = uncertainty_weight
        self.n_embeddings = n_embeddings
        
        self.form_query = BayesLinear(input_dim, query_dim, bias=True)
        self.form_key = BayesLinear(input_dim, query_dim, bias=True)
        self.form_value = BayesLinear(input_dim, query_dim, bias=True)

        self.drop = mcdropout.Dropout(p=drop_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.final_layer = BayesLinear((self.n_embeddings - 1) * self.input_dim, self.input_dim, bias=True)

    def forward(self, features, prediction_variances):
        prediction_variances = torch.stack(prediction_variances).transpose(0, 1)  # (n, N)
        
        queries = []
        keys = []
        values = []
        for i in range(self.n_embeddings):
            q = self.form_query(features[i])
            q = self.drop(q)

            key = self.form_key(features[i])
            key = self.drop(key)  # Fix: should be key, not q

            val = self.form_value(features[i])
            val = self.drop(val)

            queries.append(q)
            keys.append(key)
            values.append(val)

        queries = torch.stack(queries)  # (N, n, d)
        queries = queries.transpose(0, 1)  # (n, N, d)
        keys = torch.stack(keys)  # (N, n, d)
        keys_T = keys.transpose(0, 1).transpose(-1, -2)  # (n, d, N)
        values = torch.stack(values)  # (N, n, d)
        values = values.transpose(0, 1)  # (n, N, d)

        scores = torch.matmul(queries, keys_T)  # (n, N, N)

        vars = prediction_variances.repeat(1, self.n_embeddings).reshape(-1, self.n_embeddings, self.n_embeddings)  # (n, N, N)
        vars = vars + prediction_variances.unsqueeze(dim=-1)  # (n, N, N)
        scores = scores - self.uncertainty_weight * vars  # (n, N, N)
        scores = self.softmax(scores)

        zs = torch.matmul(scores, values)  # (n, N, d)
        z = zs.reshape((-1, self.n_embeddings * self.query_dim))  # (n, N*d)
        return z


class HybridFusionNetworkWithUncertainty(nn.Module):
    def __init__(self, feature_shapes, config):
        super(HybridFusionNetworkWithUncertainty, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.query_dim = config["query_dim"]
        self.last_hidden_dim = config["last_hidden_dim"]
        self.drop_prob = config["dropout_prob"]
        self.uncertainty_weight = config["uncertainty_weight"]
        selected_models = MODEL_SUBSETS[config["model_subset_choice"]]
        number_of_models = len(selected_models)
        self.n_embeddings = number_of_models
        '''
        input: features_i is of shape (feature_shapes[i]); y_pred_score_i
        total input size: feature_shapes[i]+1
        '''
        self.intra_linear = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.cross_attention = CrossAttention(self.hidden_dim, self.query_dim, self.drop_prob, self.uncertainty_weight, self.n_embeddings) #shared weights

        for i in range(self.n_embeddings):
            linear_layer = nn.Linear(in_features=feature_shapes[i], out_features=self.hidden_dim, bias=True)
            self.intra_linear.append(linear_layer)

        self.lin1 = nn.Linear(in_features=((self.n_embeddings*self.query_dim)+self.n_embeddings), out_features=self.last_hidden_dim)
        self.fc = nn.Linear(in_features=self.last_hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.intra_linear_dropout = mcdropout.Dropout(p = self.drop_prob)
        self.lin1_dropout = mcdropout.Dropout(p = self.drop_prob)
    
    def forward(self, inputs):
        (features, predicted_scores, prediction_variances) = inputs
        # print([features[i].shape for i in range(len(features))]) #(n, 232), (n, 1024), (n, 42)
        
        hiddens = []
        for i in range(self.n_embeddings):
            hidden_representation = self.relu(self.intra_linear[i](features[i])) #projection: (n, d_{x_i}) -> (n,d)
            hidden_representation = self.intra_linear_dropout(hidden_representation)
            hidden_representation = self.relu(hidden_representation)
            hidden_representation = self.layer_norm(hidden_representation)
            hiddens.append(hidden_representation)

        pred_scores = torch.stack(predicted_scores).transpose(0,1) #(n, N)
        context = self.cross_attention(torch.cat([torch.unsqueeze(hiddens[k],0) for k in range(self.n_embeddings)]), prediction_variances) #(n, d_q)
        outputs = torch.cat((context, pred_scores),dim=-1) #(n, N+d_q)
        outputs = self.lin1(outputs) #(n, last_hidden_dim)
        outputs = self.lin1_dropout(outputs) 
        logits = self.fc(outputs) #(n,1)
        probs = self.sigmoid(logits) #(n,1)
        return probs

class HybridFusionNetworkWithUncertaintyBNN(nn.Module):
    def __init__(self, feature_shapes, config):
        super(HybridFusionNetworkWithUncertaintyBNN, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.query_dim = config["query_dim"]
        self.last_hidden_dim = config["last_hidden_dim"]
        self.drop_prob = config["dropout_prob"]
        self.uncertainty_weight = config["uncertainty_weight"]
        selected_models = MODEL_SUBSETS[config["model_subset_choice"]]
        number_of_models = len(selected_models)
        self.n_embeddings = number_of_models

        self.intra_linear = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = CrossAttentionBNN(self.hidden_dim, self.query_dim, self.drop_prob, self.uncertainty_weight, self.n_embeddings)  # Update to use BNN CrossAttention

        for i in range(self.n_embeddings):
            linear_layer = BayesLinear(in_features=feature_shapes[i], out_features=self.hidden_dim, bias=True)
            self.intra_linear.append(linear_layer)

        self.lin1 = BayesLinear(in_features=((self.n_embeddings * self.query_dim) + self.n_embeddings), out_features=self.last_hidden_dim, bias=True)
        self.fc = BayesLinear(in_features=self.last_hidden_dim, out_features=1, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.intra_linear_dropout = mcdropout.Dropout(p=self.drop_prob)
        self.lin1_dropout = mcdropout.Dropout(p=self.drop_prob)

    def forward(self, inputs):
        (features, predicted_scores, prediction_variances) = inputs

        hiddens = []
        for i in range(self.n_embeddings):
            hidden_representation = self.relu(self.intra_linear[i](features[i]))  # (n, d_{x_i}) -> (n, d)
            hidden_representation = self.intra_linear_dropout(hidden_representation)
            hidden_representation = self.relu(hidden_representation)
            hidden_representation = self.layer_norm(hidden_representation)
            hiddens.append(hidden_representation)

        pred_scores = torch.stack(predicted_scores).transpose(0, 1)  # (n, N)
        context = self.cross_attention(torch.cat([torch.unsqueeze(hiddens[k], 0) for k in range(self.n_embeddings)]), prediction_variances)  # (n, d_q)
        outputs = torch.cat((context, pred_scores), dim=-1)  # (n, N + d_q)
        outputs = self.lin1(outputs)  # (n, last_hidden_dim)
        outputs = self.lin1_dropout(outputs)
        logits = self.fc(outputs)  # (n, 1)
        probs = self.sigmoid(logits)  # (n, 1)
        return probs

