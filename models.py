import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import time


class Outcome(nn.Module):
    def __init__(self, args):
        super(Outcome, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.n_features = args.n_features
        self.n_embedding = args.n_embedding

        self.embed_fc1 = torch.nn.Linear(self.n_features, 16)
        self.activation1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.embed_fc2 = torch.nn.Linear(16, self.n_embedding)
        self.activation2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(self.n_embedding)
        self.predict_fc1 = torch.nn.Linear(self.n_embedding + 1, 16)
        self.activation3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.predict_fc2 = torch.nn.Linear(16, 1)

        self.layers = nn.Sequential(
            torch.nn.Linear(self.n_embedding + 1, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
        )

        self.shortcut = nn.Sequential(
            torch.nn.Linear(self.n_embedding + 1, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU()
        )

        self.output = nn.Sequential(
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, 1)
        )

        self.growth_rate = 8

        self.layers1 = nn.Sequential(
            torch.nn.Linear(self.n_embedding + 1, self.growth_rate),
            torch.nn.BatchNorm1d(self.growth_rate),
            torch.nn.ReLU())

        self.layers2 = nn.Sequential(
            torch.nn.Linear(self.n_embedding + 1 + self.growth_rate, self.growth_rate),
            torch.nn.BatchNorm1d(self.growth_rate),
            torch.nn.ReLU())

        self.output_densenet= torch.nn.Linear(self.n_embedding + 1 + 2 * self.growth_rate, 1)

    def forward(self, args, data):
        x = data.x
        treatment = data.t

        din = x.view(-1, self.n_features)
        dout = self.bn1(self.activation1((self.embed_fc1(din))))
        embedding = self.bn2(self.activation2(self.embed_fc2(dout))).to(args.device)
        input = torch.cat([treatment, embedding], dim=-1)

        if args.base == 'MLP':
            din = input.view(-1, self.n_embedding + 1)
            dout = self.bn3(self.activation3(self.predict_fc1(din)))
            if args.classify == 1:
                predicted_outcome = torch.sigmoid(self.predict_fc2(dout))
            else:
                predicted_outcome = self.predict_fc2(dout)

        if args.base == 'Resnet':
            din = input.view(-1, self.n_embedding + 1)
            dout = self.layers(din)
            shortcut = self.shortcut(din)
            din = torch.add(dout, shortcut)
            if args.classify == 1:
                predicted_outcome = torch.sigmoid(self.output(din))
            else:
                predicted_outcome = self.output(din)

        if args.base == 'Densenet':
            din = input.view(-1, self.n_embedding + 1)
            dout = self.layers1(din)
            din = torch.cat([din, dout], 1)
            dout = self.layers2(din)
            din = torch.cat([din, dout], 1)
            if args.classify == 1:
                predicted_outcome = torch.sigmoid(self.output_densenet(din))
            else:
                predicted_outcome = self.output_densenet(din)
        return embedding, predicted_outcome


class Treatment(nn.Module):  
    def __init__(self, args):
        super(Treatment, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.n_embedding = args.n_embedding
        self.predictT_fc1 = torch.nn.Linear(self.n_embedding, 16)
        self.activation1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.predictT_fc2 = torch.nn.Linear(16, 1)

    def forward(self, args, embedding):
        input = embedding
        din = input.view(-1, self.n_embedding)
        dout = self.bn1(self.activation1(self.predictT_fc1(din)))
        predicted_treatment = self.predictT_fc2(dout)
        return predicted_treatment


class Resnet(nn.Module):
    def __init__(self, args):
        super(Resnet, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.n_features = args.n_features + 1  
        self.n_embedding = self.n_features
        self.layers = nn.Sequential(
            torch.nn.Linear(self.n_embedding, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
        )

        self.shortcut = nn.Sequential(
            torch.nn.Linear(self.n_embedding, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU()
        )

        self.output = nn.Sequential(
            torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, 1)
        )

    def forward(self, args, data):
        x = data.x
        treatment = data.t
        x = torch.cat([x, treatment], dim=-1)

        din = x.view(-1, self.n_features)
        dout = self.layers(din)
        shortcut = self.shortcut(din)
        din = torch.add(dout, shortcut)
        if args.classify == 1:
            predicted_outcome = torch.sigmoid(self.output(din))
        else:
            predicted_outcome = self.output(din)
        return predicted_outcome


class Densenet(nn.Module):
    def __init__(self, args):
        super(Densenet, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.n_features = args.n_features + 1   
        self.growth_rate = 8

        self.layers1 = nn.Sequential(
            torch.nn.Linear(self.n_features, self.growth_rate),
            torch.nn.BatchNorm1d(self.growth_rate),
            torch.nn.ReLU())

        self.layers2 = nn.Sequential(
            torch.nn.Linear(self.n_features+self.growth_rate, self.growth_rate),
            torch.nn.BatchNorm1d(self.growth_rate),
            torch.nn.ReLU())

        self.output = torch.nn.Linear(self.n_features+2*self.growth_rate, 1)

    def forward(self, args, data):
        x = data.x
        treatment = data.t
        x = torch.cat([x, treatment], dim=-1)

        din = x.view(-1, self.n_features)
        dout = self.layers1(din)
        din = torch.cat([din, dout], 1)
        dout = self.layers2(din)
        din = torch.cat([din, dout], 1)
        if args.classify == 1:
            predicted_outcome = torch.sigmoid(self.output(din))
        else:
            predicted_outcome = self.output(din)
        return predicted_outcome


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.n_features = args.n_features
        self.n_embedding = args.n_features
        self.predict_fc1 = torch.nn.Linear(self.n_embedding + 1, 16)
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(16)
        self.predict_fc = torch.nn.Linear(16, 1)

    def forward(self, args, data):
        x = data.x
        treatment = data.t
        x = torch.cat([x, treatment], dim=-1)

        din = x.view(-1, self.n_features + 1)
        dout = self.bn(self.activation(self.predict_fc1(din)))
        if args.classify == 1:
            predicted_outcome = torch.sigmoid(self.predict_fc(dout))
        else:
            predicted_outcome = self.predict_fc(dout)
        return predicted_outcome















