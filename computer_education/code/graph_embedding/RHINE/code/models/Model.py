# coding:utf-8
# author: lu yf
# create date: 2018/2/6
# Based on openKePyTorch: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

    def get_postive_IRs(self):
        """
        sample positive IRs triples
        :return:
        """
        self.postive_h_IRs = Variable(torch.from_numpy(self.config.batch_h_IRs[0:self.config.batch_size_IRs]))
        self.postive_t_IRs = Variable(torch.from_numpy(self.config.batch_t_IRs[0:self.config.batch_size_IRs]))
        self.postive_r_IRs = Variable(torch.from_numpy(self.config.batch_r_IRs[0:self.config.batch_size_IRs]))
        self.postive_w_IRs = Variable(torch.from_numpy(self.config.batch_w_IRs[0:self.config.batch_size_IRs]))
        return self.postive_h_IRs, self.postive_t_IRs, self.postive_r_IRs, self.postive_w_IRs

    def get_negtive_IRs(self):
        """
        sample negative IRs triples
        :return:
        """
        self.negtive_h_IRs = Variable(
            torch.from_numpy(self.config.batch_h_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs]))
        self.negtive_t_IRs = Variable(
            torch.from_numpy(self.config.batch_t_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs]))
        self.negtive_r_IRs = Variable(
            torch.from_numpy(self.config.batch_r_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs]))
        self.negtive_w_IRs = Variable(
            torch.from_numpy(self.config.batch_w_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs]))

        return self.negtive_h_IRs, self.negtive_t_IRs, self.negtive_r_IRs, self.negtive_w_IRs

    def get_pid_labels(self, t_IRs):
        """
        find positive probelm's labels
        :return:
        """
        pid_label_dict = self.config.pid_label_dict
        self.source_labels = torch.Tensor([pid_label_dict[t.item()] if (t.item() in pid_label_dict.keys()) else -1 for t in t_IRs])
        return self.source_labels

    def get_postive_ARs(self):
        self.postive_h_ARs = Variable(torch.from_numpy(self.config.batch_h_ARs[0:self.config.batch_size_ARs]))
        self.postive_t_ARs = Variable(torch.from_numpy(self.config.batch_t_ARs[0:self.config.batch_size_ARs]))
        self.postive_r_ARs = Variable(torch.from_numpy(self.config.batch_r_ARs[0:self.config.batch_size_ARs]))
        self.postive_w_ARs = Variable(torch.from_numpy(self.config.batch_w_ARs[0:self.config.batch_size_ARs]))
        return self.postive_h_ARs, self.postive_t_ARs, self.postive_r_ARs, self.postive_w_ARs

    def get_negtive_ARs(self):
        self.negtive_h_ARs = Variable(
            torch.from_numpy(self.config.batch_h_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs]))
        self.negtive_t_ARs = Variable(
            torch.from_numpy(self.config.batch_t_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs]))
        self.negtive_r_ARs = Variable(
            torch.from_numpy(self.config.batch_r_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs]))
        self.negtive_w_ARs = Variable(
            torch.from_numpy(self.config.batch_w_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs]))

        return self.negtive_h_ARs, self.negtive_t_ARs, self.negtive_r_ARs, self.negtive_w_ARs

    def predict(self):
        pass

    def forward(self):
        pass

    def loss_func(self):
        pass


class Classification(nn.Module):
    """ Calculate the classification result of y with two-layer fully-connected net. """

    def __init__(self, input_dim, output_dim, dropout=0., bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, inputs, target_labels):
        mask = torch.ne(target_labels, -1)
        pre_labels = self.linear(inputs)
        loss = nn.CrossEntropyLoss()
        cross_entropy_loss = loss(pre_labels[mask], target_labels[mask].long())
        return cross_entropy_loss

    def get_linear_model(self):
        return self.linear
