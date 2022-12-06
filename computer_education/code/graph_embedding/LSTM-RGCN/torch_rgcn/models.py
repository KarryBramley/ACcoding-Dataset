from torch_rgcn.layers import DistMult, DotMult, LSTMRelationalGraphConvolution
from torch_rgcn.utils import *
import torch.nn.functional as F
from torch import nn
import torch
import os
import json


######################################################################################
# Models for Experiment Reproduction
######################################################################################


class RelationPredictor(nn.Module):
    """ Relation Prediction via RGCN encoder and DistMult decoder """
    def __init__(self,
                 data_name=None,
                 nnodes=None,
                 nrel=None,
                 n2i=None,
                 encoder_config=None,
                 decoder_config=None):
        super(RelationPredictor, self).__init__()

        # Encoder config
        nemb = encoder_config["node_embedding"] if "node_embedding" in encoder_config else None
        nhid1 = encoder_config["hidden1_size"] if "hidden1_size" in encoder_config else None
        nhid2 = encoder_config["hidden2_size"] if "hidden2_size" in encoder_config else None
        rgcn_layers = encoder_config["num_layers"] if "num_layers" in encoder_config else 2
        edge_dropout = encoder_config["edge_dropout"] if "edge_dropout" in encoder_config else None
        lstm_w_init = encoder_config["lstm_weight_init"] if "lstm_weight_init" in encoder_config else None
        rgcn_w_init = encoder_config["rgcn_weight_init"] if "rgcn_weight_init" in encoder_config else None
        encoder_gain = encoder_config["include_gain"] if "include_gain" in encoder_config else False
        encoder_b_init = encoder_config["bias_init"] if "bias_init" in encoder_config else None
        encoder_activation = encoder_config["activation"] if "activation" in encoder_config else None

        # Decoder config
        decoder_l2_type = decoder_config["l2_penalty_type"] if "l2_penalty_type" in decoder_config else None
        decoder_l2 = decoder_config["l2_penalty"] if "l2_penalty" in decoder_config else None
        decoder_w_init = decoder_config["weight_init"] if "weight_init" in decoder_config else None
        decoder_gain = decoder_config["include_gain"] if "include_gain" in decoder_config else False
        decoder_b_init = decoder_config["bias_init"] if "bias_init" in decoder_config else None

        assert (nnodes is not None or nrel is not None or nhid1 is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"
        assert 0 < rgcn_layers < 3, "Only supports the following number of convolution layers: 1 and 2."

        self.data_name = data_name

        self.num_nodes = nnodes
        self.num_relations = nrel * 2 + 1
        self.rgcn_layers = rgcn_layers
        self.nemb = nemb
        self.n2i = n2i

        self.decoder_l2_type = decoder_l2_type
        self.decoder_l2 = decoder_l2
        self.edge_dropout = edge_dropout

        # self.lstm1 = LSTMRelationalGraphConvolution(
        #     num_nodes=nnodes,
        #     num_relations=nrel * 2 + 1,
        #     out_features=nemb,
        #     w_init=encoder_w_init,
        #     w_gain=encoder_gain,
        #     b_init=encoder_b_init,
        # )
        #
        # self.rgc1 = RelationalGraphConvolutionRP(
        #     num_nodes=nnodes,
        #     num_relations=nrel * 2 + 1,  # 原本的关系数(nrel)，加上逆转了之后的关系数(nrel)，还有self-loop(1)
        #     in_features=nemb,
        #     out_features=nhid1,
        #     w_init=encoder_w_init,
        #     w_gain=encoder_gain,
        #     b_init=encoder_b_init
        # )

        self.lstm_rgc1 = LSTMRelationalGraphConvolution(
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1, # 原本的关系数(nrel)，加上逆转了之后的关系数(nrel)，还有self-loop(1)
            hidden_features=nemb,
            out_features=nhid1,
            lstm_w_init=lstm_w_init,
            rgcn_w_init=rgcn_w_init,
            w_gain=encoder_gain,
            b_init=encoder_b_init,
            activation=encoder_activation
        )

        if rgcn_layers == 2:
            # self.lstm2 = LSTMRelationalGraphConvolution(
            #     num_nodes=nnodes,
            #     num_relations=nrel * 2 + 1,
            #     in_features=nhid1,
            #     out_features=nemb,
            #     w_init=encoder_w_init,
            #     w_gain=encoder_gain,
            #     b_init=encoder_b_init
            # )
            # self.rgc2 = RelationalGraphConvolutionRP(
            #     num_nodes=nnodes,
            #     num_relations=nrel * 2 + 1,
            #     in_features=nemb,
            #     out_features=nhid2,
            #     w_init=encoder_w_init,
            #     w_gain=encoder_gain,
            #     b_init=encoder_b_init
            # )
            self.lstm_rgc2 = LSTMRelationalGraphConvolution(
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid1,
                hidden_features=nemb,
                out_features=nhid2,
                lstm_w_init=lstm_w_init,
                rgcn_w_init=rgcn_w_init,
                w_gain=encoder_gain,
                b_init=encoder_b_init,
                activation=encoder_activation
            )

        # Decoder
        out_feat = nemb  # TODO sort this out
        if decoder_config['model'] == 'distmult':
            self.scoring_function = DistMult(nrel, out_feat, nnodes, nrel, decoder_w_init, decoder_gain, decoder_b_init)
        elif decoder_config['model'] == 'dotmult':
            self.scoring_function = DotMult(nnodes, decoder_b_init)

    def compute_penalty(self, batch, x):
        """ Compute L2 penalty for decoder """
        if self.decoder_l2 == 0.0:
            return 0

        # TODO Clean this up
        if self.decoder_l2_type == 'schlichtkrull-l2':
            return self.scoring_function.s_penalty(batch, x)
        else:
            return self.scoring_function.relations.pow(2).sum()

    def save_embeddings(self, embeddings, save_dir):
        if save_dir is None:
            save_dir = 'experiments/' + self.data_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        emb_path = os.path.join(save_dir, 'embeddings.json')

        f_emb = open(emb_path, 'w')
        i2n = {i: n for n, i in self.n2i.items()}
        emb_dict = {}
        for i, emb in enumerate(embeddings.tolist()):
            emb_dict[i2n[i]] = emb
        f_emb.write(json.dumps(emb_dict))
        print('Node embeddings have been saved!')

    def save_models(self, save_dir):
        if save_dir is None:
            save_dir = 'experiments/' + self.data_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        model_path = os.path.join(save_dir, 'model.pkl')

        model_dict = {'model_state': self.state_dict()}
        torch.save(model_dict, model_path)
        print('Model has been saved!')

    def load_model(self):
        load_dir = 'experiments/exp_rp_' + self.data_name
        emb_path = os.path.join(load_dir, 'embeddings.json')
        model_path = os.path.join(load_dir, 'model_pkl')

        embeddings = json.load(open(emb_path, 'r'))
        model_dict = torch.load(model_path)
        return embeddings, model_dict

    def forward(self, graph, nodes, total_triples=None, ret_embs=False, log=None, save_dir=None):
        """ Embed relational graph and then compute score """

        # 找到节点的不等长邻居列表L: 最外层长度sample_num_nodes, 中间neighbor_len，最里层为每个节点对应邻居的[o,p,w,t](按t顺序排列好的)
        neighbors_list = generate_neighbors_lists(graph, nodes, self.num_nodes)

        # 生成RGCN的稀疏矩阵A，仅用于保存系数1/c_ij
        adj = generate_adj_coefficient(graph, self.num_nodes, self.num_relations)

        # 在第一层没有提供feature的情况下，用节点的one-hot代替， emb_dim = one_hot_dim = num_nodes
        print('-' * 25, 'layer 1', '-' * 25)
        # print('nodes_list:', sorted(nodes), file=log)
        one_hot_features = torch.eye(self.num_nodes)    # one_hot_features: [num_nodes, num_nodes]
        x = self.lstm_rgc1(neighbors_list, nodes, adj, features=one_hot_features)   # x: [num_nodes, hid_dim1]

        if self.rgcn_layers == 2:
            print('-' * 25, 'layer 2', '-' * 25)
            x = self.lstm_rgc2(neighbors_list, nodes, adj, features=x)  # x: [num_nodes, hid_dim2]

        # print('non0:', x.ne(0).sum(dim=1).flatten().tolist(), file=log)
        # print('x:', x, file=log)

        if ret_embs:
            # self.save_embeddings(x.detach(), save_dir)
            return x.detach()
        else:
            assert total_triples is not None
            scores = self.scoring_function(total_triples, x)  # 用DistMult，就是计算sum(s*p*o)
            penalty = self.compute_penalty(total_triples, x)  # 如果需要的话就多计算一步L2惩罚
            return scores, penalty, x.detach()
