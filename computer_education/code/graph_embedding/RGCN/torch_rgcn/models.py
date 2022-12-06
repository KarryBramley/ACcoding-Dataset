from torch_rgcn.layers import RelationalGraphConvolutionRP, DistMult, DotMult
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
                 nfeat=None,
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
        decomposition = encoder_config["decomposition"] if "decomposition" in encoder_config else None
        encoder_w_init = encoder_config["weight_init"] if "weight_init" in encoder_config else None
        encoder_gain = encoder_config["include_gain"] if "include_gain" in encoder_config else False
        encoder_b_init = encoder_config["bias_init"] if "bias_init" in encoder_config else None

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
        self.num_rels = nrel
        self.rgcn_layers = rgcn_layers
        self.nemb = nemb
        self.n2i = n2i

        self.decoder_l2_type = decoder_l2_type
        self.decoder_l2 = decoder_l2

        if nemb is not None:
            self.node_embeddings = nn.Parameter(torch.FloatTensor(nnodes, nemb))
            self.node_embeddings_bias = nn.Parameter(torch.zeros(1, nemb))
            init = select_w_init(encoder_w_init)
            init(self.node_embeddings)

        if nfeat is None:
            nfeat = self.nemb

        self.rgc1 = RelationalGraphConvolutionRP(
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,  # 原本的关系数(nrel)，加上逆转了之后的关系数(nrel)，还有self-loop(1)
            in_features=nfeat,
            out_features=nhid1,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False,
            w_init=encoder_w_init,
            w_gain=encoder_gain,
            b_init=encoder_b_init
        )
        if rgcn_layers == 2:
            self.rgc2 = RelationalGraphConvolutionRP(
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid1,
                out_features=nhid2,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=False,
                w_init=encoder_w_init,
                w_gain=encoder_gain,
                b_init=encoder_b_init
            )

        # Decoder
        out_feat = nemb  # TODO sort this out
        if decoder_config['model'] == 'distmult':
            self.scoring_function = DistMult(nrel, out_feat, nnodes, nrel, decoder_w_init, decoder_gain, decoder_b_init)
        elif decoder_config['model'] == 'dotmult':
            self.scoring_function = DotMult(nnodes, decoder_w_init, decoder_b_init)

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

    def forward(self, graph, triples=None, ret_embs=False, log=None, save_dir=None):
        """ Embed relational graph and then compute score """

        if self.nemb is not None:
            x = self.node_embeddings + self.node_embeddings_bias
            x = torch.nn.functional.relu(x)
            x = self.rgc1(graph, features=x)
        else:
            x = self.rgc1(graph)

        if self.rgcn_layers == 2:
            x = self.rgc2(graph, features=x)
        # print('x:', x, file=log)

        if ret_embs:
            # self.save_embeddings(x.detach(), save_dir)
            return x.detach()
        else:
            assert triples is not None
            scores = self.scoring_function(triples, x)  # 用DistMult，就是计算sum(s*p*o)
            penalty = self.compute_penalty(triples, x)  # 如果需要的话就多计算一步L2惩罚
            return scores, penalty, x.detach()
