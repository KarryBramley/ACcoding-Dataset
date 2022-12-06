from torch_rgcn.utils import *
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import tqdm
from scipy.sparse import coo_matrix, dok_matrix


class DistMult(Module):
    """ DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """
    def __init__(self,
                 indim,
                 outdim,
                 num_nodes,
                 num_rel,
                 w_init='standard-normal',
                 w_gain=False,
                 b_init=None):
        super(DistMult, self).__init__()
        self.w_init = w_init
        self.w_gain = w_gain
        self.b_init = b_init

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create weights & biases
        # 这个relations就是DistMult乘在中间的那个矩阵R_r，形状为[num_rel, out_feat]
        self.relations = nn.Parameter(torch.FloatTensor(indim, outdim).to(device))
        if b_init:
            self.sbias = Parameter(torch.FloatTensor(num_nodes))
            self.obias = Parameter(torch.FloatTensor(num_nodes))
            self.pbias = Parameter(torch.FloatTensor(num_rel))
        else:
            self.register_parameter('sbias', None)
            self.register_parameter('obias', None)
            self.register_parameter('pbias', None)

        self.initialise_parameters()

    def initialise_parameters(self):
        """
        Initialise weights and biases

        Options for initialising weights include:
            glorot-uniform - glorot (aka xavier) initialisation using a uniform distribution
            glorot-normal - glorot (aka xavier) initialisation using a normal distribution
            schlichtkrull-uniform - schlichtkrull initialisation using a uniform distribution
            schlichtkrull-normal - schlichtkrull initialisation using a normal distribution
            normal - using a standard normal distribution
            uniform - using a uniform distribution

        Options for initialising biases include:
            ones - setting all values to one
            zeros - setting all values to zero
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """
        # Weights
        init = select_w_init(self.w_init)
        if self.w_gain:
            gain = nn.init.calculate_gain('relu')
            init(self.relations, gain=gain)
        else:
            init(self.relations)

        # Biases
        if self.b_init:
            init = select_b_init(self.b_init)
            init(self.sbias)
            init(self.pbias)
            init(self.obias)

    def s_penalty(self, triples, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = split_spo(triples)

        s, p, o = nodes[s_index, :], self.relations[p_index, :], nodes[o_index, :]

        return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()

    def forward(self, triples, nodes):
        """ Score candidate triples """

        s_index, p_index, o_index = split_spo(triples)
        # s,p,o都是[triple_size, emb_dim]
        s, p, o = nodes[s_index, :], self.relations[p_index, :], nodes[o_index, :]
        scores = (s * p * o).sum(dim=-1)

        if self.b_init:
            scores = scores + (self.sbias[s_index] + self.pbias[p_index] + self.obias[o_index])

        return scores


class DotMult(Module):
    """ Just multiply two entities vectors for scoring function """
    def __init__(self,
                 num_nodes,
                 b_init=None):
        super(DotMult, self).__init__()
        self.b_init = b_init

        # Create weights & biases
        if b_init:
            self.sbias = Parameter(torch.FloatTensor(num_nodes))
            self.obias = Parameter(torch.FloatTensor(num_nodes))
        else:
            self.register_parameter('sbias', None)
            self.register_parameter('obias', None)

        self.initialise_parameters()

    def initialise_parameters(self):
        """
        Initialise biases
        """
        if self.b_init:
            init = select_b_init(self.b_init)
            init(self.sbias)
            init(self.obias)

    def s_penalty(self, triples, nodes):
        """ Compute Schlichtkrull L2 penalty for the decoder """

        s_index, p_index, o_index = split_spo(triples)
        s, o = nodes[s_index, :], nodes[o_index, :]
        return s.pow(2).mean() + o.pow(2).mean()

    def forward(self, triples, nodes):
        """ Score candidate triples """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        s_index, p_index, o_index = split_spo(triples.to(device))

        # s, o都是[triple_size, emb_dim]
        s, o = nodes[s_index, :], nodes[o_index, :]
        scores = (s * o).sum(dim=-1)

        if self.b_init:
            scores = scores + (self.sbias[s_index] + self.obias[o_index])

        return scores


class LSTMRelationalGraphConvolution(Module):
    """
    Get h_j' of entities' neighborhoods through a LSTM layer.
    """

    def __init__(self,
                 num_nodes=None,
                 num_relations=None,
                 in_features=None,
                 hidden_features=None,
                 out_features=None,
                 lstm_w_init='xavier-normal',
                 rgcn_w_init='glorot-normal',
                 w_gain=False,
                 b_init=None,
                 activation='relu'):
        super(LSTMRelationalGraphConvolution, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.original_num_relations = int((self.num_relations - 1) / 2)  # Count without inverse and self-relations
        # self.in_feature = in_features
        # self.out_features = out_features
        self.lstm_w_init = lstm_w_init
        self.rgcn_w_init = rgcn_w_init
        self.w_gain = w_gain
        self.b_init = b_init
        self.in_dim = in_features if in_features is not None else num_nodes
        self.hidden_dim = hidden_features
        self.out_dim = out_features

        # Create weight parameters
        self.rgc_weights = Parameter(torch.FloatTensor(num_relations, self.hidden_dim, self.out_dim).to(device))

        # Create bias parameters
        if b_init:
            self.rgc_bias = Parameter(torch.FloatTensor(self.out_dim).to(device))
            self.bias = True
        else:
            self.register_parameter('rgc_bias', None)
            self.bias = False

        self.lstm = nn.LSTM(self.in_dim, self.hidden_dim, bias=self.bias, batch_first=True)
        self.activation_func = select_activation(activation)

        self.init_weights()
        if self.bias:
            self.init_biases()

    def init_weights(self):

        w_gain = self.w_gain
        if w_gain:
            gain = nn.init.calculate_gain('relu')
        else:
            gain = 1.0
        lstm_init = select_w_init(self.lstm_w_init)
        rgcn_init = select_w_init(self.rgcn_w_init)

        lstm_init(self.lstm.weight_ih_l0, gain=gain)
        lstm_init(self.lstm.weight_hh_l0, gain=gain)

        rgcn_init(self.rgc_weights, gain=gain)

    def init_biases(self):
        b_init = self.b_init
        init = select_b_init(b_init)

        init(self.lstm.bias_ih_l0)
        init(self.lstm.bias_hh_l0)
        init(self.rgc_bias)

    def forward(self, neighbors_list, nodes_list, adj, features, batch_size=16):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rng = tqdm.trange

        # 查询已有的embedding, feat_embeddings: [num_nodes+1, emb_dim/num_nodes]
        feat_embeddings = torch.cat((features.to(device), torch.zeros(1, self.in_dim, device=device)), dim=0)
        sample_nodes_num = len(nodes_list)

        # LSTM部分
        print('calculating h_j through LSTM...')
        j_indices_s, j_indices_po, j_relations, j_weights = torch.empty([0], dtype=int), torch.empty([0], dtype=int), torch.empty([0], dtype=int), torch.empty([0], dtype=int)
        j_embeddings = torch.empty([0, self.hidden_dim])

        for fr in rng(0, sample_nodes_num, batch_size):
            to = min(fr + batch_size, sample_nodes_num)
            batch = neighbors_list[fr: to]
            batch_s = nodes_list[fr: to]
            batch_neighbor_len = [len(seq) for seq in batch]

            bn = len(batch)

            max_neighbor_len = max(batch_neighbor_len)
            neighbors = -1 * torch.ones([bn, max_neighbor_len, 4])  # 最里层[o, p, w, t]

            for i, l in enumerate(batch):
                neighbors[i, :len(l)] = torch.tensor(l)

            # [batch_size, max_neighbor_num]
            neighbor_sequences = neighbors[:, :, 0]

            # feat: [batch_size, max_neighbor_num]
            # id为0的节点对应着feat_one_hot或feat_embeddings中的第一行的，所以这里将seq为-1的位置替换为num_nodes，以便后面查表将其onehot设为0，
            feat = torch.where(neighbor_sequences != -1, neighbor_sequences,
                               self.num_nodes * torch.ones_like(neighbor_sequences)).long()

            # 获取邻居的embedding: [batch_size, max_neighbor_num, emb_dim/num_nodes]
            neighbor_features = F.embedding(feat, feat_embeddings)

            # 输入lstm: [batch_size, max_neighbor_num, emb_dim]
            # neighbor_lens = torch.ne(neighbor_sequences, -1).sum(dim=1)
            packed_input = pack_padded_sequence(neighbor_features, batch_neighbor_len, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)

            padded_output, output_lens = pad_packed_sequence(packed_output, batch_first=True)

            # 将输出的隐藏状态与节点id、结果、权重连接(时间不需要，因为已经排好序了）
            timed_output = torch.cat([neighbors[:, :max_neighbor_len, :3], padded_output], dim=2)

            # 处理seq长度与节点数不匹配的问题，每个(relation，node)选择时间最大的
            # (indices_s, indices_po, relations, weights, embeddings)，前四个[unique_edge_num,]，embedding:[unique_edge_num, emb_dim]
            batched_hidden_j = select_max_time_neighbors(timed_output, batch_s, batch_neighbor_len, self.num_nodes)

            j_indices_s = torch.cat([j_indices_s, batched_hidden_j[0]])
            j_indices_po = torch.cat([j_indices_po, batched_hidden_j[1]])
            j_relations = torch.cat([j_relations, batched_hidden_j[2]])
            j_weights = torch.cat([j_weights, batched_hidden_j[3]])
            j_embeddings = torch.cat([j_embeddings, batched_hidden_j[4]])

        # RGCN聚合部分
        # 计算A*H*W，feature就是上一层的H
        # A里面已经包含了c_i,r; 因为weight是[num_rel, in_dim, out_dim]形状的，所以不同关系有不同的weight
        output = torch.zeros([self.num_nodes, self.out_dim])
        print('aggregating hidden vectors with A...')
        for i in rng(0, len(j_embeddings)):
            s, po, rel = j_indices_s[i].item(), j_indices_po[i].item(), j_relations[i].item()
            emb = j_embeddings[i]
            output[s] += torch.matmul(emb, self.rgc_weights[rel]) * adj[s, po]

        if self.rgc_bias is not None:
            output = torch.add(output, self.rgc_bias)

        output = self.activation_func(output)

        return output
