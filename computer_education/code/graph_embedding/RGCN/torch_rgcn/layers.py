from torch_rgcn.utils import *
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import math
import torch
import torch.nn.functional as F


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
                 w_init='standard-normal',
                 b_init=None):
        super(DotMult, self).__init__()
        self.w_init = w_init
        self.b_init = b_init

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        Initialise weights and biases
        """
        # Weights
        init = select_w_init(self.w_init)

        # Biases
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

        s_index, p_index, o_index = split_spo(triples)
        # s, o都是[triple_size, emb_dim]
        s, o = nodes[s_index, :], nodes[o_index, :]
        scores = (s * o).sum(dim=-1)

        if self.b_init:
            scores = scores + (self.sbias[s_index] + self.obias[o_index])

        return scores


class RelationalGraphConvolutionRP(Module):
    """
    Relational Graph Convolution (RGC) Layer for Relation Prediction
    (as described in https://arxiv.org/abs/1703.06103)
    """

    def __init__(self,
                 num_nodes=None,
                 num_relations=None,
                 in_features=None,
                 out_features=None,
                 edge_dropout=None,
                 edge_dropout_self_loop=None,
                 decomposition=None,
                 vertical_stacking=False,
                 w_init='glorot-normal',
                 w_gain=False,
                 b_init=None):
        super(RelationalGraphConvolutionRP, self).__init__()

        assert (num_nodes is not None or num_relations is not None or out_features is not None), \
            "The following must be specified: number of nodes, number of relations and output dimension!"

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # If featureless, use number of nodes instead as input dimension
        in_dim = in_features if in_features is not None else num_nodes
        out_dim = out_features

        # Unpack arguments
        weight_decomp = decomposition['type'] if decomposition is not None and 'type' in decomposition else None
        num_bases = decomposition['num_bases'] if decomposition is not None and 'num_bases' in decomposition else None
        num_blocks = decomposition['num_blocks'] if decomposition is not None and 'num_blocks' in decomposition else None

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.in_features = in_dim  # TODO in_dim
        self.out_features = out_dim  # TODO out_dim
        self.weight_decomp = weight_decomp
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.vertical_stacking = vertical_stacking
        self.edge_dropout = edge_dropout
        self.edge_dropout_self_loop = edge_dropout_self_loop
        self.w_init = w_init
        self.w_gain = w_gain
        self.b_init = b_init

        # Create weight parameters
        if self.weight_decomp is None:
            self.weights = Parameter(torch.FloatTensor(num_relations, in_dim, out_dim).to(device))
        elif self.weight_decomp == 'basis':
            # Weight Regularisation through Basis Decomposition
            assert num_bases > 0, \
                'Number of bases should be set to higher than zero for basis decomposition!'
            self.bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim).to(device))    # V，一组基变换
            self.comps = Parameter(torch.FloatTensor(num_relations, num_bases).to(device))      # a，每个关系对应的一组系数
        elif self.weight_decomp == 'block':
            # Weight Regularisation through Block Diagonal Decomposition
            assert self.num_blocks > 0, \
                'Number of blocks should be set to a value higher than zero for block diagonal decomposition!'
            assert in_dim % self.num_blocks == 0 and out_dim % self.num_blocks == 0, \
                f'For block diagonal decomposition, input dimensions ({in_dim}, {out_dim}) must be divisible ' \
                f'by number of blocks ({self.num_blocks})'
            self.blocks = nn.Parameter(
                torch.FloatTensor(num_relations - 1, self.num_blocks, in_dim // self.num_blocks,
                                  out_dim // self.num_blocks).to(device))
            self.blocks_self = nn.Parameter(torch.FloatTensor(in_dim, out_dim).to(device))
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

        # Create bias parameters
        if b_init:
            self.bias = Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)

        self.initialise_weights()
        if self.bias is not None:
            self.initialise_biases()

    def initialise_biases(self):
        """
        Initialise bias parameters using one of the following methods:
            ones - setting all values to one
            zeros - setting all values to zero
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """

        b_init = self.b_init
        init = select_b_init(b_init)
        init(self.bias)

    def initialise_weights(self):
        """
        Initialise weights parameters using one of the following methods:
            glorot-uniform - glorot (aka xavier) initialisation using a uniform distribution
            glorot-normal - glorot (aka xavier) initialisation using a normal distribution
            schlichtkrull-uniform - schlichtkrull initialisation using a uniform distribution
            schlichtkrull-normal - schlichtkrull initialisation using a normal distribution
            normal - using a standard normal distribution
            uniform - using a uniform distribution
        """

        w_init = self.w_init
        w_gain = self.w_gain

        # Add scaling factor according to non-linearity function used
        if w_gain:
            gain = nn.init.calculate_gain('relu')
        else:
            gain = 1.0

        # Select appropriate initialisation method
        init = select_w_init(w_init)

        if self.weight_decomp == 'block':
            # TODO Clean this up
            def schlichtkrull_std(shape, gain):
                """
                a = \text{gain} \times \frac{3}{\sqrt{\text{fan\_in} + \text{fan\_out}}}
                """
                fan_in, fan_out = shape[0], shape[1]
                return gain * 3.0 / sqrt(float(fan_in + fan_out))

            def schlichtkrull_normal_(tensor, shape, gain=1.):
                """Fill the input `Tensor` with values according to the Schlichtkrull method, using a normal distribution."""
                std = schlichtkrull_std(shape, gain)
                with torch.no_grad():
                    return tensor.normal_(0.0, std)

            schlichtkrull_normal_(self.blocks, shape=[(self.num_relations-1)//2, self.in_features//self.num_blocks], gain=gain)
            schlichtkrull_normal_(self.blocks_self, shape=[(self.num_relations-1)//2, self.in_features//self.num_blocks], gain=gain)
        elif self.weight_decomp == 'basis':
            init(self.bases, gain=gain)
            init(self.comps, gain=gain)
        else:
            init(self.weights, gain=gain)

    def forward(self, triples, features=None):
        """ Perform a single pass of message propagation """

        assert (features is None) == (self.in_features is None), \
            "Layer has not been properly configured to take in features!"

        # TODO clean up this
        # 处理weights矩阵，是否要正则化
        if self.weight_decomp is None:
            weights = self.weights
        elif self.weight_decomp == 'basis':
            # einsum相当于自己定义一个矩阵乘法，指定维数，这里就是basis-decomposition求权重的线性函数
            weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        elif self.weight_decomp == 'block':
            pass
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        out_dim = self.out_features
        num_nodes = self.num_nodes
        num_relations = self.num_relations
        vertical_stacking = self.vertical_stacking
        original_num_relations = int((self.num_relations-1)/2)  # Count without inverse and self-relations
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        triples = triples.to(device)
        features = features.to(device)

        # TODO - clean this
        # Edge dropout on self-loops
        if self.training and self.edge_dropout["self_loop_type"] != 'schlichtkrull-dropout':
            self_loop_keep_prob = 1 - self.edge_dropout["self_loop"]
        else:
            self_loop_keep_prob = 1

        with torch.no_grad():
            # Add inverse relations，交换三元组的头和尾，关系用一个新的id（旧id+rel_num）替代，[batch_size, 4]，注意这里的size都是dropout之后的结果
            inverse_triples = generate_inverses(triples, original_num_relations)
            # Add self-loops to triples，生成self-loop的三元组，并设置一个新的id（2*rel_num），[batch_size + nodes_num, 4]
            self_loop_triples = generate_self_loops(
                triples, num_nodes, original_num_relations, self_loop_keep_prob, device=device)
            # [2*batch_size + nodes_num, 4]
            triples_plus = torch.cat([triples, inverse_triples, self_loop_triples], dim=0)

        # Stack adjacency matrices (vertically/horizontally)
        # 把三元组压缩成二元组，[triples_plus_size, 4] -> [triples_plus_size, 2]，如果vertical_stacking，则把头和关系(offset)加到一起，尾保持不变；否则相反
        # 先忽略weight，因为这里只得到稀疏矩阵的indices
        adj_indices, adj_size = stack_matrices(
            triples_plus,
            num_nodes,
            num_relations,
            vertical_stacking=vertical_stacking,
            device=device
        )

        num_triples = adj_indices.size(0)
        # vals = torch.ones(num_triples, dtype=torch.float, device=device)
        vals = torch.tensor(triples_plus[:, 2], dtype=torch.float, device=device)

        assert vals.size(0) == (triples.size(0) + inverse_triples.size(0) + self_loop_triples.size(0))

        # 计算1/c_i,r，这里c_i,r就是(节点i,关系r)所连接的节点数
        # Apply normalisation (vertical-stacking -> row-wise rum & horizontal-stacking -> column-wise sum)
        # sum是计算每个(实体+关系)连接的实体个数
        sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
        if not vertical_stacking:
            # Rearrange column-wise normalised value to reflect original order (because of transpose-trick)
            n = triples.size(0)
            i = self_loop_triples.size(0)
            sums = torch.cat([sums[n: 2*n], sums[:n], sums[-i:]], dim=0)
        vals = vals / sums

        # Construct adjacency matrix
        # 在这里构建起二维的稀疏矩阵，用vals值填充adj_indicese提供的位置(s, r*n+o)或(s+r*n, o)
        # 得到的是A矩阵，具体就是提供其他节点的1/c_i,r，以及自己的1
        if device == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
        else:
            adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)

        # TODO - Clean up
        # 计算A*H*W，feature就是上一层的H
        # A里面已经包含了c_i,r; 因为weight是[num_rel, in_dim, out_dim]形状的，所以不同关系有不同的weight
        if self.in_features is None:
            if self.edge_dropout["self_loop_type"] == 'schlichtkrull-dropout':
                raise NotImplementedError()
            # Featureless
            if self.weight_decomp == 'block':
                weights = block_diag(self.blocks)
                weights = torch.cat([weights, self.blocks_self], dim=0)
            output = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
        elif self.vertical_stacking:
            if self.edge_dropout["self_loop_type"] == 'schlichtkrull-dropout':
                raise NotImplementedError()
            # Adjacency matrix vertically stacked
            if self.weight_decomp == 'block':
                weights = block_diag(self.blocks)
                weights = torch.cat([weights, self.blocks_self], dim=0)
            af = torch.spmm(adj, features)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        else:
            # Adjacency matrix horizontally stacked
            if self.weight_decomp == 'block':
                n = features.size(0)
                r = num_relations - 1
                input_block_size = in_dim // self.num_blocks
                output_block_size = out_dim // self.num_blocks
                num_blocks = self.num_blocks
                block_features = features.view(n, num_blocks, input_block_size)
                fw = torch.einsum('nbi, rbio -> rnbo', block_features, self.blocks).contiguous()
                assert fw.shape == (r, n, num_blocks, output_block_size), f"{fw.shape}, {(r, n, num_blocks, output_block_size)}"
                fw = fw.view(r, n, out_dim)
                self_fw = torch.einsum('ni, io -> no', features, self.blocks_self)[None, :, :]
                if self.training and self.edge_dropout["self_loop_type"] == 'schlichtkrull-dropout':
                    self_fw = nn.functional.dropout(self_fw, p=self.edge_dropout["self_loop"], training=True,inplace=False)
                fw = torch.cat([fw, self_fw], dim=0)
                output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))
            else:
                fw = torch.einsum('ni, rio -> rno', features, weights).contiguous()  # contiguous是把原来不是整块存储的tensor转为连续的
                output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))

        assert output.size() == (self.num_nodes, out_dim)
        if self.bias is not None:
            output = torch.add(output, self.bias)

        output = F.elu(output)
        return output
