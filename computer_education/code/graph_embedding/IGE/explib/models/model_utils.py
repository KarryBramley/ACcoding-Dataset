import torch
from torch import nn
from torch.nn import functional as F


class UnigramSampler(nn.Module):

    def __init__(self, vocab_size, distortion):
        super().__init__()
        self.vocab_size = vocab_size
        self.distortion = distortion
        self.weights = None

    def init_weights(self, freqs):
        if torch.cuda.is_available():
            freqs = torch.cuda.FloatTensor(freqs)
        else:
            freqs = torch.FloatTensor(freqs)
        self.weights = freqs ** self.distortion

    def forward(self, n):
        return self.sample(n)

    def sample(self, n):
        with torch.no_grad():
            samples = torch.multinomial(self.weights, n, True)
        return samples


class AttributedEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_factors, n_attrs):
        super().__init__()
        self.w_vf = nn.Parameter(torch.Tensor(vocab_size, n_factors))
        self.w_df = nn.Parameter(torch.Tensor(n_attrs, n_factors))
        self.w_fk = nn.Parameter(torch.Tensor(n_factors, embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p.data)

    def get_full_embeddings(self):
        uncond_table = self.w_vf @ self.w_fk
        return uncond_table

    def forward(self, ids, attr_vec=None):
        """
        Parameters
        ----------
        ids : torch.LongTensor
            [batch_size, node_ids]
        attr_vec : torch.FloatTensor
            [batch_size, n_attrs]
        Returns
        -------
        emb_vec : torch.FloatTensor
            [batch_size, embed_dim]
        """
        factor_vec = F.embedding(ids, self.w_vf)  # [batch_size, n_factors]
        if attr_vec is not None:
            # conditional embeddings
            af_vec = attr_vec @ self.w_df  # [batch_size, n_factors]
            emb_vec = (factor_vec * af_vec) @ self.w_fk
        else:
            emb_vec = factor_vec @ self.w_fk

        return emb_vec


class NSSoftmax(nn.Module):
    """Softmax with Negative Sampling."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.w_t = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_t.data)
        nn.init.constant_(self.b.data, 0)

    def forward(self, x, true_target, neg_targets):
        """
        x :
            [batch_size, input_size]
        true_target :
            [batch_size]
        neg_targets:
            [n_samples]
        """
        pos_w = F.embedding(true_target, self.w_t)  # [batch_size, input_size]
        neg_w = F.embedding(neg_targets, self.w_t)  # [n_samples, input_size]
        pos_logits = (x * pos_w).sum(dim=1)  # [batch_size]
        neg_logtis = x @ neg_w.t()  # [batch_size, n_samples]
        return pos_logits, neg_logtis


class AttributedNSSoftmax(nn.Module):
    """Attributed Softmax with Negative Sampling."""

    def __init__(self, input_size, output_size, n_factors, attr_size):
        super().__init__()
        self.w_if = nn.Parameter(torch.Tensor(input_size, n_factors))  # input -> factor
        self.w_df = nn.Parameter(torch.Tensor(attr_size, n_factors))  # attr -> factor
        self.w_of = nn.Parameter(torch.Tensor(output_size, n_factors))  # output -> factor

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p.data)

    def forward(self, x, d, true_target, neg_targets):
        """
        x :
            input vector, [batch_size, n_factors]
        d :
            attribute vector, [batch_size, attr_size]
        true_target :
            [batch_size]
        neg_targets :
            [n_samples]
        """
        xf = x @ self.w_if  # [batch_size, n_factors]
        df = d @ self.w_df  # [batch_size, n_factors]
        z = xf * df  # [batch_size, n_factors]
        pos_o = F.embedding(true_target, self.w_of)  # [batch_size, n_factors]
        neg_o = F.embedding(neg_targets, self.w_of)  # [n_samples, n_factors]
        pos_logits = (z * pos_o).sum(dim=1)  # [batch_size]
        neg_logits = z @ neg_o.t()  # [batch_size, n_samples]

        return pos_logits, neg_logits

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
        cross_entropy_loss = loss(pre_labels[mask], target_labels[mask])
        return cross_entropy_loss

    def get_linear_model(self):
        return self.linear