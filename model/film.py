"""FiLM hypernetwork for constraint-token conditioning (this paper).

MLP(g) -> per-DAN-layer, per-stream (op / machine) feature-wise affine
parameters, applied to the layer INPUTS post-normalization, pre-attention:

    h  <-  (1 + dgamma) * h + beta        (masked to live nodes)

Identity start: the output head is zero-initialized, so dgamma = beta = 0 and
the network is EXACTLY the unconditioned backbone at init (and stays exactly
A3 when use_film is False — flag-gated, no code-path change).

Liveness masking: deleted nodes are exact zero vectors by the environment's
contract (they must stay out of the nonzero-averaging pooling); beta would
resurrect them, so the modulation is multiplied by the same zero-row
criterion used by the type-embedding path and nonzero_averaging.
"""
import torch
import torch.nn as nn


class FiLMHypernet(nn.Module):

    def __init__(self, token_dim, hidden_dim, j_dims, m_dims):
        """
        :param token_dim: dimension of the global constraint token g
        :param hidden_dim: hidden width of the 2-layer MLP (Appendix B: 64)
        :param j_dims: per-layer op-stream input dims (e.g. [20, 128])
        :param m_dims: per-layer machine-stream input dims (e.g. [18, 128])
        """
        super().__init__()
        self.j_dims = list(j_dims)
        self.m_dims = list(m_dims)
        self.n_layers = len(self.j_dims)
        out_dim = 2 * (sum(self.j_dims) + sum(self.m_dims))
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        # identity start
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, token):
        """token [sz_b, token_dim] -> list of (dg_j, b_j, dg_m, b_m) per DAN
        layer, each [sz_b, dim]."""
        out = self.net(token)
        params, ofs = [], 0
        for l in range(self.n_layers):
            dj, dm = self.j_dims[l], self.m_dims[l]
            dg_j = out[:, ofs:ofs + dj]; ofs += dj
            b_j = out[:, ofs:ofs + dj]; ofs += dj
            dg_m = out[:, ofs:ofs + dm]; ofs += dm
            b_m = out[:, ofs:ofs + dm]; ofs += dm
            params.append((dg_j, b_j, dg_m, b_m))
        return params


def film_modulate(h, dgamma, beta):
    """Apply (1+dgamma)*h + beta to live (nonzero) rows of h [sz_b, N, d]."""
    live = (h.abs().sum(dim=-1, keepdim=True) > 0).float()
    return ((1.0 + dgamma.unsqueeze(1)) * h + beta.unsqueeze(1)) * live
