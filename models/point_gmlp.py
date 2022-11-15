import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from random import randrange
from blocks import Residual, PreNorm, PostNorm, gMLPBlock

def uniform_sampling(n: int, D: int, E: int) -> nn.Module:
    """Uniform sampling embedding, that creates chunks/patches of points.

    Args:
        n (int): Size of points to make a patch.
        D (int): Dimensions of points.
        E (int): Embedding dim.

    Returns:
        nn.Module: Embedding layer.
    """
    return nn.Sequential(
        Rearrange("B T (k n) D -> B (T k) (n D)", n=n),
        nn.Linear(n * D, E)
    )

def tubelet_sampling(t: int, n: int, d: int, E: int) -> nn.Module:
    """Tubelet sampling embedding, that creates tubelets from the input volume."""
    return nn.Sequential(
        Rearrange("B (j t) (k n) (l d) -> B (j k l) (t n d)", t=t, n=n, d=d),
        nn.Linear(t * n * d, E)
    )

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class PointGMLP(nn.Module):
    def __init__(
        self,
        input_size: list = [32, 256, 8], # (T, N, D)
        tube_size: list = [16, 8, 4],    # (t, n, d)
        dim: int = 64,                   # model embedding dim, E
        depth: int = 12,                 # how many consecutive blocks in the model
        ff_mult: int = 4,                # embedding projection factor
        prob_survival: float = 0.9,      # probability that a block won't be dropped at training time
        pre_norm: bool = False,          # pre-norm or post-norm
        num_classes: int = 14            # number of classes to predict
    ):
        super().__init__()

        P_Norm = PreNorm if pre_norm else PostNorm
        
        T, N, D = input_size
        t, n, d = tube_size
        num_tubes = int((T * N * D) / (t * n * d))
        
        dim_ff = dim * ff_mult
        
        self.prob_survival = prob_survival

        self.to_patch_embedding = tubelet_sampling(t, n, d, dim)
        
        self.layers = nn.ModuleList(
            [Residual(P_Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=num_tubes))) for i in range(depth)]
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, X):
        X = self.to_patch_embedding(X)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        X = nn.Sequential(*layers)(X)
        return self.to_logits(X)
        
