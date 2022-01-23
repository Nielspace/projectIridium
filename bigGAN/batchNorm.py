import torch
import torch.nn as nn 
import torch.nn.functional as F


def specNorm_conv2d(eps=1e-12, **kwargs):
    cnn = nn.Conv2d(**kwargs)
    o = nn.utils.spectral_norm(cnn, eps=eps)
    return o

def specNorm_linear(eps=1e-12, **kwargs):
    linear = nn.Linear(**kwargs)
    o = nn.utils.spectral_norm(linear, eps=eps)
    return o

def specNorm_embedding(eps=1e-12, **kwargs):
    emb = nn.Embedding(**kwargs)
    o = nn.utils.spectral_norm(emb, eps=eps)
    return o

class batchNorm(nn.Module):
    #https://arxiv.org/pdf/1707.00683.pdf=> Modulating early visual processing by language
    #https://arxiv.org/pdf/1502.03167.pdf => Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True):
        super(batchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional

        # We use pre-computed statistics for n_stats values of truncation between 0 and 1
        self.register_buffer('mean', torch.zeros(n_stats, num_features))
        self.register_buffer('variance', torch.ones(n_stats, num_features))
        self.step_size = 1.0 / (n_stats - 1)

        #conditional Norm 
        if conditional:
            assert condition_vector_dim is not None
            self.scale = specNorm_linear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)
            self.offset = specNorm_linear(in_features=condition_vector_dim, out_features=num_features, bias=False, eps=eps)    
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))

            

    def forward(self, x, truncation, condition_vector=None):
        # Retreive pre-computed statistics associated to this truncation
        coef, start_idx = math.modf(truncation / self.step_size)
        start_idx = int(start_idx)

        #For interpolation read : https://arxiv.org/pdf/1802.05957.pdf
        if coef != 0.0:  # Interpolate
            variance = self.variance[start_idx] * coef + self.variance[start_idx + 1] * (1 - coef)
        else:
            mean = self.mean[start_idx]
            variance = self.variance[start_idx]

        if self.conditional:
            mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            variance = variance.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            weight = 1 + self.scale(condition_vector).unsqueeze(-1).unsqueeze(-1)
            bias = self.offset(condition_vector).unsqueeze(-1).unsqueeze(-1)
            
            #BN(F_i_c_w_h|w_c, B_c) = (w_c * F_i_c_w_h - E_B[F_i_c_w_h])/sqrt(Var_b[F_i_c_w_h] + e) + B_c

            out = weight * (x - mean) / torch.sqrt(variance + self.eps) + bias
        else:
            out = F.batch_norm(x, mean, variance, self.weight, self.bias,
                               training=False, momentum=0.0, eps=self.eps)

        return out
        