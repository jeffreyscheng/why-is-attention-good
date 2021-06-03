import torch
import math


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, with_attention=False):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.with_attention = with_attention
        if with_attention:
            self.attn = SoftmaxAttention(input_dim, output_dim)

    def forward(self, x):
        if self.with_attention:
            attn_weights = self.attn(x)
        else:
            attn_weights = 1
        return self.linear(x) * attn_weights


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, with_attention1=False, with_attention2=False):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.with_attention1 = with_attention1
        self.with_attention2 = with_attention2
        if with_attention1:
            self.attn1 = SoftmaxAttention(input_dim, hidden_dim)
        if with_attention2:
            self.attn2 = SoftmaxAttention(hidden_dim, output_dim)

    def forward(self, x):
        if self.with_attention1:
            attn_weights1 = self.attn1(x)
        else:
            attn_weights1 = 1
        impulse = self.fc1(x) * attn_weights1
        if self.with_attention2:
            attn_weights2 = self.attn2(impulse)
        else:
            attn_weights2 = 1
        impulse = self.fc2(impulse) * attn_weights2
        return impulse


class SoftmaxAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # we are using query, key, value notation
        self.Q = torch.nn.Linear(input_dim, output_dim)
        self.K = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)  # assuming inputs of size (BxH)

    def forward(self, x):
        # Let Qx = q, Kx = k
        q = self.Q(x)
        k = self.K(x)
        # computes Softmax(q \cdot k^T)
        # we ignore the temperature parameter sqrt(d_k) from the transformer
        return self.softmax(q * k * 1 / math.sqrt(self.input_dim))


class AttendedLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # we are using query, key, value notation
        self.V = torch.nn.Linear(input_dim, output_dim)
        self.attn = SoftmaxAttention(input_dim, output_dim)

    def forward(self, x):
        return self.V(x) * self.attn(x)


class ConstantWidthDeepNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim, with_attention=None):
        super().__init__()

        if with_attention is None:
            with_attention = [False] * depth

        self.depth = depth
        self.with_attention = with_attention

        self.layers = torch.nn.ModuleList()
        for i in range(depth):
            if i == 0:
                a = input_dim
                b = hidden_dim
            elif i == depth - 1:
                a = hidden_dim
                b = output_dim
            else:
                a = hidden_dim
                b = hidden_dim
            if with_attention[i]:
                self.layers.append(AttendedLayer(a, b))
            else:
                self.layers.append(torch.nn.Linear(a, b))

    def fetch_value_weights(self, layer):
        if self.with_attention[layer]:
            return self.layers[layer].V.weight
        else:
            return self.layers[layer].weight

    def forward(self, x, with_activations=True):
        impulse = x
        activations = []
        for layer in self.layers:
            impulse = layer(impulse)
            activations.append(impulse)
        if with_activations:
            return impulse, activations
        else:
            del activations
            return impulse
