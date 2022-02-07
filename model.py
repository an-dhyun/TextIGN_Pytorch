import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim, steps=2): # 200 -> 96
        super(GraphLayer, self).__init__()

        self.steps = steps

        self.encode = nn.Linear(input_dim, output_dim, bias=False)

        self.z0 = nn.Linear(output_dim, output_dim, bias=True)
        self.z1 = nn.Linear(output_dim, output_dim, bias=True)

        self.r0 = nn.Linear(output_dim, output_dim, bias=True)
        self.r1 = nn.Linear(output_dim, output_dim, bias=True)

        self.h0 = nn.Linear(output_dim, output_dim, bias=True)
        self.h1 = nn.Linear(output_dim, output_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.encode.weight)
        torch.nn.init.xavier_uniform_(self.z0.weight)
        torch.nn.init.xavier_uniform_(self.z1.weight)
        torch.nn.init.xavier_uniform_(self.r0.weight)
        torch.nn.init.xavier_uniform_(self.r1.weight)
        torch.nn.init.xavier_uniform_(self.h0.weight)
        torch.nn.init.xavier_uniform_(self.h1.weight)

    def forward(self, inputs, adj_matrix, mask):
        # TODO: Add Dropout from line 219 of layers (Original Code)

        x = self.encode(inputs) # -> 58 * 96
        x = mask * F.relu(x) # 58 * 96

        for _ in range(self.steps):
            # TODO Dropout : L56 layers.py
            a = torch.matmul(adj_matrix, x) # -> 58 * 96
            # update gate
            z0 = self.z0(a)
            z1 = self.z1(x)
            z = torch.sigmoid(z0 + z1) # 논문 식 (2)
            # reset gate
            r0 = self.r0(a)
            r1 = self.r1(x)
            r = torch.sigmoid(r0 + r1) # 논문 식 (3)
            # update embeddings
            h0 = self.h0(a)
            h1 = self.h1(x * r)
            h = F.relu(mask * (h0 + h1)) # 논문 식 (4)
            # Update x for next iteration
            x = h * z + x * (1 - z) # 논문 식 (5)

        return x # -> graph


class ReadoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim): # 96 -> 2
        super(ReadoutLayer, self).__init__()

        self.att = nn.Linear(input_dim, 1, bias=True)

        self.emb = nn.Linear(input_dim, input_dim, bias=True)
        self.mlp = nn.Linear(input_dim, output_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.att.weight)
        torch.nn.init.xavier_uniform_(self.emb.weight)
        torch.nn.init.xavier_uniform_(self.mlp.weight)

    def forward(self, inputs, mask):
        x = inputs
        att = torch.sigmoid(self.att(x)) # 식 (6) 앞부분
        emb = torch.relu(self.emb(x)) # 식 (6) 뒷부분
        n = torch.sum(mask, dim=1)
        m = (mask - 1) * 1e9

        # Graph Summation
        g = mask * att * emb
        g = (torch.sum(g, dim=1) / n) + torch.max(g + m, dim=1).values

        output = self.mlp(g)
        return output # -> output


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.graph = GraphLayer(input_dim=input_dim, output_dim=hidden_dim)
        self.readout = ReadoutLayer(input_dim=hidden_dim, output_dim=output_dim)

    def forward(self, inputs, adj_matrix, mask):
        graph = self.graph(inputs, adj_matrix, mask)
        output = self.readout(graph, mask)
        return output
