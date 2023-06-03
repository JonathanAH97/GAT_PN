import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from torch.nn import BatchNorm1d, LayerNorm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Standard Attention Module
class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden)
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref

#LSTM module for the first city
class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)  
        self.Whi = nn.Linear(n_hidden, n_hidden)  
        self.wci = nn.Linear(n_hidden, n_hidden)  
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)  
        self.Whf = nn.Linear(n_hidden, n_hidden)   
        self.wcf = nn.Linear(n_hidden, n_hidden)   
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    
        self.Whc = nn.Linear(n_hidden, n_hidden)   
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)   
        self.Who = nn.Linear(n_hidden, n_hidden)  
        self.wco = nn.Linear(n_hidden, n_hidden)    
    
    
    def forward(self, x, h, c):      
        
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c

# GAT module
class GraphEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = 2
        self.dropout = 0.5
        self.conv1 = GATConv(hidden_dim, hidden_dim // self.num_heads, heads = self.num_heads, dropout = self.dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // self.num_heads, heads = self.num_heads, dropout = self.dropout)
        self.r1 = nn.Parameter(torch.ones(1))
        self.r2 = nn.Parameter(torch.ones(1))
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, context, city_size):
        batch_size, n_nodes, hidden_dim = context.size()
        context = context.view(-1, self.hidden_dim) 
        adj_matrix = torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)
        edge_index, _ = dense_to_sparse(adj_matrix)
        edge_index = edge_index.view(2, -1)

        x1 = self.r1*self.W1(context) + (1 - self.r1)* F.relu(self.conv1(context, edge_index))

        # Residual connection
        x1 = x1 + context

        x2 = self.r2*self.W2(x1) + (1 - self.r2)* F.relu(self.conv2(x1, edge_index))

        # Residual connection
        x2 = x2 + x1

        return x2




class GAT_PN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, tanh_exploration = 10, vector_context = False):
        super(GAT_PN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = hidden_dim
        self.tanh_exploration = tanh_exploration
        self.vector_context = vector_context
        
        self.lstm0 = nn.LSTM(hidden_dim, hidden_dim)
        self.pointer = Attention(hidden_dim)
        self.encoder = LSTM(hidden_dim)
        
        h0 = torch.FloatTensor(hidden_dim)
        c0 = torch.FloatTensor(hidden_dim)
        
        alpha = torch.ones(1)
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(hidden_dim), 1/math.sqrt(hidden_dim))
        self.c0.data.uniform_(-1/math.sqrt(hidden_dim), 1/math.sqrt(hidden_dim))
        
        # Point embedding
        self.embedding_x = nn.Linear(input_dim, hidden_dim)

        # Embedding for all cities
        self.embedding_all = nn.Linear(input_dim, hidden_dim)
        self.gnn = GraphEncoding(hidden_dim)

        
    
    
    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)
        
        # =============================
        # Graph context
        # =============================
        
        # This is vector context
        if self.vector_context:
            x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1) 
            X_all = X_all - x_expand

        # Node embedding
        x = self.embedding_x(x)

        # Initial graph embedding
        context = self.embedding_all(X_all)
        
        # =============================
        # Process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
        
        
        context = self.gnn(context, self.city_size)

        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # Query vector
        q = h
        
        # Pointer based on the query and context embedding
        u, _ = self.pointer(q, context)
        
        latent_u = u.clone()
        
        u = self.tanh_exploration * torch.tanh(u) + mask
        
        if latent is not None:
            u += self.alpha * latent
    
        return F.softmax(u, dim=1), h, c, latent_u


# Generic Attention, Encoder and LSTM modules for critic network from https://github.com/wouterkool/attention-learn-to-route

class AttentionCritic(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=True, C=10):
        super(AttentionCritic, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits

class EncoderCritic(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""
    def __init__(self, input_dim, hidden_dim):
        super(EncoderCritic, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden
    
    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx

class CriticNetworkLSTM(nn.Module):
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters = 3,
            tanh_exploration = 10,
            use_tanh = True):
        super(CriticNetworkLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = EncoderCritic(embedding_dim, hidden_dim)
        
        self.process_block = AttentionCritic(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()

        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out.squeeze()