


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# this class largely follows the official sonnet implementation
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks, bias=True, a = None):
        super(GroupLinearLayer, self).__init__()
        self.nb = num_blocks
        self.dout = dout

        if a is None:
            a = 1. / math.sqrt(dout * num_blocks)

        #gain = 1.0 / math.sqrt(2)
        #a = gain * math.sqrt(6.0 / (din + dout))

        self.weight = nn.Parameter(torch.FloatTensor(num_blocks,din,dout).uniform_(-a,a))

        self.bias = bias

        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(num_blocks,dout).uniform_(-a,a))
            #self.bias = nn.Parameter(torch.zeros(dout*num_blocks))
        else:
            self.bias = None

    def forward(self,x):

	#input: ts x bs x blocks*nhid
	#ts*bs , blocks, nhid
	#blocks, ts*bs, nhid
        ts,bs,m = x.shape

        x = x.reshape((ts*bs, self.nb, m//self.nb))
        x = x.permute(1,0,2)
        x = torch.bmm(x,self.weight)
        x = x.permute(1,0,2)

        if not self.bias is None:
            x = x + self.bias

        x = x.reshape((ts, bs, self.dout*self.nb))

        #if not self.bias is None:
        #    x += self.bias

        return x




class MemoryAttention(nn.Module):
    def __init__(self, n_blocks_query, n_blocks_val, dim_query, dim_val, n_heads=8):
        super(MemoryAttention, self).__init__()

        self.n_heads = n_heads
        self.n_blocks_val = n_blocks_val
        self.dim_val = dim_val
        self.block_dim_val = dim_val // self.n_blocks_val

        self.n_blocks_query = n_blocks_query
        self.dim_query = dim_query
        self.block_dim_query = dim_query // self.n_blocks_query

        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.query_net = GroupLinearLayer(self.block_dim_query, self.head_dim * self.n_heads, n_blocks_query, a=0.05)
        self.key_net = GroupLinearLayer(self.block_dim_val, self.head_dim * self.n_heads, n_blocks_val, a=0.05)
        self.value_net = GroupLinearLayer(self.block_dim_val, self.head_dim * self.n_heads, n_blocks_val, a=0.05)
        self.final = GroupLinearLayer(self.head_dim * self.n_heads, self.block_dim_query, n_blocks_query, a=0.05)

    def forward(self, q, kv):

        #comes in as: bs, pos*emb.
        #positions_attend x T*bs x emb


        #q = q.permute(1,0,2)
        #kv = kv.permute(1,0,2)

        #print('kv shape after permute', kv.shape)

        seq_len_q,bsz,_ = q.shape
        seq_len_v,bsz,_ = kv.shape

        q = q.reshape((seq_len_q, bsz, self.n_blocks_query * self.block_dim_query))

        kv = kv.reshape((seq_len_v, bsz, self.n_blocks_val * self.block_dim_val))

        q = self.query_net(q).view(seq_len_q, bsz, self.n_blocks_query, self.n_heads, self.head_dim)
        k = self.key_net(kv).view(seq_len_v, bsz, self.n_blocks_val, self.n_heads, self.head_dim)
        v = self.value_net(kv).view(seq_len_v, bsz, self.n_blocks_val, self.n_heads, self.head_dim)

        q = q.transpose(2,3) * self.scale
        k = k.transpose(2,3)
        v = v.transpose(2,3)
        score = torch.matmul(q, k.transpose(3,4))
        #print('score shape', score.shape)
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2,3)
        #print('out shape', out.shape)
        score = score.mean(dim=2)

        out = out.reshape(seq_len_q, bsz, self.n_blocks_query * self.head_dim * self.n_heads)
        out = self.final(out)
        out = out.view(seq_len_q, bsz, self.dim_query)


        return out, score

class RelationalMemory(nn.Module):
    """
    Constructs a `RelationalMemory` object.
    This class is same as the RMC from relational_rnn_models.py, but without language modeling-specific variables.
    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      input_size: The size of input per step. i.e. the dimension of each input vector
      num_heads: The number of attention heads to use. Defaults to 1.
      num_blocks: Number of times to compute attention per time step. Defaults
        to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.
      name: Name of the module.

      # NEW flag for this class
      return_all_outputs: Whether the model returns outputs for each step (like seq2seq) or only the final output.
    Raises:
      ValueError: gate_style not one of [None, 'memory', 'unit'].
      ValueError: num_blocks is < 1.
      ValueError: attention_mlp_layers is < 1.
    """

    def __init__(self, mem_slots, head_size, input_size, output_size, num_heads=1, num_blocks=1, forget_bias=1., input_bias=0.,
                 gate_style='unit', attention_mlp_layers=2, key_size=None, return_all_outputs=False):
        super(RelationalMemory, self).__init__()

        ########## generic parameters for RMC ##########
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads

        # a new fixed params needed for pytorch port of RMC
        # +1 is the concatenated input per time step : we do self-attention with the concatenated memory & input
        # so if the mem_slots = 1, this value is 2
        self.mem_slots_plus_input = self.mem_slots + 1

        if num_blocks < 1:
            raise ValueError('num_blocks must be >=1. Got: {}.'.format(num_blocks))
        self.num_blocks = num_blocks

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. got: '
                '{}.'.format(gate_style))
        self.gate_style = gate_style

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers

        self.key_size = key_size if key_size else self.head_size

        ########## parameters for multihead attention ##########
        # value_size is same as head_size
        self.value_size = self.head_size
        # total size for query-key-value
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads  # denoted as F

        # each head has qkv_sized linear projector
        # just using one big param is more efficient, rather than this line
        # self.qkv_projector = [nn.Parameter(torch.randn((self.qkv_size, self.qkv_size))) for _ in range(self.num_heads)]
        self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
        self.qkv_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.total_qkv_size])

        # used for attend_over_memory function
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.mem_size)] * self.attention_mlp_layers)
        self.attended_memory_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])
        self.attended_memory_layernorm2 = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])

        ########## parameters for initial embedded input projection ##########
        self.input_size = input_size
        self.input_projector = nn.Linear(self.input_size, self.mem_size)

        #self.output_projector = nn.Linear(self.output_size, self.input_size)

        ########## parameters for gating ##########
        self.num_gates = 2 * self.calculate_gate_size()
        self.input_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.memory_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))

        ########## number of outputs returned #####
        self.return_all_outputs = return_all_outputs

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        # needed for truncated BPTT, called at every batch forward pass
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def _init_state(self, x, trainable=False):
         """
         Creates the initial memory.
         We should ensure each row of the memory is initialized to be unique,
         so initialize the matrix to be the identity. We then pad or truncate
         as necessary so that init_state is of size
         (batch_size, self.mem_slots, self.mem_size).
         Args:
           batch_size: The size of the batch.
           trainable: Whether the initial state is trainable. This is always True.
         Returns:
           init_state: A truncated or padded matrix of size
             (batch_size, self.mem_slots, self.mem_size).
         """
         batch_size = x.shape[0]
         init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])

         # pad the matrix with zeros
         if self.mem_size > self.mem_slots:
             difference = self.mem_size - self.mem_slots
             pad = torch.zeros((batch_size, self.mem_slots, difference))
             init_state = torch.cat([init_state, pad], -1)

         # truncation. take the first 'self.mem_size' components
         elif self.mem_size < self.mem_slots:
             init_state = init_state[:, :, :self.mem_size]

         init_state.requires_grad = True
         return init_state.to(x.device)


    def initial_state(self, batch_size, trainable=False):
        """
        Creates the initial memory.
        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self.mem_slots, self.mem_size).
        Args:
          batch_size: The size of the batch.
          trainable: Whether the initial state is trainable. This is always True.
        Returns:
          init_state: A truncated or padded matrix of size
            (batch_size, self.mem_slots, self.mem_size).
        """
        init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])

        # pad the matrix with zeros
        if self.mem_size > self.mem_slots:
            difference = self.mem_size - self.mem_slots
            pad = torch.zeros((batch_size, self.mem_slots, difference))
            init_state = torch.cat([init_state, pad], -1)

        # truncation. take the first 'self.mem_size' components
        elif self.mem_size < self.mem_slots:
            init_state = init_state[:, :, :self.mem_size]

        return init_state

    def multihead_attention(self, memory):
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """

        # First, a simple linear projection is used to construct queries
        qkv = self.qkv_projector(memory)
        # apply layernorm for every dim except the batch dim
        qkv = self.qkv_layernorm(qkv)

        # mem_slots needs to be dynamically computed since mem_slots got concatenated with inputs
        # example: self.mem_slots=10 and seq_length is 3, and then mem_slots is 10 + 1 = 11 for each 3 step forward pass
        # this is the same as self.mem_slots_plus_input, but defined to keep the sonnet implementation code style
        mem_slots = memory.shape[1]  # denoted as N

        # split the qkv to multiple heads H
        # [B, N, F] => [B, N, H, F/H]
        qkv_reshape = qkv.view(qkv.shape[0], mem_slots, self.num_heads, self.qkv_size)

        # [B, N, H, F/H] => [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)

        # [B, H, N, key_size], [B, H, N, key_size], [B, H, N, value_size]
        q, k, v = torch.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)

        # scale q with d_k, the dimensionality of the key vectors
        q_use = q * (self.key_size ** -0.5)

        # make it [B, H, N, N]
        dot_product = torch.matmul(q_use, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)

        # output is [B, H, N, V]
        output = torch.matmul(weights, v)

        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        return new_memory


    @property
    def state_size(self):
        return [self.mem_slots, self.mem_size]

    @property
    def output_size(self):
        return self.mem_slots * self.mem_size

    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self.gate_style == None
            return 0

    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.

        # equation 8: since there is no output gate, h is just a tanh'ed m
        memory = torch.tanh(memory)

        # TODO: check this input flattening is correct
        # sonnet uses this, but i think it assumes time step of 1 for all cases
        # if inputs is (B, T, features) where T > 1, this gets incorrect
        # inputs = inputs.view(inputs.shape[0], -1)

        # fixed implementation
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    "input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1")
            inputs = inputs.view(inputs.shape[0], -1)
            # matmul for equation 4 and 5
            # there is no output gate, so equation 6 is not implemented
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        # this completes the equation 4 and 5
        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        # to be used for equation 7
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def attend_over_memory(self, memory):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multihead_attention(memory)

            # Add a skip connection to the multiheaded attention's input.
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # add a skip connection to the attention_mlp's input.
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)

        return memory

    def forward_step(self, inputs, memory, treat_input_as_matrix=False):
        """
        Forward step of the relational memory core.
        Args:
          inputs: Tensor input.
          memory: Memory output from the previous time step.
          treat_input_as_matrix: Optional, whether to treat `input` as a sequence
            of matrices. Default to False, in which case the input is flattened
            into a vector.
        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """

        if treat_input_as_matrix:
            # keep (Batch, Seq, ...) dim (0, 1), flatten starting from dim 2
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
            # apply linear layer for dim 2
            inputs_reshape = self.input_projector(inputs)
        else:
            # keep (Batch, ...) dim (0), flatten starting from dim 1
            inputs = inputs.view(inputs.shape[0], -1)
            # apply linear layer for dim 1
            inputs = self.input_projector(inputs)
            # unsqueeze the time step to dim 1
            inputs_reshape = inputs.unsqueeze(dim=1)


        memory_plus_input = torch.cat([memory, inputs_reshape], dim=1)
        next_memory = self.attend_over_memory(memory_plus_input)

        # cut out the concatenated input vectors from the original memory slots
        n = inputs_reshape.shape[1]
        next_memory = next_memory[:, :-n, :]

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            # these gates are sigmoid-applied ones for equation 7
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            # equation 7 calculation
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory

        output = next_memory.view(next_memory.shape[0], -1)

        return output, next_memory

    def forward(self, inputs, memory):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # memory = self.repackage_hidden(memory)

        # for loop implementation of (entire) recurrent forward pass of the model
        # inputs is batch first [batch, seq], and output logit per step is [batch, vocab]
        # so the concatenated logits are [seq * batch, vocab]

        # targets are flattened [seq, batch] => [seq * batch], so the dimension is correct

        logits = []
        # shape[1] is seq_lenth T

        for idx_step in range(inputs.shape[1]):
            logit, memory = self.forward_step(inputs[:, idx_step], memory)
            logits.append(logit)
        logits = torch.cat(logits)
        memory_out = self.output_projector(memory.view(memory.shape[0], -1))

        if self.return_all_outputs:
            return logits, memory_out + inputs.squeeze(1), memory
        else:
            return logit, memory_out + inputs.squeeze(1), memory



if __name__ == "__main__":
    #[2708, 16]
    input_size = 16
    seq_length = 2708
    batch_size = 1
    slots = 10
    #model = RelationalMemory(mem_slots=slots, head_size=40, input_size=input_size, output_size=input_size, num_heads=8, num_blocks=30, forget_bias=1., input_bias=0., gate_style='unit')
    #model_memory = model.initial_state(batch_size=seq_length)

    random_input = torch.randn((seq_length, input_size))
    #random_targets = torch.randn((32, seq_length, input_size))

    #print('random input shape', random_input.shape)
    #print('memory shape', model_memory.shape)

    #h, next_memory = model.forward_step(random_input, model_memory)#, random_targets, treat_input_as_matrix=True)

    #print('h shape', h.shape)
    #print('next mem', next_memory.shape)

    #next_memory: (bs, pos, nhid)

    #mem_read = next_memory.reshape((next_memory.shape[0], 1, next_memory.shape[1]*next_memory.shape[2]))

    #memory_attention = MemoryAttention(n_blocks_query=1, n_blocks_val=slots, dim_query=input_size, dim_val=3200)

    random_input = random_input.unsqueeze(1)

    print('random input shape', random_input.shape)
    print('mem read shape', mem_read.shape)

    
    out,_ = memory_attention(random_input, mem_read)

    out = out.squeeze(1)

    print('out shape', out.shape)

    #_, new_memory = self.memory_layer.forward_step(x_write.reshape((T*bsz, nhid)), self.memory_obj[0])
    #        self.memory_obj[0] = new_memory
    #        Tbs,num_slots,nhid_slot = new_memory.shape
    #        mem_read = new_memory.reshape((T, bsz, num_slots*nhid_slot))
    #        x,_ = self.memory_attention(x, mem_read)
    #        x = residual + x


class MemReadAndWrite(nn.Module):

    def __init__(self, slots=10, input_size=16):
        super(MemReadAndWrite, self).__init__()

        #self.model = RelationalMemory(mem_slots=slots, head_size=16, input_size=input_size, output_size=input_size, num_heads=4, num_blocks=1, forget_bias=1., input_bias=0., gate_style='unit')       
        #self.memory_attention = MemoryAttention(n_blocks_query=1, n_blocks_val=slots, dim_query=input_size, dim_val=640)
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)

        self.mha1 = nn.MultiheadAttention(input_size, 1)
        self.mha2 = nn.MultiheadAttention(input_size, 1)

        self.mha3 = nn.MultiheadAttention(input_size, 1)
        self.mha4 = nn.MultiheadAttention(input_size, 1)

        self.drop = nn.Dropout(0.1)

        self.init_query = nn.Parameter(1.0 * torch.randn(1,1,input_size)).cuda()

        self.mlp1 = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size))
        self.mlp2 = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Linear(input_size, input_size))


    def forward(self, x, aug): 

        x = x.unsqueeze(1)

        seq_length, _, input_size = x.shape

        residual = x*1.0

        num_slots = 16 #was using 64
        
        init_query = self.init_query.repeat(num_slots,1,1)

        mem,_ = self.mha1(init_query, x, x)
        out,_ = self.mha2(x, mem, mem)

        out = self.mlp1(out)

        out = self.ln1(residual+out)
        residual = out*1.0

        mem,_ = self.mha3(mem, out, out)
        out,_ = self.mha4(out, mem, mem)

        out = self.mlp2(out)

        #old
        #model_memory = self.model.initial_state(batch_size=seq_length).cuda()
        #h, next_memory = self.model.forward_step(x, model_memory)
        #mem_read = next_memory.reshape((next_memory.shape[0], 1, next_memory.shape[1]*next_memory.shape[2]))
        #x = x.unsqueeze(1)
        #out,_ = self.memory_attention(x, mem_read)

        out = self.ln2(residual + out)

        out = out.squeeze(1)

        return out



