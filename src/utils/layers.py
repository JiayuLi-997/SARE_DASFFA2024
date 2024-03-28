# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn


class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, kq_same=False, bias=True, attention_d=-1):
		super().__init__()
		"""
		It has projection layer for getting keys, queries and values. Followed by attention.
		"""
		self.d_model = d_model
		self.h = n_heads
		if attention_d<0:
			self.attention_d = self.d_model
		else:
			self.attention_d = attention_d

		self.d_k = self.attention_d // self.h
		self.kq_same = kq_same

		if not kq_same:
			self.q_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.k_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.v_linear = nn.Linear(d_model, self.attention_d, bias=bias)

	def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
		new_x_shape = x.size()[:-1] + (self.h, self.d_k)
		return x.view(*new_x_shape).transpose(-2, -3)

	def forward(self, q, k, v, mask=None):
		origin_shape = q.size()

		# perform linear operation and split into h heads
		if not self.kq_same:
			q = self.head_split(self.q_linear(q))
		else:
			q = self.head_split(self.k_linear(q))
		k = self.head_split(self.k_linear(k))
		v = self.head_split(self.v_linear(v))
		if len(mask.shape) < len(v.shape):
			mask = mask.unsqueeze(dim=1).repeat(1,self.h,1,1)

		# calculate attention using function we will define next
		output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

		# concatenate heads and put through final linear layer
		output = output.transpose(-2, -3).reshape(origin_shape)
		return output

	@staticmethod
	def scaled_dot_product_attention(q, k, v, d_k, mask=None):
		"""
		This is called by Multi-head attention object to find the values.
		"""
		scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -np.inf)
		scores = ((scores - scores.max())).softmax(dim=-1)
		scores = scores.masked_fill(torch.isnan(scores), 0)
		output = torch.matmul(scores, v)  # bs, head, q_len, d_k
		return output

class AttLayer(nn.Module):
	"""Calculate the attention signal(weight) according the input tensor.
	Reference: RecBole https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py#L236

	Args:
		infeatures (torch.FloatTensor): An input tensor with shape of[batch_size, XXX, embed_dim] with at least 3 dimensions.

	Returns:
		torch.FloatTensor: Attention weight of input. shape of [batch_size, XXX].
	"""

	def __init__(self, in_dim, att_dim):
		super(AttLayer, self).__init__()
		self.in_dim = in_dim
		self.att_dim = att_dim
		self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
		self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

	def forward(self, infeatures):
		att_signal = self.w(infeatures)  # [batch_size, XXX, att_dim]
		att_signal = fn.relu(att_signal)  # [batch_size, XXX, att_dim]

		att_signal = torch.mul(att_signal, self.h)  # [batch_size, XXX, att_dim]
		att_signal = torch.sum(att_signal, dim=-1)  # [batch_size, XXX]
		att_signal = fn.softmax(att_signal, dim=-1)  # [batch_size, XXX]

		return att_signal

class TransformerLayer(nn.Module):
	def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
		super().__init__()
		"""
		This is a Basic Block of Transformer. It contains one Multi-head attention object. 
		Followed by layer norm and position wise feedforward net and dropout layer.
		"""
		# Multi-Head Attention Block
		self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

		# Two layer norm layer and two dropout layer
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)

		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, seq, mask=None):
		context = self.masked_attn_head(seq, seq, seq, mask)
		context = self.layer_norm1(self.dropout1(context) + seq)
		output = self.linear1(context).relu()
		output = self.linear2(output)
		output = self.layer_norm2(self.dropout2(output) + context)
		return output


class TransformerMeantimeBlock(nn.Module):
	def __init__(self, args, La, Lr, item_feature_num=1):
		super().__init__()

		hidden = args.hidden_units * item_feature_num
		feed_forward_hidden = hidden * 4
		dropout = args.dropout
		self.attention = MixedAttention(args, La, Lr, item_feature_num=item_feature_num)
		self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
		self.input_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
		self.output_sublayer = SublayerConnection(args=args, size=hidden, dropout=dropout)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, mask, abs_kernel, rel_kernel, layer, info):
		# x : B x T x H
		# abs_kernel : La of [B x T x H]
		# rel_kernel : Lr of [B x T x T x H]
		x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask, abs_kernel, rel_kernel, layer, info))
		x = self.output_sublayer(x, self.feed_forward)
		return self.dropout(x)


class MixedAttention(nn.Module):
	def __init__(self, args, La, Lr, item_feature_num=1):
		super().__init__()
		d_model = args.hidden_units * item_feature_num
		dropout = args.dropout
		h = La + Lr  # num_heads
		self.La = La
		self.Lr = Lr
		self.d_k = d_model // h
		self.h = h
		self.scale = 1 / (self.d_k ** 0.5)
		## TODO
		self.content_linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
		self.abs_position_query_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(La)])
		self.abs_position_key_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(La)])
		self.rel_position_key_linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(Lr)])
		self.rel_position_bias = nn.Parameter(torch.FloatTensor(1, self.Lr, 1, self.d_k))
		## OUTPUT
		self.output_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(p=dropout)


	def forward(self, query, key, value, mask, abs_kernel, rel_kernel, layer, info):
		# q, k, v : B x T x H
		# abs_kernel : La of [B x T x H]
		# rel_kernel : Lr of [B x T x T x H]
		batch_size, T = query.size(0), query.size(1)

		# q, k, v, kernel_q, kernel_k : B x n x T x d
		query, key, value = \
			[l(x).view(batch_size, T, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.content_linear_layers, (query, key, value))]

		scores = torch.zeros(batch_size, self.h, T, T).to(query)
		if self.La > 0:
			Xq = query[:, :self.La]  # B x La x T x d
			Xk = key[:, :self.La]  # B x La x T x d
			Pq = torch.stack([l(x) for l, x in zip(self.abs_position_query_linear_layers, abs_kernel)], dim=1)  # B x La x T x d
			Pk = torch.stack([l(x) for l, x in zip(self.abs_position_key_linear_layers, abs_kernel)], dim=1)  # B x La x T x d
			abs_scores = torch.einsum('blid,bljd->blij', Xq + Pq, Xk + Pk)  # B x La x T x T
			scores[:, :self.La] += abs_scores

		if self.Lr > 0:
			Xq = query[:, self.La:]  # B x Lr x T x d
			Xk = key[:, self.La:]  # B x Lr x T x d
			R = torch.stack([l(x) for l, x in zip(self.rel_position_key_linear_layers, rel_kernel)], dim=1)  # B x Lr x T x T x d
			# rel_scores = torch.einsum('blid,bljd->blij', Xq + self.content_bias, Xk)  # B x Lr x T x T
			rel_scores = torch.einsum('blid,bljd->blij', Xq, Xk)  # B x Lr x T x T
			rel_scores += torch.einsum('blid,blijd->blij', Xq + self.rel_position_bias, R)  # B x Lr x T x T
			scores[:, self.La:] += rel_scores

		scores = scores * self.scale
		scores = scores.masked_fill(mask == 0, -1e9)

		p_attn = fn.softmax(scores, dim=-1)  # B x n x T x T
		p_attn = self.dropout(p_attn)
		x = torch.matmul(p_attn, value)  # B x n x T x d

		if info is not None:
			info['attn_{}'.format(layer)] = p_attn

		x = x.transpose(1, 2).contiguous().view(batch_size, T, self.h * self.d_k)  # B x T x H

		x = self.output_linear(x)  #  B x T x H
		return x

class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1, act='gelu', middle_drop=True):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
		if act == 'gelu':
			self.activation = GELU()
		elif act == 'relu':
			self.activation = nn.ReLU()
		else:
			raise ValueError
		self.middle_drop = middle_drop

	def forward(self, x):
		# return self.w_2(self.dropout(self.activation(self.w_1(x))))
		if self.middle_drop:
			return self.w_2(self.dropout(self.activation(self.w_1(x))))
		else:
			return self.w_2(self.activation(self.w_1(x)))

import math
class GELU(nn.Module):
	"""
	Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
	"""

	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def SublayerConnection(args, size, dropout):
	if args.residual_ln_type == 'pre':
		return SublayerConnectionPreLN(size, dropout)
	elif args.residual_ln_type == 'post':
		return SublayerConnectionPostLN(size, dropout)
	else:
		raise ValueError


class SublayerConnectionPreLN(nn.Module):
	def __init__(self, size, dropout):
		super().__init__()
		self.norm = nn.LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		# return self.norm(x + self.dropout(sublayer(x)))
		sub_output = sublayer(self.norm(x))
		if isinstance(sub_output, tuple):
			sub_output, rest = sub_output[0], sub_output[1:]
			output = x + self.dropout(sub_output)
			return (output, *rest)
		else:
			return x + self.dropout(sub_output)


class SublayerConnectionPostLN(SublayerConnectionPreLN):
	def forward(self, x, sublayer):
		sub_output = sublayer(x)
		if isinstance(sub_output, tuple):
			sub_output, rest = sub_output[0], sub_output[1:]
			output = x + self.dropout(sub_output)
			output = self.norm(output)
			return (output, *rest)
		else:
			return self.norm(x + self.dropout(sub_output))