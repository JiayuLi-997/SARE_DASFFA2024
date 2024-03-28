import torch
import torch.nn as nn
import numpy as np

''' Adding SARE module to SASRec
Reference: 
 	Self-attentive sequential recommendation.
	Kang, Wang-Cheng, and Julian McAuley. 
  	2018 IEEE international conference on data mining (ICDM). IEEE, 2018.
'''

from models.BaseModel import ImpressionContextSeqModel
from models.SARE.BaseSARE import *
from utils import layers


class SASRec_SARE(ImpressionContextSeqModel,BaseSARE):
	reader = 'ImpressionContextSeqReader'
	runner = 'ImpressionSituRunner'
	extra_log_args = ['num_layers','prob_weights','prob_loss_n','situ_weights','situ_lr','situ_l2','fix_rec']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.')
		parser.add_argument('--num_heads', type=int, default=4,
							help='Number of attention heads.')
		parser.add_argument('--fix_rec',type=int,default=0,help='Whether fix the recommender side.')
		parser.add_argument('--rec_path',type=str,default='model.pt')
		parser = BaseSARE.parse_model_args(parser)
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.emb_size = args.emb_size
		self.vec_size = args.emb_size
		self.max_his = args.history_max
		self.num_layers = args.num_layers
		self.num_heads = args.num_heads
		self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
		self.get_generalSARE_init(args, corpus)
		self.item_feature_num, self.user_feature_num = 1, 1
		self._define_params()
		self.apply(self.init_weights)
		self.fix_rec = args.fix_rec
		if self.fix_rec:
			self.load_rec_params(args.rec_path)
			self.fix_recommender_params()

	def _define_params(self):
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
		self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

		self.transformer_block = nn.ModuleList([
			layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
									dropout=self.dropout, kq_same=False)
			for _ in range(self.num_layers)
		])
		self.get_generalSARE_params()
	
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		situ_p, situ_bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'situ' in name or 'la_weights' in name:
				if 'bias' in name:
					situ_bias_p.append(p)
				else:
					situ_p.append(p)
			elif 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params':situ_p,'lr':self.situ_lr,'weight_decay':self.situ_l2},
				   {'params':situ_bias_p,'lr':self.situ_lr,'weight_decay':0},
				   {'params': weight_p,}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def forward(self, feed_dict):
		self.check_list = []
		i_ids = feed_dict['item_id']  # [batch_size, -1]
		history = feed_dict['history_items']  # [batch_size, history_max]
		lengths = feed_dict['lengths']  # [batch_size]
		batch_size, seq_len = history.shape

		valid_his = (history > 0).long()
		his_vectors = self.i_embeddings(history)

		position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
		pos_vectors = self.p_embeddings(position)
		his_vectors = his_vectors + pos_vectors

		# Self-attention
		causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
		attn_mask = torch.from_numpy(causality_mask).to(self.device)
		for block in self.transformer_block:
			his_vectors = block(his_vectors, attn_mask)
		his_vectors = his_vectors * valid_his[:, :, None].float()

		his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :] # B * 1 * emb
		# his_vector = his_vectors.sum(1) / lengths[:, None].float()
		# ↑ average pooling is shown to be more effective than the most recent embedding

		i_vectors = self.i_embeddings(i_ids)
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)

		situ_u_vectors = self.situ_u_transform(his_vector.squeeze())
		if len(situ_u_vectors.shape)==1:
			situ_u_vectors = situ_u_vectors.unsqueeze(0)
		situ_i_vectors = self.situ_i_transform(i_vectors)
		situ_predictions, pred_situ, situ_embed = self.situation_predict(situ_u_vectors, situ_i_vectors,[feed_dict[situ] for situ in self.situ_feature_cnts])

		out_dict = {'prediction': prediction, 'situ_prediction':situ_predictions,
		}
		return out_dict
	
	def loss(self, out_dict, target):
		return self.SARE_loss(out_dict,target)

	def fix_recommender_params(self):
		for name, p in self.named_parameters():
			if 'situ' in name or 'la_weights' in name:
				continue
			p.requires_grad=False

	def load_rec_params(self, model_path):
		base_model = torch.load(model_path)
		filtered_params = {k:v for k,v in base_model.items() if k in self.state_dict()}
		self.load_state_dict(filtered_params, strict=False)
		return
