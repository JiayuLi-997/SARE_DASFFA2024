import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

""" Add SARE module to LightGCN
Reference:
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    He et al., SIGIR'2020.
"""

from models.BaseModel import ImpressionModel
from models.SARE.BaseSARE import *
from models.CF_SARE.LightGCN import LGCNEncoder

class LightGCN_SARE(ImpressionModel, BaseSARE):
	reader = 'ImpressionContextReader'
	runner = 'ImpressionSituRunner'
	extra_log_args = ['prob_weights','prob_loss_n','situ_weights','situ_lr','situ_l2','fix_rec']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		parser.add_argument('--fix_rec',type=int,default=0,help='Whether fix the recommender side.')
		parser.add_argument('--rec_path',type=str,default='model.pkl')
		parser = BaseSARE.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self,args, corpus)
		self.emb_size = args.emb_size
		self.vec_size = args.emb_size
		self.n_layers = args.n_layers
		self.fix_rec = args.fix_rec
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self.get_generalSARE_init(args, corpus)
		self.item_feature_num, self.user_feature_num = 1, 1
		self._define_params()
		self.apply(self.init_weights)
		if self.fix_rec:
			self.load_rec_params(args.rec_path)
			self.fix_recommender_params()
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()
	
	def _define_params(self):
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
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
 
	def forward(self, feed_dict):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)

		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
		
		situ_u_vectors = self.situ_u_transform(u_embed)
		situ_i_vectors = self.situ_i_transform(i_embed)
		situ_predictions, pred_situ, situ_embed = self.situation_predict(situ_u_vectors, situ_i_vectors, [feed_dict[situ] for situ in self.situ_feature_cnts])
		out_dict = {'prediction': prediction, 'situ_prediction':situ_predictions,
		}
		return out_dict
	
	def loss(self, out_dict, target):
		return self.SARE_loss(out_dict,target)

	class Dataset(ImpressionModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			for feature in self.data.keys():
				if feature[:2] == 'c_':
					feed_dict[feature] = self.data[feature][index]
			return feed_dict

