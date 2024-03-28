# -*- coding: UTF-8 -*-

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from helpers.BaseReader import BaseReader


class BaseModel(nn.Module):
	reader, runner = None, None  # choose helpers in specific model classes
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		parser.add_argument('--buffer', type=int, default=1,
							help='Whether to buffer feed dicts for dev/test')
		return parser

	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def __init__(self, args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.device = args.device
		self.model_path = args.model_path
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()  # observe tensors in check_list every check_epoch

	"""
	Key Methods
	"""
	def _define_params(self):
		pass

	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		pass

	def loss(self, out_dict: dict) -> torch.Tensor:
		pass

	"""
	Auxiliary Methods
	"""
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path)
		torch.save(self.state_dict(), model_path)
		# logging.info('Save model to ' + model_path[:50] + '...')

	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	def actions_after_train(self):  # e.g., save selected parameters
		pass

	"""
	Define Dataset Class
	"""
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			#self.data = utils.df_to_dict(corpus.data_df[phase])#this raise the VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences warning
			self.data = corpus.data_df[phase].to_dict('list')
			# ↑ DataFrame is not compatible with multi-thread operations

		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			pass

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=np.object)
					else:
						stack_val = np.array([d[key] for d in feed_dicts])
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
					feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
				else:
					feed_dict[key] = torch.from_numpy(stack_val)
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict

class GeneralModel(BaseModel):
	reader, runner = 'BaseReader', 'BaseRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--num_neg', type=int, default=1,
							help='The number of negative items during training.')
		parser.add_argument('--dropout', type=float, default=0,
							help='Dropout probability for each deep layer')
		parser.add_argument('--test_all', type=int, default=0,
							help='Whether testing on all the items.')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all

	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
		# neg_pred = (neg_pred * neg_softmax).sum(dim=1)
		# loss = F.softplus(-(pos_pred - neg_pred)).mean()
		# ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
		return loss

	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			if (self.phase != 'train' or self.status=='test') and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			else:
				if self.phase != 'train' or self.status == 'train':
					neg_items = self.data['neg_items'][index]
				else:
					neg_items = self.sample_negative(user_id)
			item_ids = np.concatenate([[target_item], neg_items]).astype(int)
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
			return feed_dict

		# Sample negative items for all the instances
		def actions_before_epoch(self):
			neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
			for i, u in enumerate(self.data['user_id']):
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				for j in range(self.model.num_neg):
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			# if 'neg_items' not in self.data:
			self.data['neg_items'] = neg_items

		def sample_negative(self, u):
			neg_items = np.random.randint(1, self.corpus.n_items, size=99)
			clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
			# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
			for j in range(self.model.num_neg):
				while neg_items[j] in clicked_set:
					neg_items[j] = np.random.randint(1, self.corpus.n_items)
			return neg_items

class SequentialModel(GeneralModel):
	reader = 'SeqReader'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			return feed_dict

class ContextModel(GeneralModel):
	reader = 'ContextReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BPR',
							help='Type of loss functions.')
		parser.add_argument('--include_id',type=int,default=1,
							help='Whether add ids in context information.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		self.include_id = args.include_id
	
	def loss(self, out_dict: dict):
		"""
		utilize BPR loss (as general model) or log loss as most context-aware models
		"""
		if self.loss_n == 'BPR':
			return super().loss(out_dict)
		elif self.loss_n == 'BCE':
			predictions = out_dict['prediction'].sigmoid()
			pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
			loss = - (pos_pred.log() + (1-neg_pred).log().sum(dim=1)).mean()
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		return loss
	
	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			self.include_id = model.include_id

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			All_context = [] # concatenate all context information
			if len(self.corpus.user_feature_names):
				user_features = np.array([self.corpus.user_features[feed_dict['user_id']][c] 
							for c in self.corpus.user_feature_names]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				All_context = user_features
			if len(self.corpus.item_feature_names):
				items_features = np.array([[self.corpus.item_features[iid][c] for c in self.corpus.item_feature_names] 
											for iid in feed_dict['item_id'] ])
				All_context = np.concatenate([All_context, items_features],axis=-1) if len(All_context) else items_features
			if len(self.corpus.context_feature_names):
				context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1).repeat(
							feed_dict['item_id'].shape[0],axis=0)
				All_context = np.concatenate([All_context, context_features], axis=-1) if len(All_context) else context_features
			id_names = []
			if self.include_id:
				user_id = np.array([feed_dict['user_id']]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				item_ids = feed_dict['item_id'].reshape(-1,1)
				All_context = np.concatenate([All_context, user_id, item_ids],axis=-1) if len(All_context) else np.concatenate([user_id,item_ids],axis=-1)
				id_names = ['user_id','item_id']
			# transfer into multi-hot embedding
			base = 0
			for i, feature in enumerate(self.corpus.user_feature_names + self.corpus.item_feature_names + self.corpus.context_feature_names + id_names):
				All_context[:,i] += base
				base += self.corpus.feature_max[feature]
			feed_dict['context_mh'] = All_context # item num * feature num (will repeat 'item num' times for user and context features)
			return feed_dict

class ContextSeqModel(ContextModel):
	reader='ContextSeqReader'
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return ContextModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(SequentialModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]
			self.include_id = model.include_id
		
		def _convert_multihot(self, fname_list, data):
			base = 0
			for i, feature in enumerate(fname_list):
				data[:,i] += base
				base += self.corpus.feature_max[feature]
			return data

		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			if len(self.corpus.user_feature_names):
				user_fnames = self.corpus.user_feature_names
				user_features = [self.corpus.user_features[feed_dict['user_id']][c] for c in self.corpus.user_feature_names] 
				if self.include_id:
					user_fnames = ['user_id'] + user_fnames
					user_features = [feed_dict['user_id']] + user_features
				feed_dict['user_features'] = self._convert_multihot(user_fnames, np.array(user_features).reshape(1,-1))
			elif self.include_id:
				feed_dict['user_features'] = feed_dict['user_id'].reshape(-1,1)
			
			if len(self.corpus.item_feature_names):
				item_fnames = self.corpus.item_feature_names
				items_features = np.array([[self.corpus.item_features[iid][c] for c in self.corpus.item_feature_names] for iid in feed_dict['item_id'] ])
				items_features_history = np.array([[self.corpus.item_features[iid][c] for c in self.corpus.item_feature_names] for iid in feed_dict['history_items'] ])
				if self.include_id:
					item_fnames = ['item_id']+item_fnames
					item_ids = feed_dict['item_id'].reshape(-1,1)
					items_features = np.concatenate([item_ids, items_features],axis=1)
					his_item_ids = feed_dict['history_items'].reshape(-1,1)
					items_features_history = np.concatenate([his_item_ids,items_features_history],axis=1)
				feed_dict['item_features'] = self._convert_multihot(item_fnames, np.array(items_features))
				feed_dict['history_item_features'] = self._convert_multihot(item_fnames,np.array(items_features_history))
			elif self.include_id:
				feed_dict['item_features'] = feed_dict['item_id'].reshape(-1,1)
				feed_dict['history_item_features'] = feed_dict['history_items'].reshape(-1,1)

			if len(self.corpus.context_feature_names):
				context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1).repeat(
							feed_dict['item_id'].shape[0],axis=0)
				feed_dict['context_features'] = self._convert_multihot(self.corpus.context_feature_names, context_features)
			
			return feed_dict

class ImpressionModel(GeneralModel):
	reader='ImpressionReader'
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n', type=str, default='BPRsession',
							help='loss name,BPR or softmaxCE or pointwiseCE or BPR_hard or listnet or attention_rank')
		parser.add_argument('--train_max_pos_item', type=int, default=50,
						help='max positive item sample for test')
		parser.add_argument('--test_max_pos_item', type=int, default=50,
						help='max positive item sample for test')
		parser.add_argument('--test_max_neg_item', type=int, default=100,
						help='max negative item sample')
		parser.add_argument('--train_max_neg_item', type=int, default=100,
						help='max negative item sample')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args,corpus)
		self.loss_n = args.loss_n
		self.train_max_pos_item=args.train_max_pos_item
		self.test_max_pos_item=args.test_max_pos_item
		self.test_max_neg_item=args.test_max_neg_item
		self.train_max_neg_item=args.train_max_neg_item

	def loss(self, out_dict: dict, target=None):
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		prediction = out_dict['prediction']
		batch_size = prediction.shape[0]
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
		test_have_neg = mask[:,self.train_max_pos_item]#if no neg 0, has neg 1

		if self.loss_n == 'BPRsoftall': # hanyu
			prediction=torch.where(mask==1,prediction,-torch.ones_like(prediction)*100000)
			pos_pred, neg_pred = prediction[:, :self.train_max_pos_item], prediction[:,self.train_max_pos_item:]
			pos_softmax = (pos_pred - pos_pred.max()).softmax(dim=1)
			pos_pred = (pos_pred * pos_softmax).sum(dim=1)
			neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
			neg_pred = (neg_pred * neg_softmax).sum(dim=1)
			loss = F.softplus(-(pos_pred - neg_pred))
			#loss = -(pos_pred - neg_pred).sigmoid().log()

			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss

		elif 'BPR' in self.loss_n:
			valid_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-1).transpose(-1,-2)
			pos_mask = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) < self.train_max_pos_item).to(self.device)
			neg_mask = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) >= self.train_max_pos_item).to(self.device)
			select_mask = pos_mask.unsqueeze(dim=-1) * neg_mask.unsqueeze(dim=-1).transpose(-1,-2) * valid_mask # get all valid mask in the two-dimensional matrix
			score_diff = prediction.unsqueeze(dim=-1) - prediction.unsqueeze(dim=-1).transpose(-1,-2) # batch * impression list * impression list
			score_diff_mask = score_diff * select_mask
			
			neg_pred=torch.where(neg_mask*mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
			neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
			if 'hard' in self.loss_n: # Higher weights for lower-score positive items
				pos_pred=torch.where(pos_mask*mask==1,prediction,torch.tensor(float("Inf")).float().to(self.device))
				pos_softmax = (pos_pred.min() - pos_pred).softmax(dim=1)
			else:
				pos_pred=torch.where(pos_mask*mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
				pos_softmax = (pos_pred - pos_pred.max()).softmax(dim=1)

			if 'pair' in self.loss_n: # reweight after log-softmax
				loss = ((F.softplus(-score_diff_mask)*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1)
				loss = loss.mean()
			elif 'session' in self.loss_n: # reweight between log and softmax
				loss = -((score_diff_mask.sigmoid()*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1).log()
				loss = loss.mean()
			elif 'simple' in self.loss_n: # directly get the BPR of each pair
				loss = ((F.softplus(-score_diff_mask)*select_mask).sum(dim=-1)).sum(dim=-1)
				loss = loss.mean()
			else: # reweight within log-softmax
				loss = F.softplus(-(score_diff_mask*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1)
				loss = loss.mean()
			return loss

		elif self.loss_n=='listnet': #assume that the click probability is softmax(label) for pos and neg items
			target=torch.where(target!=-1,target.float(),-torch.tensor(float("Inf")).float().to(self.device))

			target_softmax = (target-target.max()).softmax(dim=1)
			prediction_softmax = (prediction-prediction.max()).softmax(dim=1)
			prediction_softmax=torch.where(mask==1,prediction_softmax,torch.ones_like(prediction_softmax)) #make sure that paddings are 1, so that after log they are 0

			loss = -( target_softmax * prediction_softmax.log() ).sum(dim=1)
			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss

		elif self.loss_n=='softmaxCE':#assume that the click probility is 1/k for k pos items, and zero for neg items
			pos_mask=torch.where(target==1,target,torch.zeros_like(target))
			pos_length=pos_mask.sum(axis=1)
			prediction=torch.where(mask==1,prediction,-torch.ones_like(prediction)*100000)
			pre_softmax = (prediction - prediction.max(dim=1, keepdim=True)[0]).softmax(dim=1)  # B * (Sample_num)
			target_pre = pre_softmax[:, :self.train_max_pos_item]  # B * pos_max_num
			target_pre = torch.where(mask[:,:self.train_max_pos_item]==1,target_pre,torch.ones_like(target_pre))
			loss = -(target_pre).log().sum(axis=1).div(pos_length)

			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss

		elif self.loss_n=='attention_rank': #softmax CE, but add more punishment for neg samples by adding the (1-label_i)log(1-predict_i) term
			target=torch.where(target!=-1,target.float(),-torch.tensor(float("Inf")).float().to(self.device))
			target_softmax = (target-target.max()).softmax(dim=1)

			prediction=torch.where(mask==1,prediction,-torch.ones_like(target)*100000)
			prediction_softmax = (prediction-prediction.max()).softmax(dim=1)

			prediction_softmax1 = torch.where(mask==1,prediction_softmax,torch.ones_like(prediction_softmax))#make sure that paddings are 1, so that after log they are 0
			loss_1 = -(target_softmax*prediction_softmax1.log()).sum(dim=1)

			prediction_softmax2 = torch.where(mask==1,prediction_softmax,torch.zeros_like(prediction_softmax))#make sure that paddings are 0, so that after log 1-they are 0
			prediction_softmax2 = torch.where(prediction_softmax2!=1,prediction_softmax2,torch.zeros_like(prediction_softmax2))#make sure that no 1 in prediction_softmax2, because it only happens when only 1 sample, so its loss must be 0
			loss_2 = -((1-target_softmax)*(1-prediction_softmax2).log()).sum(dim=1)

			loss = loss_1+loss_2
			loss = loss*test_have_neg/test_have_neg.sum()*len(test_have_neg)
			loss = loss.mean()
			return loss
		
		elif self.loss_n=='pointwiseCE':
			sample_length=mask.sum(axis=1)
			prediction = torch.sigmoid(out_dict['prediction'])
			loss = F.binary_cross_entropy(prediction,target.float(),reduction='none')
			loss = loss.mul(mask)
			return loss.sum(axis=1).div(sample_length).mean()

		elif self.loss_n=='sampled_softmax': 
			'''
			Reference:
				On the effectiveness of sampled softmax loss for item recommendation. Wu et al. 2022. Arxiv.
			'''
			pos_mask=torch.where(target==1,target,torch.zeros_like(target))
			relative_exp = (torch.exp(prediction*pos_mask)*pos_mask).sum(dim=-1) / (torch.exp(prediction*mask)*mask).sum(dim=-1)
			loss = -relative_exp.log()
			loss = loss.mean()
			return loss
		
		elif self.loss_n=='probCE':
			sample_length=mask.sum(axis=1)
			loss = F.binary_cross_entropy(prediction*mask,target.float(),reduction='none')
			loss = loss.mul(mask)
			# return loss.sum(axis=1).div(sample_length).mean()
			return loss.sum(axis=1).mean()
		
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))


	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			if self.phase=='train':
				self.pos_len=self.model.train_max_pos_item
				self.neg_len=self.model.train_max_neg_item
			else:
				self.pos_len=self.model.test_max_pos_item
				self.neg_len=self.model.test_max_neg_item

		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['pos_items'][index]

			if self.phase=='train':
				pos_len=self.model.train_max_pos_item
				neg_len=self.model.train_max_neg_item
			else:
				pos_len=self.model.test_max_pos_item
				neg_len=self.model.test_max_neg_item
			user_id, target_item = self.data['user_id'][index], self.data['pos_items'][index]
			
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			elif self.phase != 'train':
				neg_items = self.data['neg_items'][index]
			else:
				neg_items = self.data['neg_items'][index]
			'''pos_num=self.data['pos_num'][index]
			neg_num=self.data['neg_num'][index]'''
			if self.model.test_all and self.status !='train':
				item_ids = np.array(self.padsample(target_item[:pos_len],pos_len)+neg_items.tolist())
			else:
				item_ids = np.array(self.padsample(target_item[:pos_len],pos_len)+self.padsample(neg_items[:neg_len],neg_len))

			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids,
				'pos_num': min(self.data['pos_num'][index],self.pos_len),
				'neg_num': min(self.data['neg_num'][index],self.neg_len)
			}
			return feed_dict

		def padsample(self,li,max_sample):
			li=list(li)
			li=li+[0 for i in range(max_sample-len(li))]
			li=li[:max_sample]
			return li

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]):
			feed_dict = super().collate_batch(feed_dicts)
			# assert 'pos_items' in feed_dict and 'neg_items' in feed_dict

			# pos_items = feed_dict['pos_items']
			# if pos_items.shape[-1] < self.pos_len: # padding positive items
			# 	pos_items = torch.cat((pos_items, torch.zeros(pos_items.shape[0],self.pos_len-pos_items.shape[-1])),dim=-1)
			# neg_items = feed_dict['neg_items']
			# if neg_items.shape[-1] < self.neg_len: # padding negative items
			# 	neg_items = torch.cat((neg_items, torch.zeros(neg_items.shape[0],self.neg_len-neg_items.shape[-1])),dim=-1)
			# feed_dict['item_id'] = torch.cat((pos_items,neg_items),dim=-1).long()
			# feed_dict.pop('pos_items')
			# feed_dict.pop('neg_items')
			return feed_dict
		
		def actions_before_epoch(self): 
			# Have to define it in order to use the pre-defined negative items for training.
			# Or else the negative sampling function of general model dataset will be called.
			pass

class ImpressionContextModel(ImpressionModel):

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--include_id',type=int,default=1,
							help='Whether add ids in context information.')
		return ImpressionModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.include_id = args.include_id
	
	class Dataset(ImpressionModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			self.include_id = model.include_id

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			All_context = [] # concatenate all context information
			if len(self.corpus.user_feature_names):
				user_features = np.array([self.corpus.user_features[feed_dict['user_id']][c] 
							for c in self.corpus.user_feature_names]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				All_context = user_features
			if len(self.corpus.item_feature_names):
				items_features = np.array([[self.corpus.item_features[iid][c] if iid>0 else 0 for c in self.corpus.item_feature_names] 
											for iid in feed_dict['item_id'] ])
				All_context = np.concatenate([All_context, items_features],axis=-1) if len(All_context) else items_features
			if len(self.corpus.context_feature_names):
				context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1).repeat(
							feed_dict['item_id'].shape[0],axis=0)
				All_context = np.concatenate([All_context, context_features], axis=-1) if len(All_context) else context_features
			id_names = []
			if self.include_id:
				user_id = np.array([feed_dict['user_id']]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				item_ids = feed_dict['item_id'].reshape(-1,1)
				All_context = np.concatenate([All_context, user_id, item_ids],axis=-1) if len(All_context) else np.concatenate([user_id,item_ids],axis=-1)
				id_names = ['user_id','item_id']
			# transfer into multi-hot embedding
			base = 0
			u_idx, i_idx = [], []	
			for i, feature in enumerate(self.corpus.user_feature_names + self.corpus.item_feature_names + self.corpus.context_feature_names + id_names):
				All_context[:,i] += base
				base += self.corpus.feature_max[feature]
				if feature in self.corpus.user_feature_names+['user_id']:
					u_idx.append(i)
				if feature in self.corpus.item_feature_names+['item_id']:
					i_idx.append(i)
			feed_dict['context_mh'] = All_context # item num * feature num (will repeat 'item num' times for user and context features)
			feed_dict['umh_idx'] = u_idx
			feed_dict['imh_idx'] = i_idx
			return feed_dict

class ImpressionContextSituModel(ImpressionModel):
    # situation相关的feature不加入multi-hot的计算
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--include_id',type=int,default=1,
							help='Whether add ids in context information.')
		return ImpressionModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.include_id = args.include_id
	
	class Dataset(ImpressionModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			self.include_id = model.include_id

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			for feature in self.data.keys():
				if feature[:2] == 'c_':
					feed_dict[feature] = self.data[feature][index]
			All_context = [] # concatenate all context information
			if len(self.corpus.user_feature_names):
				user_features = np.array([self.corpus.user_features[feed_dict['user_id']][c] 
							for c in self.corpus.user_feature_names]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				All_context = user_features
			if len(self.corpus.item_feature_names):
				items_features = np.array([[self.corpus.item_features[iid][c] if iid>0 else 0 for c in self.corpus.item_feature_names] 
											for iid in feed_dict['item_id'] ])
				All_context = np.concatenate([All_context, items_features],axis=-1) if len(All_context) else items_features
			id_names = []
			if self.include_id:
				user_id = np.array([feed_dict['user_id']]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				item_ids = feed_dict['item_id'].reshape(-1,1)
				All_context = np.concatenate([All_context, user_id, item_ids],axis=-1) if len(All_context) else np.concatenate([user_id,item_ids],axis=-1)
				id_names = ['user_id','item_id']
			# transfer into multi-hot embedding
			base = 0
			u_idx, i_idx = [], []	
			for i, feature in enumerate(self.corpus.user_feature_names + self.corpus.item_feature_names + id_names):
				All_context[:,i] += base
				base += self.corpus.feature_max[feature]
				if feature in self.corpus.user_feature_names+['user_id']:
					u_idx.append(i)
				if feature in self.corpus.item_feature_names+['item_id']:
					i_idx.append(i)
			feed_dict['context_mh'] = All_context # item num * feature num (will repeat 'item num' times for user and context features)
			feed_dict['umh_idx'] = u_idx
			feed_dict['imh_idx'] = i_idx
			return feed_dict

class ImpressionContextSeqModel(ImpressionContextModel):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return ImpressionContextModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(ImpressionModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			self.include_id = model.include_id
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key])[idx_select]
		
		def _convert_multihot(self, fname_list, data):
			base = 0
			for i, feature in enumerate(fname_list):
				data[:,i] += base
				base += self.corpus.feature_max[feature]
			return data

		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			
			if len(self.corpus.user_feature_names):
				user_fnames = self.corpus.user_feature_names
				user_features = [self.corpus.user_features[feed_dict['user_id']][c] for c in self.corpus.user_feature_names] 
				if self.include_id:
					user_fnames = ['user_id'] + user_fnames
					user_features = [feed_dict['user_id']] + user_features
				feed_dict['user_features'] = self._convert_multihot(user_fnames, np.array(user_features).reshape(1,-1))
			elif self.include_id:
				feed_dict['user_features'] = feed_dict['user_id'].reshape(-1,1)
			
			if len(self.corpus.item_feature_names):
				item_fnames = self.corpus.item_feature_names
				items_features = np.array([[self.corpus.item_features[iid][c] if iid in self.corpus.item_features else 0
                                		for c in self.corpus.item_feature_names] for iid in feed_dict['item_id'] ])
				items_features_history = np.array([[self.corpus.item_features[iid][c] if iid in self.corpus.item_features else 0
                                        for c in self.corpus.item_feature_names] for iid in feed_dict['history_items'] ])
				if self.include_id:
					item_fnames = ['item_id']+item_fnames
					item_ids = feed_dict['item_id'].reshape(-1,1)
					items_features = np.concatenate([item_ids, items_features],axis=1)
					his_item_ids = feed_dict['history_items'].reshape(-1,1)
					items_features_history = np.concatenate([his_item_ids,items_features_history],axis=1)
				feed_dict['item_features'] = self._convert_multihot(item_fnames, np.array(items_features))
				feed_dict['history_item_features'] = self._convert_multihot(item_fnames,np.array(items_features_history))
			elif self.include_id:
				feed_dict['item_features'] = feed_dict['item_id'].reshape(-1,1)
				feed_dict['history_item_features'] = feed_dict['history_items'].reshape(-1,1)

			if len(self.corpus.context_feature_names):
				context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1)#.repeat(
							# feed_dict['item_id'].shape[0],axis=0)
				feed_dict['context_features'] = self._convert_multihot(self.corpus.context_feature_names, context_features)
				for feature in self.corpus.context_feature_names:
					feed_dict[feature] = self.data[feature][index]
			
			return feed_dict
	
		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)