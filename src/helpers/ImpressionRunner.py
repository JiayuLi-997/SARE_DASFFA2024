# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner

from torch.utils.tensorboard import SummaryWriter

def dcg_at_k(r, k, method=1):
	"""
	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider
		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
	Returns:
		Discounted cumulative gain
	"""
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			return np.sum(r / np.log2(np.arange(2, r.size + 2)))
		else:
			raise ValueError('method must be 0 or 1.')
	return 0.


def ndcg_at_k(r, k, method=1):
	"""
	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider
		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
	Returns:
		Normalized discounted cumulative gain
	"""
	dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
	if not dcg_max:
		return 0.
	return dcg_at_k(r, k, method) / dcg_max

def map_at_k(r,k):
	if r.size and r.sum()>0:
		p_sum=0
		pos_num=0
		for i in range(len(r)):
			if r[i]==1:
				pos_num+=1
				if i<k:
					p_sum+=(pos_num/(i+1))
		return p_sum/min(pos_num,k)
	return 0.

def hr_at_k(r,k):
	return int(r[:k].sum()>0)

class ImpressionRunner(BaseRunner):
	@staticmethod
	def parse_runner_args(parser):
		return BaseRunner.parse_runner_args(parser)

	@staticmethod
	def evaluate_method(predictions: np.ndarray, topk: list, metrics: list, test_all: bool,neg_num,pos_num_max,pos_num=None,check_sort_idx=0,test_num_neg=0,ret_all=0) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, when pos_num=None, the first column is the score for ground-truth item, if pos_num!=None, the 0:pos_num column is ground-truth. Also, pos_num:pos_num+neg_num is negative item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		if test_all:
			pass
		else:
			if pos_num is None:
				pos_num=[1 for i in range(len(predictions))]
			#predictions: pos, -inf, neg, -inf

			#make sure that positive items will be ranked lower than neg items, when they have the same prediction values
			pos_mask = np.ones((predictions.shape[0],pos_num_max))
			rest_mask = np.zeros((predictions.shape[0],predictions.shape[1]-pos_num_max))
			a_mask = np.concatenate((pos_mask,rest_mask),axis=1)
			eps=1e-6
			predictions=predictions-eps*a_mask


			sort_idx = (-predictions).argsort(axis=1,kind='mergesort') #快排不保序
			if check_sort_idx==1:
				logging.info(str(sort_idx[:10]))

			labels = np.zeros_like(predictions)
			for i in range(len(predictions)):
				labels[i][:min(pos_num[i],pos_num_max)] = 1

			for x in range(len(labels)):
				labels[x] = labels[x][sort_idx[x]]	

			neg_num_max = len(predictions[0])-pos_num_max
			for k in topk:
				ndcg = np.zeros(len(labels))
				for label_i,label in enumerate(labels): #
					valid_len = min(neg_num[label_i],neg_num_max)+min(pos_num[label_i],pos_num_max)
					if k > valid_len:
						ndcg[label_i] = ndcg_at_k(label[:valid_len],valid_len)
					else:
						ndcg[label_i] = ndcg_at_k(label[:valid_len],k)
				if ret_all == 0: 
					evaluations['NDCG@{}'.format(k)] = ndcg.mean()
				else:
					evaluations['NDCG@{}'.format(k)] = ndcg
			
			for k in topk:
				map = np.zeros(len(labels))
				for label_i,label in enumerate(labels): #
					valid_len = min(neg_num[label_i],neg_num_max)+min(pos_num[label_i],pos_num_max)
					if k > valid_len:
						map[label_i] = map_at_k(label[:valid_len],valid_len)
					else:
						map[label_i] = map_at_k(label[:valid_len],k)
				if ret_all == 0: 
					evaluations['MAP@{}'.format(k)] = map.mean()
				else:
					evaluations['MAP@{}'.format(k)] = map
			
			for k in topk:
				hr = np.zeros(len(labels))
				for label_i,label in enumerate(labels): #
					valid_len = min(neg_num[label_i],neg_num_max)+min(pos_num[label_i],pos_num_max)
					if k > valid_len:
						hr[label_i] = hr_at_k(label[:valid_len],valid_len)
					else:
						hr[label_i] = hr_at_k(label[:valid_len],k)
				if ret_all == 0:
					evaluations['HR@{}'.format(k)] = hr.mean()
				else:
					evaluations['HR@{}'.format(k)] = hr

			'''
			length is not a constant value
			'''		
		return evaluations

	def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list,check_sort_idx=0,all=0, writer=None, epoch=0) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		predictions = self.predict(data)
		if data.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(data.data['user_id']):
				clicked_items = [x[0] for x in data.corpus.user_his[u]]
				# clicked_items = [data.data['item_id'][i]]
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		#print('data',data.data)
		rows, cols = list(), list()
		mask = np.full_like(predictions,0)
		if 'pos_num' not in data.data.keys():
			pos_num=[1 for i in range(len(predictions))]
		else:
			pos_num=data.data['pos_num']
		neg_num=data.data['neg_num']
		if data.phase != 'train':
			mp = data.model.test_max_pos_item
			mn = data.model.test_max_neg_item
		else:
			mp = data.model.train_max_pos_item
			mn = data.model.train_max_neg_item

		for i in range(len(data.data['neg_num'])):
			rows.extend([i for _ in range(min(pos_num[i],mp))])
			rows.extend([i for _ in range(min(neg_num[i],mn))])
			cols.extend([_ for _ in range(min(pos_num[i],mp))])
			cols.extend([_ for _ in range(mp,mp+min(neg_num[i],mn))])
			#cols.extend([_ for _ in range(len(nonzero))])
			#cols.extend([_ for _ in range(min(pos_num[i],mp)+min(neg_num[i],mn))]) #别忘了开头的一个
		mask[rows, cols] = 1

		predictions = np.where(mask == 1,predictions,-np.inf)
		if 'pos_num' in data.data.keys():
			evaluate_dict = self.evaluate_method(predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,data.data['pos_num'],check_sort_idx,test_num_neg=data.neg_len,ret_all=all)
		else:
			evaluate_dict = self.evaluate_method(predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,check_sort_idx,test_num_neg=data.neg_len,ret_all=all)
		if writer is not None:
			writer.add_scalar('%s_NDCG3'%(data.phase),evaluate_dict['NDCG@3'],global_step=epoch)
		return evaluate_dict
	
	def fit(self, data: BaseModel.Dataset, epoch=-1, writer=None) -> float:
		model = data.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		data.actions_before_epoch()  # must sample before multi thread start

		model.train()
		loss_lst = list()
		dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=data.collate_batch, pin_memory=self.pin_memory)
		# dl = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
		#				 collate_fn=data.collate_batch, pin_memory=self.pin_memory)
		idx = 0
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
			model.optimizer.zero_grad()
			out_dict = model(batch)
			max_pos_num = model.train_max_pos_item
			pos_mask = 2*(torch.arange(max_pos_num)[None,:].to(model.device) < batch['pos_num'][:,None]).int()-1
			neg_mask=(torch.arange(out_dict['prediction'].size(1)-max_pos_num)[None,:].to(model.device) < batch['neg_num'][:,None]).int()-1
			labels = torch.cat([pos_mask,neg_mask],dim=-1)
			'''for i in range(len(batch['user_id'])):
				labels[i][:batch['pos_num'][i]] = 1
				labels[i][max_pos_num:max_pos_num+batch['neg_num'][i]] = 0'''
			loss = model.loss(out_dict,labels)
			if loss.isinf() or loss.isnan():
				print(loss)
			loss.backward()
			nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=1.0,norm_type=2)
			# if len(loss_lst) in [200,400,600]:
			#	 print("Last epoch")
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
			if epoch==1 and writer is not None:
				writer.add_scalar('train_loss_batch',loss_lst[-1],global_step=idx)
			idx += 1

		if epoch == 1:
			print(batch['item_id'].shape)

		if writer is not None:
			writer.add_scalar('train_loss',np.mean(loss_lst).item(),global_step=epoch)
		return np.mean(loss_lst).item()
