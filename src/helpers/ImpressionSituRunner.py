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
from helpers.ImpressionRunner import ImpressionRunner


class ImpressionSituRunner(ImpressionRunner):
	def adjust_learning_rate(self, optimizer, epoch):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		self.learning_rate = self.learning_rate * 0.5 #(0.1 ** (epoch // 30))
		for idx, param_group in enumerate(optimizer.param_groups):
			if idx == 0:
				rec_lr = param_group['lr']
			if idx>1 and param_group['lr']>rec_lr:
				param_group['lr'] = self.learning_rate
		self.learning_rate = param_group['lr']
	
	def fit(self, data: BaseModel.Dataset, epoch=-1, writer=None) -> float:
		model = data.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		data.status = 'train'
		data.actions_before_epoch()  # must sample before multi thread start

		model.train()
		loss_lst = list()
		rec_loss_lst, situ_loss_lst, prob_loss_lst = list(),list(), list()
		# dl = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
		#				 collate_fn=data.collate_batch, pin_memory=self.pin_memory)
		dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=data.collate_batch, pin_memory=self.pin_memory)
		cnt = 0
		logging.info(model.optimization_method)
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
			loss, prob_loss, situ_loss, rec_loss = model.loss(out_dict,labels)
			loss.backward()
			nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=1.0,norm_type=2)
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
			rec_loss_lst.append(rec_loss.detach().cpu().data.numpy())
			situ_loss_lst.append(situ_loss.detach().cpu().data.numpy())
			prob_loss_lst.append(prob_loss.detach().cpu().data.numpy())
		logging.info('Rec loss: %.4f, Situ loss: %.4f, Prob loss: %.4f'%(np.mean(rec_loss_lst),
				np.mean(situ_loss_lst), np.mean(prob_loss_lst)))
		
		# self.adjust_learning_rate(optimizer=model.optimizer,epoch=epoch)
		# logging.info("Adjust lr to %.4f"%(self.learning_rate))

		if model.optimization_method == 'separate' and epoch%2==0:
			model.optimization_method = 'joint'
		# else:
		#	 model.optimization_method = 'separate'
		return np.mean(loss_lst).item()

	def predict(self, dataset: BaseModel.Dataset) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.status = 'test'
		dataset.model.eval()
		predictions = list()
		situ_predictions = list()
		rec_predictions = list()
		targets_all = list()
		user_ids = list()

		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				all_prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))
				prediction = all_prediction['prediction']
				situ_prediction = all_prediction['situ_prediction_prob']
				rec_prediction = all_prediction['rec_prediction_prob']
			else:
				all_prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
				prediction = all_prediction['prediction']
				situ_prediction = all_prediction['situ_prediction_prob']
				rec_prediction = all_prediction['rec_prediction_prob']
			predictions.extend(prediction.cpu().data.numpy())
			situ_predictions.extend(situ_prediction.cpu().data.numpy())
			rec_predictions.extend(rec_prediction.cpu().data.numpy())
			max_pos_num = dataset.model.train_max_pos_item
			pos_mask = 2*(torch.arange(max_pos_num)[None,:].to(dataset.model.device) < batch['pos_num'][:,None]).int()-1
			neg_mask=(torch.arange(all_prediction['prediction'].size(1)-max_pos_num)[None,:].to(dataset.model.device) < batch['neg_num'][:,None]).int()-1
			labels = torch.cat([pos_mask,neg_mask],dim=-1)
			targets_all.extend(labels.cpu().data.numpy())
			user_ids.extend(batch['user_id'].cpu().data.numpy())
		predictions = np.array(predictions)
		situ_predictions = np.array(situ_predictions)
		rec_predictions = np.array(rec_predictions)
		targets_all = np.array(targets_all)
		user_ids = np.array(user_ids)

		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		if not self.train_models and hasattr(dataset,'phase') and dataset.phase in ['dev','test']:
			logging.info(os.path.join(self.log_path,dataset.phase+'_prediction_%s.npy'%(self.save_appendix)))
			np.save( os.path.join(self.log_path,dataset.phase+'_prediction_%s.npy'%(self.save_appendix)),predictions,)
			np.save( os.path.join(self.log_path,dataset.phase+'_targets_%s.npy'%(self.save_appendix)),targets_all,)
			np.save( os.path.join(self.log_path,dataset.phase+'_userids_%s.npy'%(self.save_appendix)),user_ids,)

		return predictions, situ_predictions,targets_all, rec_predictions
	
	def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list,check_sort_idx=0,all=0, writer=None,epoch=-1) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		predictions, situ_predictions, targets_all, rec_predictions = self.predict(data)
		if data.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(data.data['user_id']):
				clicked_items = [x[0] for x in data.corpus.user_his[u]]
				# clicked_items = [data.data['item_id'][i]]
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf

		rows, cols = list(), list()
		mask = np.full_like(predictions,0)
		if 'pos_num' not in data.data.keys():
			pos_num=[1 for i in range(len(predictions))]
		else:
			pos_num=data.data['pos_num']
		neg_num=data.data['neg_num']
		if data.phase == 'train':
			mp = data.model.train_max_pos_item
			mn = data.model.train_max_neg_item
		else:
			mp = data.model.test_max_pos_item
			mn = data.model.test_max_neg_item
		for i in range(len(data.data['neg_num'])):
			rows.extend([i for _ in range(min(pos_num[i],mp))])
			rows.extend([i for _ in range(min(neg_num[i],mn))])
			cols.extend([_ for _ in range(min(pos_num[i],mp))])
			cols.extend([_ for _ in range(mp,mp+min(neg_num[i],mn))])
			#cols.extend([_ for _ in range(len(nonzero))])
			#cols.extend([_ for _ in range(min(pos_num[i],mp)+min(neg_num[i],mn))]) #别忘了开头的一个
		mask[rows, cols] = 1

		predictions = np.where(mask == 1,predictions,-np.inf)
		rec_predictions = np.where(mask == 1,rec_predictions,-np.inf)
		situ_predictions = np.where(mask == 1,situ_predictions,-np.inf)
		if 'pos_num' in data.data.keys():
			# rec_predictions = predictions / np.clip(situ_predictions, a_min=1e-6, a_max=1)	
			rec_evaluate = self.evaluate_method(rec_predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,data.data['pos_num'],check_sort_idx,test_num_neg=data.neg_len,ret_all=all)
			logging_str = 'Rec results: {}'.format(utils.format_metric(rec_evaluate))
			logging.info(logging_str)		 
			situ_evaluate = self.evaluate_method(situ_predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,data.data['pos_num'],check_sort_idx,test_num_neg=data.neg_len,ret_all=all)
			logging_str = 'Situ results: {}'.format(utils.format_metric(situ_evaluate))
			logging.info(logging_str)		 

			return self.evaluate_method(predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,data.data['pos_num'],check_sort_idx,test_num_neg=data.neg_len,ret_all=all)
		else:
			return self.evaluate_method(predictions, topks, metrics, data.model.test_all,data.data['neg_num'],mp,check_sort_idx,test_num_neg=data.neg_len,ret_all=all)