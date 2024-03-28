# -*- coding: UTF-8 -*-
'''
Hanyu Li 2023.07.04
'''

import logging
import numpy as np
import pandas as pd
import os
import sys

from helpers.BaseReader import BaseReader
from utils import utils

class ImpressionReader(BaseReader):
	"""
	Impression Reader reads impression data. In each impression there are pre-defined unfixed number of positive items and negative items
	"""
	@staticmethod
	def parse_data_args(parser):
		return BaseReader.parse_data_args(parser)
	
	def __init__(self, args):
		super().__init__(args)
		self._append_impression_info()

	def _read_data(self):
		logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
		self.data_df = dict()
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
			self.data_df[key] = utils.eval_list_columns(self.data_df[key])

		logging.info('Counting dataset statistics...')
		# In impression data, negative item lists can have unseen items (i.e., items without click)
		self.all_df = pd.concat([df[['user_id', 'item_id', 'time']] for df in self.data_df.values()])	
		self.n_users = max(self.all_df['user_id'].unique())+1#plus 0 padding index, the item and user id all begin from 1
		neg_df = pd.concat([df[['neg_items']] for df in self.data_df.values()])
		max_pos = max(self.all_df['item_id'].unique())
		max_neg = 0
		for neg_l in neg_df['neg_items'].tolist():
			if len(neg_l)==0:
				continue
			max_neg=max(max_neg,max(neg_l))
		self.n_items = max(max_pos,max_neg)+1#plus 0 padding index
		logging.info('Update impression data -- "# user": {}, "# item": {}, "# entry": {}'.format(
			self.n_users - 1, self.n_items - 1, len(self.all_df)))

	def _append_impression_info(self): # -> NoReturn:
		"""
		Merge all positive items of a request based on the timestamp, and get column 'pos_items' for self.data_df
		Add impression info to data_df: neg_num, pos_num
		"""
		logging.info('Merging positive items by timestamp...')
		#train,val,test
		neg_item_lists={'train':[],'dev':[],'test':[]}
		pos_item_lists={'train':[],'dev':[],'test':[]}
		
		mask = {'train':[],'dev':[],'test':[]}
		for key in self.data_df.keys():
			df=self.data_df[key].copy()
			df['last_user_id'] = df['user_id'].shift(1)
			df['last_time'] = df['time'].shift(1)

			positive_items, negative_items = [], []
			current_pos, current_neg = set(), set()
			for uid, last_uid, time, last_time, iid, neg_items in \
					df[['user_id','last_user_id','time','last_time','item_id','neg_items']].to_numpy():
				if uid == last_uid and time == last_time:
					positive_items.append([])
					negative_items.append([])
					mask[key].append(0)
				else:
					if len(current_pos):
						positive_items.append(list(current_pos))
						negative_items.append(list(current_neg))
						mask[key].append(1)
					current_pos, current_neg = set(), set()
				current_pos = current_pos.union(set([iid]))
				current_neg = current_neg.union(set(neg_items))
			# last session
			if len(current_pos):
				positive_items.append(list(current_pos))
				negative_items.append(list(current_neg))
				mask[key].append(1)
			self.data_df[key]['pos_items'] = positive_items
			self.data_df[key]['neg_items'] = negative_items
			self.data_df[key]=self.data_df[key][np.array(mask[key])==1]

		logging.info('Appending neg_num & pos_num...')

		neg_num_sum, pos_num_sum = 0,0
		for key in ['train', 'dev', 'test']:
			df = self.data_df[key]
			neg_num = list()
			pos_num = list()
			for neg_items in df['neg_items']:
				if 0 in neg_items:
					neg_num.append(neg_items.index(0))
				else:
					neg_num.append(len(neg_items))
			self.data_df[key]['neg_num']=neg_num
			for pos_items in df['pos_items']:
				if 0 in pos_items:
					pos_num.append(pos_items.index(0))
				else:
					pos_num.append(len(pos_items))
			self.data_df[key]['pos_num']=pos_num
			# !!! TODO: Ask hanyu how it was performed in previous versions
			self.data_df[key] = self.data_df[key].loc[self.data_df[key].neg_num>0].reset_index(drop=True) # Retain sessions with negative data only
			neg_num_sum += sum(neg_num)
			pos_num_sum += sum(pos_num)
		neg_num_avg = neg_num_sum / sum([self.data_df[key].shape[0] for key in self.data_df])
		pos_num_avg = pos_num_sum / sum([self.data_df[key].shape[0] for key in self.data_df])
		
		logging.info('train, dev, test request num: '+str(len(self.data_df['train']))+' '+str(len(self.data_df['dev']))+' '+str(len(self.data_df['test'])))
		logging.info("Average positive items / impression = %.3f, negative items / impression = %.3f"%(
			pos_num_avg,neg_num_avg))