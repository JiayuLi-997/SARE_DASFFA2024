import logging
import pandas as pd
import os
import sys

from helpers.ImpressionContextReader import ImpressionContextReader

class ImpressionContextSeqReader(ImpressionContextReader):
	
	def __init__(self, args):
		super().__init__(args)
		self._append_his_info()

	def _append_his_info(self):
		"""
		self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
		add the 'position' of each interaction in user_his to data_df
		"""
		logging.info('Appending history info...')
		# sort_df = self.all_df.sort_values(by=['time', 'user_id'], kind='mergesort')
		data_dfs = dict()
		for key in ['train','dev','test']:
			data_dfs[key] = self.data_df[key].copy()
			data_dfs[key]['phase'] = key
		sort_df = pd.concat([data_dfs[phase][['user_id','item_id','time','pos_items','phase']] for phase in ['train','dev','test']]).sort_values(by=['time', 'user_id'], kind='mergesort')
		position = list()
		self.user_his = dict()  # store the already seen sequence of each user
		for uid, iid, t, pos_items, phase in zip(sort_df['user_id'], sort_df['item_id'], sort_df['time'], sort_df['pos_items'],sort_df['phase']):
			if uid not in self.user_his:
				self.user_his[uid] = list()
			position.append(len(self.user_his[uid]))
			if phase == 'train':
				for pos_item in pos_items:
					self.user_his[uid].append((pos_item, t))
		
		sort_df['position'] = position
		for key in ['train', 'dev', 'test']:
			self.data_df[key] = pd.merge(
				left=self.data_df[key], right=sort_df.drop(columns=['pos_items','phase']), how='left',
				on=['user_id', 'item_id', 'time'])
		del sort_df