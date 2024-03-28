import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings
from collections import OrderedDict
from models.BaseModel import ImpressionModel
import logging

class BaseSARE(nn.Module):
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--situ_lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--situ_l2', type=float, default=0,
							help='Weight decay in optimizer.')
		parser.add_argument('--select_situations',type=str,default='c_weekday,c_period')
		parser.add_argument('--situ_emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--temperature',type=float,default=1)
		parser.add_argument('--prob_loss_n',type=str,default='BPRsimple')
		parser.add_argument('--prob_weights',type=float,default=0.01)
		parser.add_argument('--situ_loss_n',type=str,default='BPRsession')
		parser.add_argument('--situ_weights',type=float, default=1,)
		parser.add_argument('--rec_weights',type=int,default=1)	
		parser.add_argument('--avg_method',type=str,default='geo')
		parser.add_argument('--situ_fusion_method',type=str,default='user')
		parser.add_argument('--new_confidence',type=int,default=1)
		parser.add_argument('--ablation_ucfe',type=int,default=0, help='Whether to use ablation of ucfe. 0 for not ablation.')
		parser.add_argument('--ablation_psf',type=int,default=0)
		return parser

	def get_generalSARE_init(self, args, corpus):
		self.ablation_ucfe = args.ablation_ucfe
		self.ablation_psf = args.ablation_psf

		self.situ_lr = args.situ_lr
		self.situ_l2 = args.situ_l2
  
		self.avg_method = args.avg_method
		self.situ_embedding_size = args.situ_emb_size
		self.rec_weight = args.rec_weights
  
		self.new_confidence = args.new_confidence
		
		select_situations = args.select_situations.split(",")
		self.context_feature_dim = sum([v for k,v in corpus.feature_max.items() if k not in select_situations]) 
		self.context_feature_num = len([k for k in corpus.feature_max if k not in select_situations])
		self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names+['item_id']])
		self.item_feature_num = len(corpus.item_feature_names)+1
		self.user_feature_dim = sum([corpus.feature_max[f] for f in corpus.user_feature_names+['user_id']])
		self.user_feature_num = len(corpus.user_feature_names)+1

		self.optimization_method = 'joint'
		self.alpha = args.situ_weights
		self.prob_alpha = args.prob_weights
		self.situ_loss_n = args.situ_loss_n
		self.temp = args.temperature
		self.situ_feature_cnts = OrderedDict()
		self.all_situ_num = 1
		for situ, f_num in corpus.feature_max.items():
			if situ in select_situations:
				self.situ_feature_cnts[situ] = f_num
				self.all_situ_num *= f_num
		self.prob_loss_n = args.prob_loss_n
		self.situ_fusion = args.situ_fusion_method

		self.situ_feature_dim = sum([corpus.feature_max[f] for f in select_situations])
		return

	def get_generalSARE_params(self):
		self.situ_i_transform = nn.Linear(self.vec_size*self.item_feature_num,self.situ_embedding_size)
		self.situ_u_transform = nn.Linear(self.vec_size*self.user_feature_num,self.situ_embedding_size)
		# { elu, exponential, hard_sigmoid, linear, relu, selu, sigmoid, softplus, softsign, swish, tanh }
		self.activations = nn.ModuleList([
			nn.ELU(),
			# exp()
			nn.Hardsigmoid(),
			nn.Identity(),
			nn.ReLU(),
			nn.SELU(),
			nn.Sigmoid(),
			nn.Softplus(),
			nn.Softsign(),
			nn.Hardswish(),
			nn.Tanh()
		])
		if self.ablation_ucfe==1:
			self.la_weights = nn.Parameter(torch.randn(len(self.activations)),requires_grad=True)
		else:
			self.la_weights = nn.Linear(self.situ_embedding_size,len(self.activations))
		self.situ_embeddings = torch.nn.ModuleList()
		for situ, f_num in self.situ_feature_cnts.items():
			self.situ_embeddings.append(nn.Embedding(f_num,self.situ_embedding_size))
		if self.ablation_psf==1:
			self.situ_weights_W = nn.Parameter(torch.randn(len(self.situ_feature_cnts)),requires_grad=True)
		elif self.situ_fusion == 'user':
			self.situ_weights_W = nn.Linear(self.situ_embedding_size,len(self.situ_feature_cnts))
			# self.situ_weights_W = nn.Linear(self.vec_size*self.user_feature_num,len(self.situ_feature_cnts))
		elif self.situ_fusion == 'attention' or self.situ_fusion == 'attention_new':
			self.situ_weights_W = nn.Linear(self.situ_embedding_size,self.situ_embedding_size)

		if self.situ_loss_n in ['SoftmaxCE','BPRsession','BPRsoftall']:
			self.situ_loss_fn = eval(self.situ_loss_n)(self.train_max_pos_item,self.device)
		else:
			raise ValueError('No situation loss function!')
		if self.prob_loss_n in ['BPRall','BPRsession','BPRsimple','Logposavg','probCE']:
			self.prob_loss_fn = eval(self.prob_loss_n)(self.train_max_pos_item, self.device)
		else:
			raise ValueError('No probability loss functon!')
		if self.loss_n in ['BPRsession','BPRsoftall']:
			self.rec_loss_fn = eval(self.loss_n)(self.train_max_pos_item,self.device)
		else:
			raise ValueError('No recommender loss function!')

		logging.info("Rec loss: %s, Situ loss: %s, Prob loss: %s"%(str(self.rec_loss_fn),str(self.situ_loss_fn),
									str(self.prob_loss_fn)))

	def situation_predict(self, u_embeddings, i_embeddings, situ_target):
		if self.ablation_ucfe:
			s = self.la_weights.unsqueeze(0)#.repeat(i_embeddings.shape[0],1)
		else:
			s = self.la_weights(u_embeddings) # batch * activation layers
		i_activated = []
		for i,f in enumerate(self.activations):
			i_activated.append(s[:,i][:,None,None]*f(i_embeddings))
		pred_situ = torch.stack(i_activated,dim=-1).sum(dim=-1)

		situ_embeds = [situ_embedding(situ_id) for situ_id,situ_embedding in zip(situ_target,self.situ_embeddings)]
		situ_embed = torch.stack(situ_embeds,dim=-1) # batch * context embed * context num
		situ_embed_weights = self.get_situ_fusion_weights(u_embeddings, situ_embed) # batch * context num
		situ_embed = (situ_embed_weights[:,None,:] * situ_embed).sum(dim=-1) # / torch.norm(situ_embed,p=2,dim=1)[:,None,:] ).sum(dim=-1)

		pred_situ_prob = (pred_situ*situ_embed[:,None,:]).sum(dim=-1) / torch.norm(pred_situ,p=2,dim=2
						) / torch.norm(situ_embed,p=2,dim=1)[:,None]
		if pred_situ_prob.isnan().max() or pred_situ_prob.isinf().max():
			print('Situation probability error!')
		return pred_situ_prob, pred_situ, situ_embed


	def get_situ_fusion_weights(self, u_embeddings, situ_embed):
		# situ_embed: batch * emb size * situ num
		if self.ablation_psf:
			situ_embed_weights = self.situ_weights_W.unsqueeze(0)# batch * situ num
		elif self.situ_fusion == 'user':
			situ_embed_weights = self.situ_weights_W(u_embeddings).softmax(dim=-1) # batch * context num
		elif self.situ_fusion == 'attention':
			Qu = self.situ_weights_W(u_embeddings) # batch * situ emb size
			situ_embed_weights = (Qu[:,:,None]*situ_embed).sum(dim=1) # batch size * context num
			situ_embed_weights = (situ_embed_weights-situ_embed_weights.max(dim=-1,keepdim=True)[0]).softmax(dim=-1)
		elif self.situ_fusion == 'attention_new':
			Qu = self.situ_weights_W(u_embeddings) # batch * emb size 
			d_k = situ_embed.shape[-1]
			situ_embed_weights = torch.matmul(Qu.unsqueeze(1), situ_embed) / d_k**0.5 # batch * 1 * len
			situ_embed_weights = (situ_embed_weights-situ_embed_weights.max(dim=-1,keepdim=True)[0]).softmax(dim=-1)
			situ_embed_weights = situ_embed_weights.squeeze(dim=1)# [:,None,:] * situ_embed
		else:
			raise ValueError('Situation Fusion method %s is not defined!'%(self.situ_fusion))
		return situ_embed_weights
	
	def inference(self, feed_dict):
		out_dict = self.forward(feed_dict)
		prediction = out_dict['prediction']
		pos_mask = torch.arange(prediction.size(1))[None,:].to(self.device) < feed_dict['pos_num'][:,None]
		neg_mask = torch.arange(prediction.size(1))[None,:].to(self.device) < (feed_dict['neg_num']+self.train_max_pos_item)[:,None]
		neg_mask2 = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) >= self.train_max_pos_item).to(self.device)
		neg_mask = neg_mask & neg_mask2
		all_mask = (pos_mask | neg_mask).float()
		return self.get_situ_prob(out_dict, all_mask)
	
	def get_situ_prob(self, out_dict, all_mask):
		prediction = out_dict['prediction']
		prediction_temp = prediction*self.temp
		prediction_prob = prediction_temp - prediction_temp.max(dim=-1,keepdim=True)[0] - ((prediction_temp - prediction_temp.max(dim=-1,keepdim=True)[0]).exp()*all_mask).sum(dim=-1,keepdim=True).log()
		prediction_prob = prediction_prob.exp()
		out_dict['rec_prediction_prob'] = prediction_prob

		pred_situ = out_dict['situ_prediction']
		pred_situ_temp = pred_situ*self.temp
		pred_situ_prob = pred_situ_temp - pred_situ_temp.max(dim=-1,keepdim=True)[0] - ((pred_situ_temp - pred_situ_temp.max(dim=-1,keepdim=True)[0]).exp()*all_mask).sum(dim=-1,keepdim=True).log()
		pred_situ_prob = pred_situ_prob.exp()
		out_dict['situ_prediction_prob'] = pred_situ_prob

		if self.avg_method == 'situ':
			out_dict['prediction'] = pred_situ_prob
		elif self.avg_method == 'har':
			out_dict['prediction'] = prediction_prob * pred_situ_prob / (prediction_prob + pred_situ_prob)
		elif self.avg_method == 'geo':
			out_dict['prediction'] = torch.sqrt(pred_situ_prob*prediction_prob)
		elif self.avg_method == 'math':
			out_dict['prediction'] = (prediction_prob + pred_situ_prob) / 2 # batch * item num
		elif 'weights' in self.avg_method:
			if self.new_confidence:
				alpha = self.get_confidence_new(prediction_prob, all_mask) # batch
				beta = self.get_confidence_new(pred_situ_prob, all_mask) # batch
			else:
				alpha = self.get_confidence(prediction_prob, all_mask) # batch
				beta = self.get_confidence(pred_situ_prob, all_mask) # batch
			if self.avg_method == 'har_weights':
				out_dict['prediction'] = (alpha+beta)[:,None]*prediction_prob*pred_situ_prob / (alpha[:,None]*pred_situ_prob+beta[:,None]*prediction_prob)
			elif self.avg_method == 'geo_weights':
				out_dict['prediciton'] = torch.pow(prediction_prob**alpha[:,None] * pred_situ_prob**beta[:,None], (alpha+beta)[:,None])
			elif self.avg_method == 'math_weights':
				out_dict['prediction'] = (prediction_prob*alpha[:,None]+pred_situ_prob*beta[:,None])/(alpha+beta)[:,None]
		else:
			warnings.warn("Final prediction combination is not defined.")
			out_dict['prediction'] = prediction_prob

		if pred_situ_prob.isinf().max() or prediction_prob.isinf().max() or pred_situ_prob.isnan().max() or prediction_prob.isnan().max():
			print('Predicted situation probability error!')

		return out_dict
	
	def get_confidence(self, probability, mask):
		S = ((probability+1)*mask).sum(dim=-1)
		uncertainty = mask.sum(dim=-1) / S

		return 1-uncertainty
	
	def get_confidence_new(self, probability, mask):
		S = (probability*mask).max(dim=-1)[0] + 1
		uncertainty = 1 / S

		return 1-uncertainty

	def SARE_loss(self, out_dict, target):
		if self.optimization_method in  ['separate','joint']:
			pred_situ = out_dict['situ_prediction']
			prediction = out_dict['prediction']
			rec_loss = self.rec_loss_fn(prediction, target)
			situ_loss = self.situ_loss_fn(pred_situ,target) 
			
			if self.optimization_method == 'separate':
				return rec_loss + self.alpha * situ_loss, rec_loss, situ_loss, rec_loss
			else:
				mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
				out_dict = self.get_situ_prob(out_dict, mask)
				prob_loss = self.prob_loss_fn(out_dict['prediction'],target)
				if prob_loss.isnan() or prob_loss.isinf() or situ_loss.isnan() or rec_loss.isnan() or situ_loss.isinf() or rec_loss.isinf() or prediction.isinf().sum() or pred_situ.isinf().sum():
					print('Loss error!')
				return self.rec_weight*rec_loss + self.alpha * situ_loss + self.prob_alpha * prob_loss, self.prob_alpha*prob_loss, self.alpha*situ_loss, rec_loss
		else:
			mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
			out_dict = self.get_situ_prob(out_dict, mask)
			rec_loss = self.rec_loss_fn(prediction, target)
			return rec_loss, rec_loss, rec_loss, rec_loss


class SoftmaxCE(nn.Module):
	def __init__(self, train_max_pos_item,_):
		super(SoftmaxCE, self).__init__()
		self.train_max_pos_item = train_max_pos_item

	def forward(self, prediction, target):
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
		test_have_neg = mask[:,self.train_max_pos_item]#if no neg 0, has neg 1
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

class BPRsession(nn.Module):
	def __init__(self, train_max_pos_item, device):
		super(BPRsession, self).__init__()
		self.train_max_pos_item = train_max_pos_item
		self.device = device
	
	def forward(self, prediction, target):
		batch_size = prediction.shape[0]
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
		test_have_neg = mask[:,self.train_max_pos_item]#if no neg 0, has neg 1

		valid_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-1).transpose(-1,-2)
		pos_mask = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) < self.train_max_pos_item).to(self.device)
		neg_mask = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) >= self.train_max_pos_item).to(self.device)
		select_mask = pos_mask.unsqueeze(dim=-1) * neg_mask.unsqueeze(dim=-1).transpose(-1,-2) * valid_mask # get all valid mask in the two-dimensional matrix
		score_diff = prediction.unsqueeze(dim=-1) - prediction.unsqueeze(dim=-1).transpose(-1,-2) # batch * impression list * impression list
		score_diff_mask = score_diff * select_mask
			
		neg_pred=torch.where(neg_mask*mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		pos_pred=torch.where(pos_mask*mask==1,prediction,-torch.tensor(float("Inf")).float().to(self.device))
		pos_softmax = (pos_pred - pos_pred.max()).softmax(dim=1)

		pos_num = (target==1).sum(dim=-1)
		neg_num = (target==0).sum(dim=-1)

		loss = -((score_diff_mask.sigmoid()*neg_softmax.unsqueeze(dim=1)).sum(dim=-1)*pos_softmax).sum(dim=-1).log()
		loss = loss.mean()
		
		if loss.isinf().max() or loss.isnan().max():
			print(loss)
		return loss

class Logposavg(nn.Module):
	def __init__(self,_,device):
		super().__init__()
		self.device = device
		
	def forward(self, prediction, target=None):
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
		pos_num = (target==1).sum(dim=-1)
		posavg_target = (target*mask)/pos_num[:,None] # batch size * list
		neg_target = torch.where(target==0,torch.ones_like(target),torch.zeros_like(target))
		no_target = torch.where(target<0,torch.ones_like(target),torch.zeros_like(target))

		max_value, min_value = prediction.max(), prediction.min()
		mask_prediction = prediction*mask + no_target*0.5
		CE = - ( (mask_prediction).log()*posavg_target + (1-mask_prediction).log()*neg_target )

		loss_sample = (CE*mask).sum(dim=-1)
		loss = loss_sample.mean()

		if loss.isinf().max() or loss.isnan().max():
			print(loss)
		return loss

class BPRsimple(nn.Module):
	def __init__(self, train_max_pos_item, device):
		super(BPRsimple, self).__init__()
		self.train_max_pos_item = train_max_pos_item
		self.device = device
	
	def forward(self, prediction, target):
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
		valid_mask = mask.unsqueeze(dim=-1) * mask.unsqueeze(dim=-1).transpose(-1,-2)
		pos_mask = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) < self.train_max_pos_item).to(self.device)
		neg_mask = (torch.arange(prediction.size(1)).unsqueeze(0).repeat(prediction.shape[0],1) >= self.train_max_pos_item).to(self.device)
		select_mask = pos_mask.unsqueeze(dim=-1) * neg_mask.unsqueeze(dim=-1).transpose(-1,-2) * valid_mask # get all valid mask in the two-dimensional matrix
		score_diff = prediction.unsqueeze(dim=-1) - prediction.unsqueeze(dim=-1).transpose(-1,-2) # batch * impression list * impression list
		score_diff_mask = score_diff * select_mask
		
		loss = ((F.softplus(-score_diff_mask)*select_mask).sum(dim=-1)).sum(dim=-1)
		loss = loss.mean()

		return loss

class probCE(nn.Module):
	def __init__(self, train_max_pos_item, device):
		super(probCE, self).__init__()
		self.train_max_pos_item = train_max_pos_item
		self.device = device

	def forward(self, prediction, target):
		mask=torch.where(target==-1,target,torch.zeros_like(target))+1#only non pad item is 1,shape like prediction
		sample_length=mask.sum(axis=1)
		loss = F.binary_cross_entropy(prediction*mask,target.float(),reduction='none')
		loss = loss.mul(mask)
		if loss.isinf().max() or loss.isnan().max():
			print(loss)
		return loss.sum(axis=1).mean()
