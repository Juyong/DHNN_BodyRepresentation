import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch_geometric.utils as geo_utils
import numpy as np
import os.path as osp
import openmesh as om
#some utils
def quat2mat(quat):
	"""Convert quaternion coefficients to rotation matrix.
	Args:
		quat: size = [B, 4] 4 <===>(w, x, y, z)
	Returns:
		Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
	"""
	norm_quat = quat
	norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
	w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

	B = quat.size(0)

	w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
	wx, wy, wz = w*x, w*y, w*z
	xy, xz, yz = x*y, x*z, y*z

	rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
						  2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
						  2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
	return rotMat

def batch_rodrigues(theta):
	#theta N x 3
	batch_size = theta.shape[0]
	l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
	angle = torch.unsqueeze(l1norm, -1)
	normalized = torch.div(theta, angle)
	angle = angle * 0.5
	v_cos = torch.cos(angle)
	v_sin = torch.sin(angle)
	quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
	
	return quat2mat(quat)

#scaled_features(N,dim)
def unscale_features(scaled_features,scales):
	dim=scales.shape[1]
	scaled_features=scaled_features.reshape(-1,dim)	
	lenths=scales[0,:]-scales[1,:]
	features=(scaled_features+0.95)/1.9 * lenths.reshape(1,dim) + scales[1,:].reshape(1,dim)
	return features

def scale_features(unscaled_features,scales):
	dim=scales.shape[1]
	unscaled_features=unscaled_features.reshape(-1,dim)
	lenths=scales[0,:]-scales[1,:]
	features=(unscaled_features-scales[1,:].reshape(1,dim))/lenths.reshape(1,dim) * 1.9-0.95
	return features

#model related
class AcapDecodeLayer(Module):
	def __init__(self,acap_data_root,**kwargs):
		super(AcapDecodeLayer,self).__init__()
		mesh=om.read_trimesh(osp.join(acap_data_root,'template.obj'))
		self.vnum=mesh.n_vertices()
		self.fnum=mesh.n_faces()
		self.register_buffer('ref_points',torch.from_numpy(mesh.points().astype(np.float32)))
		self.register_buffer('face_index',torch.from_numpy(mesh.face_vertex_indices()).to(torch.long))
		self.register_buffer('edge_index',torch.from_numpy(mesh.hv_indices().transpose()).to(torch.long))
		self.register_buffer('edge_weights',torch.from_numpy(np.load(osp.join(acap_data_root,'template_ews.npy')).astype(np.float32)))
		self.register_buffer('pA',torch.from_numpy(np.load(osp.join(acap_data_root,'pA_acap.npy')).astype(np.float32)))
		if 'scale_npy' in kwargs:
			scales=np.load(osp.join(acap_data_root,kwargs['scale_npy'])).astype(np.float32)
		else:
			scales=np.load(osp.join(acap_data_root,'acap_scales.npy')).astype(np.float32)
		scales_min=scales[1,:]
		scales_len=scales[0,:]-scales[1,:]
		self.register_buffer('scales_min',torch.from_numpy(scales_min).reshape(1,-1))
		self.register_buffer('scales_len',torch.from_numpy(scales_len).reshape(1,-1))
		self.anchor_id=None
		if 'anchor_id' in kwargs:
			self.anchor_id=kwargs['anchor_id']
	# def unscale_acaps(self,acaps,scales=None):
	# 	if scales is None:
	# 		acaps=(acaps.reshape(-1,self.scales_len.numel())+0.95)/1.9 * self.scales_len+ self.scales_min
	# 	else:
	# 		scales_min=scales[1,:]
	# 		scales_len=scales[0,:]-scales[1,:]
	# 		acaps=(acaps.reshape(-1,scales_len.numel())+0.95)/1.9 * scales_len+ scales_min
	# 	return acaps
	def acap_prior(self,acaps,ps,batch_mean=False):
		batch_num=acaps.shape[0]
		refps=self.ref_points[None,:,:].expand(batch_num,self.ref_points.shape[0],3)
		vnum=ps.shape[1]
		assert(vnum==self.vnum)
		acaps=acaps.reshape(-1,9)		
		Rs=batch_rodrigues(acaps[:,0:3])
		Ss=acaps[:,3:9][:,[[0,1,2],[1,3,4],[2,4,5]]]
		Ts=Rs.matmul(Ss).reshape(batch_num,vnum,3,3)
		loss=(ps[:,self.edge_index[0,:],:]-ps[:,self.edge_index[1,:],:]).unsqueeze(-1)-(Ts[:,self.edge_index[0,:],:,:]).matmul((refps[:,self.edge_index[0,:],:]-refps[:,self.edge_index[1,:],:]).unsqueeze(-1))
		loss=(loss.squeeze(-1).pow(2).sum(-1))*self.edge_weights.reshape(1,-1)
		loss=geo_utils.scatter_('add',loss,self.edge_index[0,:],dim=1,dim_size=vnum)
		if batch_mean:
			return loss.sum(-1)
		else:
			return loss.sum()
	def forward(self,acaps,refps=None,require_rhs=False):		
		batch_num=acaps.shape[0]
		if refps is None:
			refps=self.ref_points[None,:,:].expand(batch_num,self.ref_points.shape[0],3)
		vnum=refps.shape[1]
		assert(vnum==self.vnum)
		acaps=acaps.reshape(-1,9)
		
		Rs=batch_rodrigues(acaps[:,0:3])
		Ss=acaps[:,3:9][:,[[0,1,2],[1,3,4],[2,4,5]]]
		Ts=Rs.matmul(Ss).reshape(batch_num,vnum,3,3)
		# Ts=torch.bmm(Rs,Ss).reshape(batch_num,vnum,3,3)
		rhs=self.edge_weights.reshape(1,-1,1,1)*(Ts[:,self.edge_index[0,:],:,:]+Ts[:,self.edge_index[1,:],:,:]).matmul((refps[:,self.edge_index[0,:],:]-refps[:,self.edge_index[1,:],:]).unsqueeze(-1))
		#b,v,3
		rhs=geo_utils.scatter_('add',rhs,self.edge_index[0,:],dim=1,dim_size=vnum).squeeze(-1)
		recps=self.pA.matmul(rhs)
		# align
		if self.anchor_id is None:
			recps=recps+torch.median(refps-recps,1,keepdim=True)[0]
		else:
			recps=recps-recps[:,self.anchor_id,:].reshape(batch_num,1,3)
		if require_rhs:
			return recps,rhs
		else:
			return recps

class learnable_skinning_layer(Module):
	#mask tensor(base_num,vnum), non zero ele represent this vertex related to the part
	def __init__(self,mask,dim=9):
		super(learnable_skinning_layer,self).__init__()		
		self.dim=dim
		self.base_num=mask.shape[0]
		self.vnum=mask.shape[1]
		#transpose, make vv_index continuous indices
		infos=mask.transpose(0,1).nonzero()
		self.register_buffer('vb_index',infos[:,1])
		self.register_buffer('vv_index',infos[:,0])		
		self.ws=Parameter(torch.zeros(self.vv_index.numel()))
	def smooth_loss(self,edge_index):
		ws=geo_utils.softmax(self.ws,self.vv_index,self.vnum)
		vws=ws.new_zeros(self.vnum,self.base_num)
		vws[self.vv_index,self.vb_index]=ws
		laplace=vws-geo_utils.scatter_('mean',vws[edge_index[1,:],:],edge_index[0,:])
		return torch.pow(laplace,2).sum(1).mean()
	#base_fs(batch,base_num,dim) memory order
	def forward(self,base_fs):
		ws=geo_utils.softmax(self.ws,self.vv_index,self.vnum)
		v_fs=geo_utils.scatter_('add',(base_fs.reshape(-1,self.base_num,self.dim)[:,self.vb_index,:])*ws[None,:,None],self.vv_index,1,self.vnum)
		return v_fs



class BodyRep(Module):
	def __init__(self,skin_layer,parts_scale,vs_scale,shapenum=50,posenum=72,vnum=12500,dim=9):
		super(BodyRep,self).__init__()
		self.skin=skin_layer
		self.shapenum=shapenum
		self.posenum=posenum
		self.parts_num=skin_layer.base_num
		self.vnum=vnum
		self.dim=dim
		assert(parts_scale.shape==(2,self.parts_num*dim))
		assert(vs_scale.shape==(2,vnum*dim))
		self.register_buffer('ps_scale',torch.from_numpy(parts_scale.astype(np.float32)))
		self.register_buffer('vs_scale',torch.from_numpy(vs_scale.astype(np.float32)))
		self.Dp=nn.Sequential(nn.Linear(posenum,400),nn.Tanh(),nn.Linear(400,800),nn.Tanh())
		self.Ds=nn.Sequential(nn.Linear(shapenum,400),nn.Tanh(),nn.Linear(400,800),nn.Tanh())
		self.Td=nn.Linear(800,vnum*dim)
		self.Cp=nn.Sequential(nn.Linear(posenum,400),nn.Tanh(),nn.Linear(400,800),nn.Tanh())
		self.Cs=nn.Sequential(nn.Linear(shapenum,400),nn.Tanh(),nn.Linear(400,800),nn.Tanh())
		self.Tc=nn.Linear(800,self.parts_num*dim)
	#return feature is scaled, which need to be unscaled before convert to mesh 
	def forward(self,ss,ps,request_neutral=False,**kwargs):
		ss=ss.reshape(-1,self.shapenum)
		ps=ps.reshape(-1,self.posenum)
		d=self.Dp(ps)
		ds=self.Ds(ss)
		d=d+ds
		d=self.Td(d)
		if request_neutral:
			ds=self.Td(ds)
		# if 'ts_ske_scale' not in kwargs or 'tp_ske_scale' not in kwargs:
		g=self.Cp(ps)
		gs=self.Cs(ss)
		g=g+gs
		g=self.Tc(g)
		b=self.skin(unscale_features(g,self.ps_scale))
		b=scale_features(b,self.vs_scale)
		f=b+d
		if request_neutral:
			gs=self.Tc(gs)
			bs=self.skin(unscale_features(gs,self.ps_scale))
			bs=scale_features(bs,self.vs_scale)
			fs=bs+ds
		if request_neutral:
			return (gs,fs),(g,f)
		else:
			return (g,f)


def initial_BodyRep(root,shapenum=50,posenum=72,vnum=12500,dim=9,require_grad=False):
	mask=torch.from_numpy(np.load(osp.join(root,'skinning_mask.npy')))
	skin_layer=learnable_skinning_layer(mask,dim)
	if osp.isfile(osp.join(root,'part_scales.txt')):
		parts_scale=np.loadtxt(osp.join(root,'part_scales.txt'))
	elif osp.isfile(osp.join(root,'part_scales.npy')):
		parts_scale=np.load(osp.join(root,'part_scales.npy'))
	else:
		raise Exception("Unable to find part_scales from "+root)
	vs_scale=np.load(osp.join(root,'acap_scales.npy'))
	Rep=BodyRep(skin_layer,parts_scale,vs_scale,shapenum,posenum,vnum,dim)
	Rep.load_state_dict(torch.load(osp.join(root,'RepWs.pth'),map_location='cpu'),False)
	if not require_grad:
		for param in Rep.parameters():
			param.requires_grad=False
		Rep.eval()
	return Rep