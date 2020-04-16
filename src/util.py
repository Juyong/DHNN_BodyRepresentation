import torch
import torch_geometric
import torch_geometric.utils as geo_utils
import openmesh as om
import numpy as np
import os
def read_target_points(name_list):
	return np.concatenate([om.read_trimesh(name).points().T for name in name_list],axis=0).astype(np.float32)

def save_obj_files(points, tris, save_root, prefix, save_interval=1):
	points_num,num=points.shape
	mesh_num=int(num/3)
	use_input_name=False
	if type(prefix)==list:
		assert(len(prefix)==mesh_num)
		use_input_name=True
	for id in range(mesh_num):
		if id%save_interval != 0:
			continue
		if use_input_name:
			file_name=os.path.join(save_root,prefix[id])
		else:
			file_name=os.path.join(save_root,prefix+'_{}.obj'.format(id))
		vs=points[:,3*id:3*id+3]
		with open(file_name,'w') as fp:
			for v in vs:
				fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
			if tris is not None:
				for f in tris+1:
					fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
			fp.close()

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


def RT_points_Tensor(points,Rs,Ts):
	angles=Rs.norm(2,1)
	indices=angles.nonzero()
	tRs=torch.zeros((Rs.shape[0],9),device=Rs.device)
	tRs[:,0::4]=1.0
	if indices.numel()>0 and indices.numel()<angles.numel():			
		indices=indices[:,0]
		crparas=Rs[indices]/angles[indices].reshape(-1,1)
		temp=(1-torch.cos(angles[indices]).reshape(-1,1))
		tempS=torch.sin(angles[indices]).reshape(-1,1)
		tRs[indices,0::4] = torch.cos(angles[indices]).reshape(-1,1) + temp*crparas*crparas
		tRs[indices,1] = temp.view(-1)*crparas[:,0]*crparas[:,1]-tempS.view(-1)*crparas[:,2]
		tRs[indices,2] = temp.view(-1)*crparas[:,0]*crparas[:,2]+tempS.view(-1)*crparas[:,1]
		tRs[indices,3] = temp.view(-1)*crparas[:,0]*crparas[:,1]+tempS.view(-1)*crparas[:,2]
		tRs[indices,5] = temp.view(-1)*crparas[:,1]*crparas[:,2]-tempS.view(-1)*crparas[:,0]
		tRs[indices,6] = temp.view(-1)*crparas[:,0]*crparas[:,2]-tempS.view(-1)*crparas[:,1]
		tRs[indices,7] = temp.view(-1)*crparas[:,1]*crparas[:,2]+tempS.view(-1)*crparas[:,0]
	elif indices.numel()==angles.numel():
		rparas=Rs/angles.reshape(-1,1)
		temp=(1-torch.cos(angles).reshape(-1,1))
		tempS=torch.sin(angles).reshape(-1,1)
		tRs[:,0::4] = torch.cos(angles).reshape(-1,1) + temp*rparas*rparas
		tRs[:,1] = temp.view(-1)*rparas[:,0]*rparas[:,1]-tempS.view(-1)*rparas[:,2]
		tRs[:,2] = temp.view(-1)*rparas[:,0]*rparas[:,2]+tempS.view(-1)*rparas[:,1]
		tRs[:,3] = temp.view(-1)*rparas[:,0]*rparas[:,1]+tempS.view(-1)*rparas[:,2]
		tRs[:,5] = temp.view(-1)*rparas[:,1]*rparas[:,2]-tempS.view(-1)*rparas[:,0]
		tRs[:,6] = temp.view(-1)*rparas[:,0]*rparas[:,2]-tempS.view(-1)*rparas[:,1]
		tRs[:,7] = temp.view(-1)*rparas[:,1]*rparas[:,2]+tempS.view(-1)*rparas[:,0]
	points=torch.bmm(tRs.reshape(-1,3,3),points.reshape(Rs.shape[0],3,-1))+Ts.reshape(Ts.shape[0],Ts.shape[1],1)
	return points.reshape(Rs.shape[0]*3,-1)

def GeometryEqualLoss(points1,points2,pnum=12500):
	delta=points1.reshape(-1,3,pnum).permute(0,2,1).reshape(-1,3)-points2.reshape(-1,3,pnum).permute(0,2,1).reshape(-1,3)
	del_len=delta.norm(2,1)
	return del_len.mean()

def Geman_McClure_Loss(input,c):
	return input*input*2.0/c/c/(input*input/c/c + 4.)

def compute_connectivity_infos(tm,device):
	face_index=torch.zeros(0,dtype=torch.long)
	vertex_index=torch.zeros(0,dtype=torch.long)
	for vid,fids in enumerate(tm.vertex_face_indices()):
		fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
		face_index=torch.cat((face_index,fids),dim=0)
		vertex_index=torch.cat((vertex_index,fids.new_ones(fids.shape)*vid),dim=0)
	face_index=face_index.to(device)
	vertex_index=vertex_index.to(device)
	return vertex_index,face_index

#verts:(v,3) or (b,v,3), tri_fs:(f,3)
def compute_fnorms(verts,tri_fs):
	v0=verts.index_select(-2,tri_fs[:,0])
	v1=verts.index_select(-2,tri_fs[:,1])
	v2=verts.index_select(-2,tri_fs[:,2])
	e01=v1-v0
	e02=v2-v0
	fnorms=torch.cross(e01,e02,-1)
	diss=fnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	fnorms=fnorms/diss
	return fnorms

#verts(b,vnum,3) or (vnum,3)
def compute_vnorms(verts,tri_fs,vertex_index,face_index):
	fnorms=compute_fnorms(verts,tri_fs)
	if torch_geometric.__version__<'1.3.2':
		vnorms=torch_scatter.scatter_add(fnorms.index_select(-2,face_index),vertex_index,-2,None,verts.shape[-2])
	else:
		vnorms=geo_utils.scatter_('add',fnorms.index_select(-2,face_index),vertex_index,dim=-2)
	diss=vnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	vnorms=vnorms/diss
	return vnorms