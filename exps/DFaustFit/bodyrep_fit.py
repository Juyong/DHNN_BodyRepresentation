import sys
sys.path.append('../../src')
import model as M
import util as U
# from bodyrec_src import optimize as OP
from gpu_batch_knn.batch_gpu_knn_pytorch import batch_knn_gpu_pytorch
import argparse
import torch
from torch import optim
import numpy as np
import os
import os.path as osp
import pickle
import math
import openmesh as om
from glob import glob
def read_target_pointclouds(name_list):
	points_list=[]
	norms_list=[]
	vnums=[]
	for name in name_list:
		mesh=om.read_trimesh(name)
		temp=mesh.points().T
		vnums.append(temp.shape[1])
		points_list.append(temp)
		mesh.request_face_normals()
		mesh.request_vertex_normals()
		mesh.update_normals()
		norms_list.append(mesh.vertex_normals().T)
	points=np.zeros((3*len(vnums),max(vnums)),np.float32)
	norms=np.zeros((3*len(vnums),max(vnums)),np.float32)
	for i,(p,n) in enumerate(zip(points_list,norms_list)):
		points[3*i:3*i+3,0:vnums[i]]=p
		norms[3*i:3*i+3,0:vnums[i]]=n
	return points,norms,np.array(vnums,np.int32)


parser = argparse.ArgumentParser(description='optimize for target meshs')
# parser.add_argument('--vgan-model',default='None',metavar='M',
#                     help='pretrained vgan model'))
parser.add_argument('--gpu-id',type=int,default=0,metavar='ID',
                    help='if cuda enable, which gpu to use')
args = parser.parse_args()


shapenum=50
posenum=72
vnum=12500

device = torch.device("cuda:"+str(args.gpu_id))
torch.cuda.set_device(args.gpu_id)

bodyrep=M.initial_BodyRep('../../models/bodyTem',shapenum,posenum,vnum).to(device)
convert_f_points=M.AcapDecodeLayer('../../models/bodyTem',anchor_id=6263).to(device)

#used for computing vertex normals
tri_fs=convert_f_points.face_index
vertex_index,face_index=U.compute_connectivity_infos(om.read_trimesh('../../models/bodyTem/template.obj'),device)

rec_data_root='scan_data'
save_root='bodyrep_rec'
if not os.path.isdir(save_root):
	os.system('mkdir '+save_root)

names=glob(osp.join(rec_data_root,'*.obj'))
pt_names=[name[0:-3]+'ply' for name in names]
num=len(pt_names)
with open('bodyrep_corres.txt','r') as ff:
	corre_ids=ff.read().split()
	corre_ids=[int(ind) for ind in corre_ids]
	corre_ids=torch.from_numpy(np.array(corre_ids).astype(np.int64).reshape(-1,2).T).to(device)
print('total {} meshes...'.format(num))

batch_size=200
start_id=0
max_iters=1600
thred=0.002
log_internal=50
icp_thread=0.15
icp_angle_thread=45.

weights=list(zip([100.,50.,10.,5.,1.],
[100.,50.,10.,5.,1.],
[1.e5,1.e5,-1,-1,-1],
[-1,-1,1.e5,1.e5,1.e5]))
def get_weight(time):
	if time<200:
		return weights[0]
	elif time<400:
		return weights[1]
	elif time<600:
		return weights[2]
	elif time<1000:
		return weights[3]
	else:
		return weights[4]
# def get_weight(time):
# 	if time<10:
# 		return weights[0]
# 	elif time<20:
# 		return weights[1]
# 	elif time<30:
# 		return weights[2]
# 	elif time<40:
# 		return weights[3]
# 	else:
# 		return weights[4]
while start_id<num:
	end_id=start_id+batch_size
	if end_id>num:
		end_id=num
	tar_corre_points=U.read_target_points(names[start_id:end_id])
	tar_corre_points=torch.from_numpy(tar_corre_points).to(device)
	tar_corre_points=tar_corre_points[:,corre_ids[1,:]]
	tar_pt_points,tar_pt_norms,tar_pt_vnums=read_target_pointclouds(pt_names[start_id:end_id])	
	tar_pt_points=torch.from_numpy(tar_pt_points).to(device)
	tar_pt_norms=torch.from_numpy(tar_pt_norms).to(device)
	data_size=end_id-start_id
	ss=torch.zeros((data_size,shapenum)).to(device)
	ss.requires_grad=True
	ps=torch.zeros((data_size,posenum)).to(device)
	Rs=torch.zeros((data_size,3)).to(device)
	Rs.data.uniform_(-0.01,0.01)
	Ts=torch.zeros((data_size,3)).to(device)	
	ps.requires_grad=True
	Rs.requires_grad=True
	Ts.requires_grad=True
	optimizer=optim.Adam([ss,ps,Rs,Ts], lr=0.04)
	scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[600,1200,1500], gamma=0.4)
	rec_loss=(torch.ones((1))*100.0).to(device)
	times=0
	print('start optimize {} to {} meshes...'.format(start_id,end_id))
	while rec_loss.item()>thred and times<max_iters:
		(r_ps_w,r_ss_w,rec_corres_w,rec_pt_w)=get_weight(times)
		regular_shape_loss=torch.pow(ss,2).mean()
		regular_pose_loss=torch.pow(ps,2).mean()	
		loss=r_ps_w*regular_pose_loss+r_ss_w*regular_shape_loss
		(ske_pfs,pfs)=bodyrep(ss,ps)
		rec_points=convert_f_points(U.unscale_features(pfs,bodyrep.vs_scale)).permute(0,2,1).reshape(-1,vnum)
		rec_points=U.RT_points_Tensor(rec_points,Rs,Ts)
		if rec_corres_w>0:
			rec_corre_loss=U.GeometryEqualLoss(rec_points[:,corre_ids[0,:]],tar_corre_points,corre_ids.shape[-1])
			loss=loss+rec_corres_w*rec_corre_loss
		if rec_pt_w>0:
			_,knn_indexs=batch_knn_gpu_pytorch(tar_pt_points.reshape(data_size,3,tar_pt_points.shape[-1]),rec_points.reshape(data_size,3,vnum).permute(1,0,2).reshape(3,-1),tar_pt_vnums,
					np.array([vnum]*data_size,np.int32),1)
			knn_indexs=knn_indexs.reshape(data_size,vnum)
			knn_vecs=torch.zeros(data_size*3,vnum,device=device)
			knn_vecs[0::3,:]=rec_points[0::3,:]-tar_pt_points[0::3,:].gather(1,knn_indexs)
			knn_vecs[1::3,:]=rec_points[1::3,:]-tar_pt_points[1::3,:].gather(1,knn_indexs)
			knn_vecs[2::3,:]=rec_points[2::3,:]-tar_pt_points[2::3,:].gather(1,knn_indexs)
			knn_dists=knn_vecs.reshape(data_size,3,vnum).norm(p=None,dim=1)
			rec_norms=U.compute_vnorms(rec_points.reshape(data_size,3,vnum).permute(0,2,1),tri_fs,vertex_index,face_index).permute(0,2,1).reshape(3*data_size,vnum)
			# rec_norms=OP.compute_batch_vnorms(rec_points,trifs_gpu,vfadj_mean_matrix).reshape(3*data_size,12500);
			tar_corre_norms=torch.zeros(data_size*3,vnum,device=device)
			check1=(knn_dists<icp_thread).to(torch.float)
			tar_corre_norms[0::3,:]=tar_pt_norms[0::3,:].gather(1,knn_indexs)
			tar_corre_norms[1::3,:]=tar_pt_norms[1::3,:].gather(1,knn_indexs)
			tar_corre_norms[2::3,:]=tar_pt_norms[2::3,:].gather(1,knn_indexs)
			check2=((rec_norms*tar_corre_norms).reshape(data_size,3,vnum).sum(1)>math.cos(icp_angle_thread*math.pi/180.)).to(torch.float)
			valid_pair_index=check1*check2
			#p2n dists
			knn_dists=torch.abs((knn_vecs*tar_corre_norms).reshape(data_size,3,vnum).sum(1))
			#can modify knn_dists as Geman_McClure_Loss
			if valid_pair_index.sum()>0:
				rec_p2p_loss=(knn_dists*valid_pair_index).sum()/valid_pair_index.sum()
			else:
				rec_p2p_loss=(knn_dists*valid_pair_index).sum()
			loss=loss+rec_pt_w*rec_p2p_loss
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		if times%log_internal==0:
			print('{}th total mean loss:{:.4f}'.format(times,loss.item()))
			if rec_corres_w>0:
				print('rec_coore_loss mean:{:.4f}'.format(rec_corre_loss.item()))
			if rec_pt_w>0:
				print('rec_p2p_loss mean:{:.4f}'.format(rec_p2p_loss.item()))
			print('regular_shape_loss:{:.4f}, regular_pose_loss:{:.4f}'.format(regular_shape_loss.item(),regular_pose_loss.item()))
			print('Rangle mean:{:.4f}, Ts mean:{:.4f}'.format(Rs.norm(2,1).mean().item()/3.1415926*180.0,Ts.norm(2,1).mean().item()))
		times+=1
	(ske_pfs,pfs)=bodyrep(ss,ps)
	rec_points=convert_f_points(U.unscale_features(pfs,bodyrep.vs_scale)).permute(0,2,1).reshape(-1,vnum)
	rec_points=U.RT_points_Tensor(rec_points,Rs,Ts)
	

	snames=names[start_id:end_id]
	snames=[os.path.splitext(os.path.basename(name))[0] for name in snames]
	U.save_obj_files(rec_points.detach().cpu().numpy().T,tri_fs.cpu().numpy(),save_root,[name+'.obj' for name in snames])
	start_id=end_id
print('all done')