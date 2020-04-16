import torch_geometric.utils as geo_utils
import sys
sys.path.append('../../src')
import model as M
from model import batch_rodrigues
import util as U
from gpu_batch_knn.batch_gpu_knn_pytorch import batch_knn_gpu_pytorch
import argparse
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import pickle
import math
import openmesh as om
import scipy
import cv2


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


scales=bodyrep.vs_scale
ske_scales=bodyrep.ps_scale
pnum=bodyrep.parts_num

tri_fs=convert_f_points.face_index
vertex_index,face_index=U.compute_connectivity_infos(om.read_trimesh('../../models/bodyTem/template.obj'),device)
# # head top,left finger,left foot finger,right foot finger,right finger,left elbow,right elbow,nose tip,butt,left knee,right knee
# select_landmarks=torch.from_numpy(np.array([12466,5407,105,39,5004,8668,8735,11491,4775,2687,2645],np.int64)).to(device)
#FARM arap landmarks
select_landmarks=torch.from_numpy(np.array([11556,8792,6265,8437,5732,4727,2687,739,2645,753],np.int64)).to(device)

obj_files=[os.path.join('Faust_test_scan',f+'.ply') for f in ['test_scan_050','test_scan_103']]
save_root='bodyrep_rec'
if not os.path.isdir(save_root):
	os.system('mkdir '+save_root)
ref_pos=convert_f_points(U.unscale_features(bodyrep(torch.zeros(1,shapenum,device=device),torch.zeros(1,posenum,device=device))[1],scales)).permute(0,2,1).reshape(3,-1)
ref_mean_pos=ref_pos.mean(1)
ref_mean_pos=ref_mean_pos.cpu().numpy()



with open('body_finger_ear_vids.txt','r') as ff:
	body_hids=np.array([int(v) for v in ff.read().split()],np.int32)
	checks_not_hands=np.ones((12500),np.float32)
	checks_not_hands[body_hids]=0.
	assert(checks_not_hands.sum()==12500-body_hids.size)
	checks_not_hands=torch.from_numpy(checks_not_hands).to(device)

weights=list(zip([20,10,1,0,0,0],[20,10,1,0,0,0],
# [500,900,400,100.,10.,0.],
[1000,2000,400,100.,10.,0.],
[200,200,600,700,800,850],
#not accurate
[0.15,0.18,0.18,0.20,0.20,0.20],
# #more accurate landmarks
# [0.1,0.1,0.15,0.18,0.20,0.20],
[-1,-1,40,15,5,2.],
[3,3,-1,-1,-1,-1]))
unreal_pose_w=200
def get_weight(time):
	global unreal_pose_w,icp_thread,icp_angle_thread
	#icp thread can be decreased
	if time<450:
		icp_thread=0.4
		icp_angle_thread=80.
	elif time<750:
		icp_thread=0.2
		icp_angle_thread=60.
	else:
		icp_thread=0.1
		icp_angle_thread=45.

	if time<900:
		unreal_pose_w=200.
	else:
		unreal_pose_w=-1.

	if time<150:
		return weights[0]
	elif time<450:
		return weights[1]
	elif time<600:
		return weights[2]
	elif time<750:
		return weights[3]
	elif time<1000:
		return weights[4]
	else:
		return weights[5]




def unreal_loss(rRs,vec_axis,vec_sign,check_axis,thred_angle,thred_type):
	assert(vec_sign*vec_sign==1.0)
	if thred_type=='miner':
		return torch.exp(F.relu(vec_sign*rRs[:,check_axis,vec_axis]-math.sin(thred_angle/180.*np.pi)))-1.
	elif thred_type=='maxer':
		return torch.exp(F.relu(-vec_sign*rRs[:,check_axis,vec_axis]+math.sin(thred_angle/180.*np.pi)))-1.
	else:
		assert(False)
#note template coordinate, which is different with smpl
def unrealistic_pose_loss(ske_fs):
	ske_fs=U.unscale_features(ske_fs,ske_scales)
	data_size=ske_fs.shape[0]
	Rs=batch_rodrigues(ske_fs.reshape(-1,9)[:,:3]).reshape(-1,pnum,3,3)
	loss=0
	rR=Rs[:,9,:].matmul(Rs[:,13,:].permute(0,2,1))
	loss+=unreal_loss(rR,0,1,1,-10,'maxer')
	rR=Rs[:,10,:].matmul(Rs[:,14,:].permute(0,2,1))
	loss+=unreal_loss(rR,0,1,1,-10,'maxer')
	rR=Rs[:,2,:].matmul(Rs[:,4,:].permute(0,2,1))
	loss+=unreal_loss(rR,0,1,1,0,'miner')
	# loss+=unreal_loss(rR,0,1,2,0,'maxer')

	# loss+=unreal_loss(-(rR[:,0,1]+math.sin(3./180.*np.pi)))
	rR=Rs[:,3,:].matmul(Rs[:,5,:].permute(0,2,1))
	loss+=unreal_loss(rR,0,1,1,0,'miner')
	# loss+=unreal_loss(rR,0,1,2,0,'miner')

	# rR=Rs[:,13,:].matmul(Rs[:,12,:].permute(0,2,1))
	# loss+=unreal_loss(rR,0,1,1,-10,'maxer')
	# rR=Rs[:,14,:].matmul(Rs[:,12,:].permute(0,2,1))
	# loss+=unreal_loss(rR,0,1,1,-10,'maxer')	
	# rR=Rs[:,4,:].matmul(Rs[:,6,:].permute(0,2,1))
	# loss+=unreal_loss(rR,2,1,2,10,'maxer')
	# loss+=unreal_loss(rR,0,1,1,-8,'maxer')
	# rR=Rs[:,5,:].matmul(Rs[:,6,:].permute(0,2,1))
	# loss+=unreal_loss(rR,2,-1,2,-10,'miner')
	# loss+=unreal_loss(rR,0,1,1,-8,'maxer')
	# rR=Rs[:,6,:].matmul(Rs[:,11,:].permute(0,2,1))
	# loss+=unreal_loss(rR,0,1,1,-20,'maxer')
	return loss
def pose_prior(ps):
	return U.Geman_McClure_Loss(F.relu(torch.abs(ps)-1.5),0.5).mean(1)

def RT_points_Tensor(points,deltaRs,Rs,Ts):
	if deltaRs is not None:
		Rotations=batch_rodrigues(deltaRs).matmul(Rs)
	else:
		Rotations=Rs
	points=torch.bmm(Rotations,points.reshape(Ts.shape[0],3,-1))+Ts.reshape(Ts.shape[0],Ts.shape[1],1)
	return points.reshape(Ts.shape[0]*3,-1)

def read_datas(name_list,expand_corre=False):
	points_list=[]
	meanpos_list=[]
	norms_list=[]
	vnums=[]
	corre_ps=[]
	for name in name_list:
		mesh=om.read_trimesh(name)
		temp=mesh.points().T
		vnums.append(temp.shape[1])
		points_list.append(temp)
		meanpos_list.append(temp.mean(1))
		mesh.request_face_normals()
		mesh.request_vertex_normals()
		mesh.update_normals()
		norms_list.append(mesh.vertex_normals().T)
		corre_ps.append(np.load(name[:-4]+'_farmarap10.npy').T)
	points=np.zeros((3*len(vnums),max(vnums)),np.float32)
	meanpos=np.stack(meanpos_list).astype(np.float32)
	norms=np.zeros((3*len(vnums),max(vnums)),np.float32)
	corre_ps=np.stack(corre_ps).astype(np.float32)
	corre_ps=corre_ps.reshape(-1,corre_ps.shape[-1])
	for i,(p,n) in enumerate(zip(points_list,norms_list)):
		points[3*i:3*i+3,0:vnums[i]]=p
		norms[3*i:3*i+3,0:vnums[i]]=n
	return points,meanpos,norms,np.array(vnums,np.int32),corre_ps

def opt_rep_to_points(iss,ips,tarps,Rs,Ts,initRs,initTs,r_ps_w,r_ss_w,unreal_pose_w):
	ss=iss.new_zeros(iss.shape)
	ps=ips.new_zeros(ips.shape)
	tarps=tarps.detach()
	Rs=Rs.detach()
	Ts=Ts.detach().clone()
	ss=ss.copy_(iss.detach())
	ps=ps.copy_(ips.detach())
	ss.requires_grad=True
	ps.requires_grad=True
	Ts.requires_grad=True
	data_size=Ts.shape[0]
	optimizer=optim.Adam([ss,ps,Ts], lr=0.004)
	time=0
	while time<1000:
		regular_shape_loss=torch.pow(ss,2).sum()
		regular_pose_loss=pose_prior(ps).sum()	
		loss=r_ps_w*regular_pose_loss+r_ss_w*regular_shape_loss
		(ske_pfs,pfs)=bodyrep(ss,ps)
		if unreal_pose_w>0.:
			unreal_pose_loss=unrealistic_pose_loss(ske_pfs).sum()
			loss+=unreal_pose_w*unreal_pose_loss
		rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,12500)
		rec_points=RT_points_Tensor(rec_points,Rs,initRs,Ts+initTs)
		corre_loss=(rec_points-tarps).reshape(-1,3,12500).norm(2,1).mean(1).sum()
		loss+=corre_loss*1000.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if time%100==0:
			out_info='inner %d opt %.4f, p2p:%.4f, regu shape:%.4f, regu pose:%.4f'%(time,loss.item()/data_size,corre_loss.item()/data_size,regular_shape_loss.item()/data_size,regular_pose_loss.item()/data_size)
			if unreal_pose_w>0.:
				out_info+=', unreal:{:.6f}'.format(unreal_pose_loss.item()/data_size)
			out_info+=', Rangle:{:.4f}, Ts:{:.4f}'.format(Rs.norm(2,1).mean().item()/3.1415926*180.0,Ts.norm(2,1).mean().item())
			print(out_info)
		time+=1
	return ss,ps,Ts

def opt_rep_to_surface(ss,ps,tar_pt_points,tar_pt_norms,tar_pt_vnums,Rs,Ts,initRs,initTs,data_size,r_ps_w,r_ss_w,unreal_pose_w,icp_thread,icp_angle_thread):
	global tri_fs,vertex_index,face_index,checks_not_hands
	ss=ss.detach().clone()
	ps=ps.detach().clone()	
	Rs=Rs.detach().clone()
	Ts=Ts.detach().clone()
	ss.requires_grad=True
	ps.requires_grad=True
	Ts.requires_grad=True
	Rs.requires_grad=True
	optimizer=optim.Adam([ss,ps,Rs,Ts], lr=0.001)
	time=0
	while time<500:
		regular_shape_loss=torch.pow(ss,2).sum()
		regular_pose_loss=pose_prior(ps).sum()	
		loss=r_ps_w*regular_pose_loss+r_ss_w*regular_shape_loss
		(ske_pfs,pfs)=bodyrep(ss,ps)
		if unreal_pose_w>0.:
			unreal_pose_loss=unrealistic_pose_loss(ske_pfs).sum()
			loss+=unreal_pose_w*unreal_pose_loss
		rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,12500)
		rec_points=RT_points_Tensor(rec_points,Rs,initRs,Ts+initTs)

		_,knn_indexs=batch_knn_gpu_pytorch(tar_pt_points.reshape(data_size,3,tar_pt_points.shape[-1]),rec_points.reshape(data_size,3,12500).permute(1,0,2).reshape(3,-1),tar_pt_vnums,
					np.array([12500]*data_size,np.int32),1)
		knn_indexs=knn_indexs.reshape(data_size,12500)
		knn_vecs=torch.zeros(data_size*3,12500,device=device)
		knn_vecs[0::3,:]=rec_points[0::3,:]-tar_pt_points[0::3,:].gather(1,knn_indexs)
		knn_vecs[1::3,:]=rec_points[1::3,:]-tar_pt_points[1::3,:].gather(1,knn_indexs)
		knn_vecs[2::3,:]=rec_points[2::3,:]-tar_pt_points[2::3,:].gather(1,knn_indexs)
		knn_dists=knn_vecs.reshape(data_size,3,12500).norm(p=None,dim=1)
		rec_norms=U.compute_vnorms(rec_points.reshape(data_size,3,12500).permute(0,2,1),tri_fs,vertex_index,face_index).permute(0,2,1).reshape(3*data_size,12500)
		tar_corre_norms=torch.zeros(data_size*3,12500,device=device)
		check1=(knn_dists<icp_thread).to(torch.float)*checks_not_hands.reshape(1,-1)
		tar_corre_norms[0::3,:]=tar_pt_norms[0::3,:].gather(1,knn_indexs)
		tar_corre_norms[1::3,:]=tar_pt_norms[1::3,:].gather(1,knn_indexs)
		tar_corre_norms[2::3,:]=tar_pt_norms[2::3,:].gather(1,knn_indexs)
		check2=((rec_norms*tar_corre_norms).reshape(data_size,3,12500).sum(1)>math.cos(icp_angle_thread*np.pi/180.)).to(torch.float)
		valid_pair_index=check1*check2
		#p2n dists
		knn_dists=torch.abs((knn_vecs*tar_corre_norms).reshape(data_size,3,12500).sum(1))
		#can modify knn_dists as Geman_McClure_Loss
		if valid_pair_index.sum()>0:
			rec_p2p_loss=((knn_dists*valid_pair_index).sum(1)/valid_pair_index.sum(1)).sum()
		else:
			rec_p2p_loss=(knn_dists*valid_pair_index).sum()
		loss=loss+1000.*rec_p2p_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if time%50==0:
			out_info='final %d opt %.4f, p2n:%.4f, regu shape:%.4f, regu pose:%.4f'%(time,loss.item()/data_size,rec_p2p_loss.item()/data_size,regular_shape_loss.item()/data_size,regular_pose_loss.item()/data_size)
			if unreal_pose_w>0.:
				out_info+=', unreal:{:.6f}'.format(unreal_pose_loss.item()/data_size)
			out_info+=', Rangle:{:.4f}, Ts:{:.4f}'.format(Rs.norm(2,1).mean().item()/3.1415926*180.0,Ts.norm(2,1).mean().item())
			print(out_info)
		time+=1
	return ss,ps,Rs,Ts

# overlap=False
# if not overlap:
# 	valid_fs=[]
# 	for f in obj_files:
# 		if osp.isfile(osp.join(save_root,osp.splitext(osp.basename(f))[0]+'.obj')) and osp.isfile(osp.join(save_root,osp.splitext(osp.basename(f))[0]+'_wo.obj')):
# 			print('already exist, skip '+f)
# 			continue
# 		valid_fs.append(f)
# 	obj_files=valid_fs


num=len(obj_files)
print('total {} meshes...'.format(num))

batch_size=25
start_id=0
max_iters=1350
thred=0.002
log_internal=5
icp_thread=0.40
icp_angle_thread=80.


while start_id<num:
	end_id=start_id+batch_size
	if end_id>num:
		end_id=num
	tar_pt_points,tar_pt_means,tar_pt_norms,tar_pt_vnums,tar_corre_points=read_datas(obj_files[start_id:end_id])	
	tar_pt_points=torch.from_numpy(tar_pt_points).to(device)
	tar_pt_norms=torch.from_numpy(tar_pt_norms).to(device)
	tar_corre_points=torch.from_numpy(tar_corre_points).to(device)
	data_size=end_id-start_id
	ss=torch.zeros((data_size,shapenum)).to(device)
	ps=torch.zeros((data_size,posenum)).to(device)
	initRs=torch.zeros((data_size,3,3)).to(device)
	initR=cv2.Rodrigues(np.array([0,0,np.pi/2.0]))[0].dot(cv2.Rodrigues(np.array([-np.pi/2.0,0,0]))[0])
	initRs=initRs.copy_(torch.from_numpy(initR).reshape(1,3,3)).to(device)
	Rs=torch.zeros((data_size,3)).to(device)
	initTs=torch.zeros((data_size,3))
	initTs=initTs.copy_(torch.from_numpy(tar_pt_means-(initR@ref_mean_pos).reshape(1,3))).to(device)
	Ts=torch.zeros(data_size,3).to(device)
	points=torch.zeros(data_size,3,12500).to(device)
	points=points.copy_(ref_pos).reshape(-1,12500)
	ss.requires_grad=True	
	ps.requires_grad=True
	Rs.requires_grad=True
	Ts.requires_grad=True
	opt_points=False
	# points.requires_grad=True
	optimizer=optim.Adam([ss,ps,Rs,Ts], lr=0.01)
	# optimizer=optim.SGD([ss,ps,Rs,Ts], lr=0.0002)
	scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[400,600,900,1200,1400], gamma=0.4)
	rec_loss=(torch.ones((1))*100.0).to(device)
	times=0
	print('start optimize {} to {} meshes...'.format(start_id,end_id))
	while rec_loss.item()>thred and times<max_iters:
		(r_ps_w,r_ss_w,rec_corres_w,rec_pt_w,corre_c,prior_w,rt_w)=get_weight(times)
		if times in [700,1050] and opt_points:
			tss,tps,_=opt_rep_to_points(ss,ps,RT_points_Tensor(points,Rs,initRs,Ts+initTs),Rs,Ts,initRs,initTs,10,10,100)
			ss=tss.detach().clone()
			ps=tps.detach().clone()
			ss.requires_grad=True
			ps.requires_grad=True
			optimizer.add_param_group({'params':[ss,ps]})
		regular_shape_loss=torch.pow(ss,2).mean()
		regular_pose_loss=pose_prior(ps).sum()	
		# regular_pose_loss=torch.pow(ps,2).mean(1).sum()
		loss=r_ps_w*regular_pose_loss+r_ss_w*regular_shape_loss
		rot_angle_loss=Rs.norm(2,1).sum()
		tran_regu_loss=Ts.norm(2,1).sum()
		if rt_w>0.:
			loss+=rt_w*(rot_angle_loss+tran_regu_loss)

		(ske_pfs,pfs)=bodyrep(ss,ps)
		if unreal_pose_w>0.:
			unreal_pose_loss=unrealistic_pose_loss(ske_pfs).sum()
			loss+=unreal_pose_w*unreal_pose_loss
		if prior_w>0.:
			if not opt_points:
				tmp=points.new_zeros(data_size*3,12500)
				tmp=tmp.copy_(points.detach())
				points=tmp
				points.requires_grad=True
				optimizer.add_param_group({'params':points})
				opt_points=True
			prior_loss=convert_f_points.acap_prior(U.unscale_features(pfs,scales),points.reshape(data_size,3,12500).permute(0,2,1),batch_mean=True).sum()
			loss+=prior_loss*prior_w
		else:
			points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,12500)
		# rec_points=convert_f_points(U.unscale_features_perdim(pfs,scales)).permute(0,2,1).reshape(-1,12500)
		rec_points=RT_points_Tensor(points,Rs,initRs,Ts+initTs)
		if rec_corres_w>0:
			select_num=tar_corre_points.numel()//(data_size*3)
			rec_corre_loss=rec_points[:,select_landmarks[:select_num]].reshape(-1,3,select_num).permute(0,2,1).reshape(-1,3)-tar_corre_points.reshape(-1,3,select_num).permute(0,2,1).reshape(-1,3)
			rec_corre_loss=rec_corre_loss.norm(2,1)
			if corre_c>0.:
				rec_corre_loss=U.Geman_McClure_Loss(rec_corre_loss,corre_c).reshape(data_size,select_num).mean(-1).sum()
			else:
				rec_corre_loss=rec_corre_loss.reshape(data_size,select_num).mean(-1).sum()
			loss=loss+rec_corres_w*rec_corre_loss
		if rec_pt_w>0:
			_,knn_indexs=batch_knn_gpu_pytorch(tar_pt_points.reshape(data_size,3,tar_pt_points.shape[-1]),rec_points.reshape(data_size,3,12500).permute(1,0,2).reshape(3,-1),tar_pt_vnums,
					np.array([12500]*data_size,np.int32),1)
			knn_indexs=knn_indexs.reshape(data_size,12500)
			knn_vecs=torch.zeros(data_size*3,12500,device=device)
			knn_vecs[0::3,:]=rec_points[0::3,:]-tar_pt_points[0::3,:].gather(1,knn_indexs)
			knn_vecs[1::3,:]=rec_points[1::3,:]-tar_pt_points[1::3,:].gather(1,knn_indexs)
			knn_vecs[2::3,:]=rec_points[2::3,:]-tar_pt_points[2::3,:].gather(1,knn_indexs)
			knn_dists=knn_vecs.reshape(data_size,3,12500).norm(p=None,dim=1)
			rec_norms=U.compute_vnorms(rec_points.reshape(data_size,3,12500).permute(0,2,1),tri_fs,vertex_index,face_index).permute(0,2,1).reshape(3*data_size,12500)	
			tar_corre_norms=torch.zeros(data_size*3,12500,device=device)
			check1=(knn_dists<icp_thread).to(torch.float)*checks_not_hands.reshape(1,-1)
			tar_corre_norms[0::3,:]=tar_pt_norms[0::3,:].gather(1,knn_indexs)
			tar_corre_norms[1::3,:]=tar_pt_norms[1::3,:].gather(1,knn_indexs)
			tar_corre_norms[2::3,:]=tar_pt_norms[2::3,:].gather(1,knn_indexs)
			check2=((rec_norms*tar_corre_norms).reshape(data_size,3,12500).sum(1)>math.cos(icp_angle_thread*np.pi/180.)).to(torch.float)
			valid_pair_index=check1*check2
			#p2n dists
			knn_dists=torch.abs((knn_vecs*tar_corre_norms).reshape(data_size,3,12500).sum(1))
			#can modify knn_dists as Geman_McClure_Loss
			#warning!!!!!! there can be bug, if valid_pair_index is zero for some row, it will lead a / zero!!!
			if valid_pair_index.sum()>0:
				rec_p2p_loss=((knn_dists*valid_pair_index).sum(1)/valid_pair_index.sum(1)).sum()
			else:
				rec_p2p_loss=(knn_dists*valid_pair_index).sum()
			loss=loss+rec_pt_w*rec_p2p_loss
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		if times%log_internal==0:
			out_info='{}th total:{:.4f}'.format(times,loss.item()/data_size)
			if rec_corres_w>0:
				out_info+=', coore:{:.4f}'.format(rec_corre_loss.item()/data_size)
			if rec_pt_w>0:
				out_info+=', p2n:{:.4f}'.format(rec_p2p_loss.item()/data_size)
			if prior_w>0.:
				out_info+=', prior:%.4f'%(prior_loss.item()/data_size)
			if unreal_pose_w>0.:
				out_info+=', unreal:{:.8f}'.format(unreal_pose_loss.item()/data_size)
			out_info+=', regu shape:{:.4f}, regu pose:{:.4f}'.format(regular_shape_loss.item()/data_size,regular_pose_loss.item()/data_size)
			out_info+=', Rangle:{:.4f}, Ts:{:.4f}'.format(rot_angle_loss.item()/3.1415926*180.0/data_size,tran_regu_loss.item()/data_size)
			print(out_info)
		times+=1
	# (ske_pfs,pfs)=bodyrep(ss,ps)
	# rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,12500)
	rec_points=RT_points_Tensor(points,Rs,initRs,Ts+initTs)
	# RTs=np.concatenate((RTs,torch.cat((Rs,Ts+initTs),1).detach().cpu().numpy()),0)
	# rec_shapes=np.concatenate((rec_shapes,ss.detach().cpu().numpy()),0)
	# rec_poses=np.concatenate((rec_poses,ps.detach().cpu().numpy()),0)

	snames=obj_files[start_id:end_id]
	snames=[os.path.splitext(os.path.basename(name))[0] for name in snames]
	U.save_obj_files(rec_points.detach().cpu().numpy().T,convert_f_points.face_index.cpu().numpy(),save_root,[name+'.obj' for name in snames])

	tss,tps,Ts=opt_rep_to_points(ss,ps,rec_points,Rs,Ts,initRs,initTs,1,1,5.)
	tss,tps,Rs,Ts=opt_rep_to_surface(tss,tps,tar_pt_points,tar_pt_norms,tar_pt_vnums,Rs,Ts,initRs,initTs,data_size,0.1,0.1,1,0.04,45)
	(ske_pfs,pfs)=bodyrep(tss,tps)
	rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,12500)
	rec_points=RT_points_Tensor(rec_points,Rs,initRs,Ts+initTs)
	U.save_obj_files(rec_points.detach().cpu().numpy().T,convert_f_points.face_index.cpu().numpy(),save_root,[name+'_wo.obj' for name in snames])

	start_id=end_id
# with open(os.path.join(save_root,'rec_data.pkl'),'wb') as ff:
# 	pickle.dump({'RTs':RTs,'rec_shapes':rec_shapes,'rec_poses':rec_poses},ff)
print('all done')