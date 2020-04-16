import sys
sys.path.append('../../src')
import model as M
from model import batch_rodrigues
import util as U
import torch
from torch import optim
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
import math
import pickle
import cv2
import openmesh as om

#zeros
init_p=np.zeros(72,np.float32)
#qu tui wan shou harder pose			
init_p0=np.array([ -3.5965698 , -1.0452684 ,  1.0228496 ,  0.9382113 , -2.967554  ,
        1.158851  , -0.47977582,  0.5978575 ,  3.623389  , -1.2452182 ,
        0.8276693 ,  0.94772995, -1.9177337 , -0.5760466 ,  3.0537398 ,
        0.06562585, -0.43180752,  1.9703777 , -3.7283118 , -5.0183506 ,
        2.474941  , -0.5486858 ,  0.1184738 ,  0.6042451 ,  0.16842298,
       -0.11401533, -0.12243794, -0.74235916,  3.1015017 ,  0.87992775,
       -0.43341097, -0.3266809 , -2.3326895 ,  1.3434585 , -2.52835   ,
        0.19613329,  0.18576084,  3.4795096 , -1.8865469 , -1.3922924 ,
       -0.1858143 ,  2.3556128 ,  1.6456158 ,  2.438428  , -0.7008178 ,
        1.4544111 ,  0.19850102,  0.8253175 , -0.41703418,  0.6340501 ,
       -0.91456306,  1.117118  , -0.06071685,  1.2164707 , -0.16878422,
       -0.46295047,  0.0112942 ,  1.9305491 ,  0.9263188 ,  0.4018722 ,
       -1.0522007 ,  1.2720987 , -4.994249  , -0.92930526,  2.4944696 ,
        5.8618484 ,  0.30419856, -0.9945524 , -0.703939  ,  1.1260905 ,
        1.4927981 , -1.2389627],np.float32)
	# smplify mean
init_p1=np.array([ -2.6872869 ,  0.24806152,  0.53979474,  3.6496172 , -2.0365214 ,
        5.4008923 , -0.39970723,  0.2586176 ,  5.7718935 , -1.7248734 ,
        1.9542556 ,  1.7187555 , -0.89558864,  0.71378237,  3.6481988 ,
        0.3111647 ,  1.7293143 ,  0.4112228 , -0.22693552, -1.7575588 ,
        1.3733801 ,  0.08887531,  0.3547497 ,  0.9397675 , -0.09600753,
        1.8578084 , -0.02773498, -0.01181239,  0.58568233, -0.35902962,
       -0.20257865,  0.42106718, -1.0951831 ,  3.1789649 , -0.08759502,
       -2.6903362 ,  0.45268556,  1.8904393 , -0.6433758 ,  2.8975956 ,
        0.419504  ,  4.315125  ,  1.7942199 ,  1.274159  ,  0.15372787,
       -0.3448576 , -0.47850814,  0.15965892, -0.13379307, -0.5384903 ,
       -3.4403589 , -0.23564826, -0.18551011,  0.31759128, -1.4268105 ,
        0.21256953, -0.23832056, -1.5827223 ,  0.34569564, -2.0170712 ,
       -0.4911562 ,  0.7686065 , -1.3337756 , -1.1881784 ,  5.7489886 ,
        2.373298  , -0.91719043, -0.35715294, -5.5650883 ,  0.75898373,
        0.18198682, -2.5202062 ],np.float32)

def read_deepcut_js(folder):
	with open('DeepCut/human36M/'+folder+'/all_results.pkl','rb') as ff:
		rec_data=pickle.load(ff,encoding='bytes')
		temp=rec_data[b'crop_box']
		[crop_x,crop_y,crop_w,crop_h]=[temp[b'x'],temp[b'y'],temp[b'w'],temp[b'h']]
	est_js=np.load('DeepCut/human36M/'+folder+'/est_joints.npz')['est_joints'].transpose(2,1,0)
	est_js=est_js+np.array([crop_x,crop_y,0]).reshape(1,1,-1)
	return est_js[::INTERVAL,:].astype(np.float32)


def ProjectPointRadial(Pcames,fs,cs,ks,ps):
	assert(Pcames.shape[-1]==3)
	xs=Pcames[:,0]/Pcames[:,2]
	ys=Pcames[:,1]/Pcames[:,2]
	r2=xs*xs+ys*ys
	# proj=np.zeros((*Pcames.shape[0:-1],2))
	prox=fs[0]*(xs*(1+ks[0]*r2+ks[1]*r2*r2+ks[2]*r2*r2*r2+(ps[0]*ys+ps[1]*xs)*1.0)+ps[1]*r2)+cs[0]
	proy=fs[1]*(ys*(1+ks[0]*r2+ks[1]*r2*r2+ks[2]*r2*r2*r2+(ps[0]*ys+ps[1]*xs)*1.0)+ps[0]*r2)+cs[1]
	return torch.cat((prox.reshape(-1,1),proy.reshape(-1,1)),1)




parser = argparse.ArgumentParser(description='optimize for target meshs')
# parser.add_argument('--vgan-model',default='None',metavar='M',
#                     help='pretrained vgan model'))
parser.add_argument('--gpu-id',type=int,default=0,metavar='ID',
                    help='if cuda enable, which gpu to use')
parser.add_argument('--model',default='None',metavar='M',
                    help='body rep model')
args = parser.parse_args()

shapenum=50
posenum=72
vnum=6890

device = torch.device("cuda:"+str(args.gpu_id))
torch.cuda.set_device(args.gpu_id)

INTERVAL=120

bodyExtRep=M.initial_BodyRep('../../models/bodyextTem',shapenum,posenum,vnum).to(device)
convert_f_points=M.AcapDecodeLayer('../../models/bodyextTem',anchor_id=3500).to(device)

pnum=bodyExtRep.parts_num
scales=bodyExtRep.vs_scale
ske_scales=bodyExtRep.ps_scale

j_regressor=torch.from_numpy(np.load('cocoplus_joint_regressor.npy'))
j_regressor=j_regressor[:,:14].to(device)

save_root='bodyextrep_results'
if not os.path.isdir(save_root):
	os.system('mkdir '+save_root)
cam_fs=[1145.51133842318,1144.77392807652]
cam_cs=[514.968197319863,501.882018537695]
cam_ks=[-0.198384093827848,0.218323676298049,-0.00894780704152122]
cam_ps=[-0.00181336200488089,-0.000587205583421232]

# #c1
# weights=list(zip(np.array([45,54,63,69])*500.,
# [10,5,5,1],[100,50,10,5],[-1.,10.,1.,0.],[-500,-100,-100,-10],[120,100,50,30]))
# unreal_pose_w=1.e5
#c2
weights=list(zip(np.array([45,54,63,69])*500.,
[10,5,2,1],[1,1,5,10],[-100.,1.,0.,0.],[120,110,100,80]))
unreal_pose_w=1.e4
def get_weight(time):
	if time<100:
		return weights[0]
	elif time<250:
		return weights[1]
	elif time<500:
		return weights[2]
	else:
		return weights[3]

import openmesh as om
ref_points=om.read_trimesh('../../models/bodyextTem/template.obj').points().T
ref_joints=ref_points@j_regressor.cpu().numpy()
mean_height3d=(np.linalg.norm(ref_joints[:,9]-ref_joints[:,3])+np.linalg.norm(ref_joints[:,8]-ref_joints[:,2]))/2.0
def init_trans(tar_js):
	#for deepcut points
	# 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
	mean_height2d=((tar_js[:,9,:2]-tar_js[:,3,:2]).norm(p=2,dim=-1)+(tar_js[:,8,:2]-tar_js[:,2,:2]).norm(p=2,dim=-1))/2.0
	est_d=cam_fs[0]*mean_height3d/mean_height2d
	return est_d
def choose_two_initial(tar_js):
	# check=(tar_js[:,9,:2]-tar_js[:,8,:2]).norm(p=2,dim=-1)<1000.0
	check=(tar_js[:,9,:2]-tar_js[:,8,:2]).norm(p=2,dim=-1)<28.0
	choose_ids=torch.arange(0,tar_js.shape[0]).to(tar_js.device)
	choose_ids=choose_ids[check]
	return check,choose_ids,choose_ids.numel()>0
def reverse_rotation(rs,axisangle):
	outs=np.zeros_like(rs)
	for i,r in enumerate(rs):
		flip_r=cv2.Rodrigues(r)[0].dot(cv2.Rodrigues(axisangle)[0])
		flip_r=cv2.Rodrigues(flip_r)[0]
		outs[i,:]=flip_r.reshape(-1)
	return outs
def RT_points_Tensor(points,deltaRs,Rs,Ts):
	if deltaRs is not None:
		Rotations=batch_rodrigues(deltaRs).matmul(batch_rodrigues(Rs))
	else:
		Rotations=batch_rodrigues(Rs)
	points=torch.bmm(Rotations,points.reshape(Ts.shape[0],3,-1))+Ts.reshape(Ts.shape[0],Ts.shape[1],1)
	return points.reshape(Ts.shape[0]*3,-1)
def unreal_loss(angle):
		return torch.exp(F.relu(angle))-1
def unrealistic_pose_loss(ske_fs):
	ske_fs=U.unscale_features(ske_fs,ske_scales)
	Rs=batch_rodrigues(ske_fs.reshape(-1,9)[:,:3]).reshape(-1,pnum,3,3)
	loss=0
	rR=Rs[:,13,:].matmul(Rs[:,11,:].permute(0,2,1))
	loss+=unreal_loss(rR[:,2,0])
	rR=Rs[:,14,:].matmul(Rs[:,12,:].permute(0,2,1))
	loss+=unreal_loss(-rR[:,2,0])
	rR=Rs[:,4,:].matmul(Rs[:,1,:].permute(0,2,1))
	loss+=unreal_loss(-rR[:,2,1])
	loss+=unreal_loss(rR[:,0,1]-math.sin(3./180.*np.pi))
	# loss+=unreal_loss(-(rR[:,0,1]+math.sin(3./180.*np.pi)))
	rR=Rs[:,5,:].matmul(Rs[:,2,:].permute(0,2,1))
	loss+=unreal_loss(-rR[:,2,1])
	# loss+=unreal_loss(rR[:,0,1]-math.sin(3./180.*np.pi))
	loss+=unreal_loss(-(rR[:,0,1]+math.sin(3./180.*np.pi)))
	rR=Rs[:,11,:].matmul(Rs[:,6,:].permute(0,2,1))
	loss+=unreal_loss(rR[:,2,0]-0.1)
	loss+=unreal_loss(-(rR[:,0,0]+math.sin(10./180.*np.pi)))
	rR=Rs[:,12,:].matmul(Rs[:,7,:].permute(0,2,1))
	loss+=unreal_loss(-(rR[:,2,0]+0.1))
	loss+=unreal_loss(-(rR[:,0,0]+math.sin(10./180.*np.pi)))
	rR=Rs[:,1,:].matmul(Rs[:,0,:].permute(0,2,1))
	loss+=unreal_loss(rR[:,2,0]-0.7)
	loss+=unreal_loss(rR[:,2,1]-math.sin(8./180.*np.pi))
	loss+=unreal_loss(-(rR[:,0,2]+math.sin(65./180.*np.pi)))
	rR=Rs[:,2,:].matmul(Rs[:,0,:].permute(0,2,1))
	loss+=unreal_loss(-(rR[:,2,0]+0.7))
	loss+=unreal_loss(rR[:,2,1]-math.sin(8./180.*np.pi))
	loss+=unreal_loss(rR[:,0,2]-math.sin(65./180.*np.pi))
	rR=Rs[:,0,:].matmul(Rs[:,3,:].permute(0,2,1))
	loss+=unreal_loss(rR[:,2,1]-math.sin(8./180.*np.pi))
	return loss
def pose_prior(ps):
	# return U.Geman_McClure_Loss(F.relu(torch.abs(ps)-1.5),0.25).mean(1)
	return ps.pow(2).mean(1)

def optimize_with_initp(ori_rRs,ori_Ts,initp,ss,tar_js,data_size,max_iters,log_internal):
	rTs=torch.zeros((data_size,3)).to(device)
	rTs=rTs.copy_(ori_Ts.detach())			
	rRs=torch.zeros((data_size,3)).to(device)
	rRs.requires_grad=True
	rTs.requires_grad=True
	rss=torch.zeros((data_size,shapenum)).to(device)			
	rps=torch.from_numpy(initp.reshape(1,-1).repeat(data_size,axis=0)).to(device)
	# rps=torch.zeros((data_size,posenum)).to(device)
	rss.copy_(ss.detach())
	# rps.copy_(ps.detach())
	rss.requires_grad=True
	rps.requires_grad=True
	re_optimizer=optim.Adam([rss,rps,rRs,rTs], lr=0.01)
	re_scheduler=optim.lr_scheduler.MultiStepLR(re_optimizer,milestones=[400,600,800], gamma=0.4)
	times=0
	while times<max_iters:
		(js_w,r_ps_w,r_ss_w,r_rot_w,gmc)=get_weight(times)
		# record_pose_loss=torch.pow(rps,2).mean(1)
		record_pose_loss=pose_prior(rps)
		regular_pose_loss=record_pose_loss.sum()
		record_shape_loss=torch.pow(rss,2).mean(1)
		regular_shape_loss=record_shape_loss.sum()
		record_rot_loss=rRs.norm(2,1)
		regular_rot_loss=record_rot_loss.sum()				
		loss=max(r_ps_w,0)*regular_pose_loss+r_ss_w*regular_shape_loss
		# record_loss=max(r_ps_w,0)*record_pose_loss+r_ss_w*record_shape_loss
		if r_rot_w>0.:
			loss=loss+r_rot_w*regular_rot_loss
			# record_loss=record_loss+r_rot_w*record_rot_loss

		(ske_pfs,pfs)=bodyExtRep(rss,rps)
		if unreal_pose_w>0.:
			record_unreal_loss=unrealistic_pose_loss(ske_pfs)
			record_loss=unreal_pose_w*record_unreal_loss
			unreal_loss=record_unreal_loss.sum()
			loss=loss+unreal_pose_w*unreal_loss
		else:
			unreal_loss=torch.zeros(1,device=device)

		rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,6890)
		rec_points=RT_points_Tensor(rec_points,rRs,ori_rRs,rTs)
		extract_bjoints=rec_points.matmul(j_regressor)				
		projs=ProjectPointRadial(extract_bjoints.reshape(data_size,3,14).permute(0,2,1).reshape(-1,3),cam_fs,cam_cs,cam_ks,cam_ps)
		projs=projs.reshape(data_size,14,2)
		record_k2d_loss0=(projs-tar_js[:,:,:2]).norm(p=2,dim=-1)
		if gmc>0.:
			record_k2d_loss=(U.Geman_McClure_Loss(record_k2d_loss0,gmc)*tar_js[:,:,2]).mean(1)
		else:
			record_k2d_loss=record_k2d_loss0.mean(1)
		# record_k2d_loss=(record_k2d_loss0*tar_js[:,:,2]).mean(1)
		k2d_loss=record_k2d_loss.sum()

		loss=js_w*k2d_loss+loss
		# record_loss=record_loss+js_w*record_k2d_loss
		record_loss=record_k2d_loss
		re_optimizer.zero_grad()
		loss.backward()
		if r_rot_w<0.:
			rRs.grad.zero_()
		if r_ps_w<0.:
			rps.grad.zero_()
		re_optimizer.step()
		re_scheduler.step()
		# print('{:.6f}'.format(rec_loss.item()))
		if times%log_internal==0:
			print('{}th total mean loss:{:.6f}\nk2d_loss mean:{:.6f},regular_shape_loss:{:.6f}, regular_pose_loss:{:.6f}, unreal loss:{:.6f}'
					.format(times,loss.item()/data_size,k2d_loss.item()/data_size,regular_shape_loss.item()/data_size,regular_pose_loss.item()/data_size,
						unreal_loss.item()/data_size))
			print('Rangle mean:{:.6f}, Ts mean:{:.6f}'.format(regular_rot_loss.item()/3.1415926*180.0/data_size,rTs.norm(2,1).mean().item()))				
		times+=1
	return 	rRs,rTs,rss,rps,record_loss

def optimize_for_deepcut_j2ds(folder,js,log_internal,max_iters):
	fsave_root=os.path.join(save_root,folder+'_ms')
	# fsave_root=save_root
	if not os.path.isdir(fsave_root):
		os.makedirs(fsave_root)
	batch_size=350
	start_id=0
	fnum=js.shape[0]
	RTs=np.zeros((0,6),dtype=np.float32)
	rec_shapes=np.zeros((0,shapenum),dtype=np.float32)
	rec_poses=np.zeros((0,posenum),dtype=np.float32)
	rec_apts=np.zeros((0,6890),dtype=np.float32)
	print('start optimize '+folder+' data, with interval %d...'%INTERVAL)
	while start_id<fnum:
		end_id=start_id+batch_size
		if end_id>fnum:
			end_id=fnum
		print('{} to {} frames...'.format(start_id,end_id))
		tar_js=torch.from_numpy(js[start_id:end_id,:,:]).to(device)
		data_size=end_id-start_id
		ss=torch.zeros((data_size,shapenum)).to(device)
		
		# ps=torch.from_numpy(init_p.reshape(1,-1).repeat(data_size,axis=0)).to(device)
		ps=torch.zeros((data_size,posenum)).to(device)
		# Rs=torch.zeros((data_size,3)).to(device)
		# Rs.data.uniform_(-0.01,0.01)
		# make initial stand in front of camera
		Rs=torch.zeros((data_size,3))
		Rs=Rs.copy_(torch.from_numpy(np.array([np.pi,0,0],np.float32)).reshape(1,3)).to(device)
		
		Ts=torch.zeros((data_size,3)).to(device)
		initial_depths=init_trans(tar_js)		
		Ts[:,2]=initial_depths
		Rs.requires_grad=True
		Ts.requires_grad=True
		optimizerInitial=optim.Adam([Rs,Ts],lr=0.04)
		schedulerInitial=optim.lr_scheduler.MultiStepLR(optimizerInitial,milestones=[200], gamma=0.4)
		times=0
		print('start initial 0th cam external params...')
		(_,pfs)=bodyExtRep(ss,ps)
		initial_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,6890)
		while times<400:
			rec_points=U.RT_points_Tensor(initial_points,Rs,Ts)
			body_joints=rec_points.matmul(j_regressor)
			extract_bjoints=body_joints[:,[9,3,8,2]]
			projs=ProjectPointRadial(extract_bjoints.reshape(data_size,3,4).permute(0,2,1).reshape(-1,3),cam_fs,cam_cs,cam_ks,cam_ps)
			k2d_loss=(projs.reshape(data_size,4,2)-tar_js[:,[9,3,8,2],:2]).norm(p=2,dim=-1).mean(1).sum()
			depth_regu_loss=torch.pow(Ts[:,2]-initial_depths,2).sum()
			loss=k2d_loss+100.*depth_regu_loss
			optimizerInitial.zero_grad()
			loss.backward()
			optimizerInitial.step()
			schedulerInitial.step()
			if times%log_internal==0:
				print('{}th total mean loss:{:.6f}\nk2d_loss mean:{:.6f},depth_regu_loss:{:.6f}'.format(times,loss.item()/data_size,k2d_loss.item()/data_size,depth_regu_loss.item()/data_size))
				print('Rangle mean:{:.6f}, Ts mean:{:.6f}'.format(Rs.norm(2,1).mean().item()/3.1415926*180.0,Ts.norm(2,1).mean().item()))
			times+=1
		final_k2d_loss=torch.ones((data_size,),device=device)*10000.
		final_Ts=torch.zeros_like(Ts,device=device)
		final_Rs=torch.zeros_like(Rs,device=device)
		final_oriRs=torch.zeros_like(Rs,device=device)
		final_ss=torch.zeros_like(ss,device=device)
		final_ps=torch.zeros_like(ps,device=device)
		# offset_axisangle=[np.array([0.,0.,0.]),np.array([0.,np.pi,0.]),np.array([np.pi,0.,0.])]
		offset_axisangle=[np.array([0.,0.,0.]),np.array([0.,np.pi,0.])]
		for i,offset in enumerate(offset_axisangle):
			ori_rRs=reverse_rotation(Rs.detach().cpu().numpy().astype(np.float64),offset)
			ori_rRs=torch.from_numpy(ori_rRs.astype(np.float32)).to(device)
			ori_Ts=torch.zeros((data_size,3)).to(device)
			ori_Ts=ori_Ts.copy_(Ts.detach())
			if i>0 :				
				ori_rRs.requires_grad=True
				ori_Ts.requires_grad=True
				optimizerInitial=optim.Adam([ori_rRs,ori_Ts],lr=0.04)
				schedulerInitial=optim.lr_scheduler.MultiStepLR(optimizerInitial,milestones=[200], gamma=0.4)
				times=0
				print('start initial %dth cam external params...'%i)
				while times<400:
					rec_points=U.RT_points_Tensor(initial_points,ori_rRs,ori_Ts)
					body_joints=rec_points.matmul(j_regressor)
					extract_bjoints=body_joints[:,[9,3,8,2]]
					projs=ProjectPointRadial(extract_bjoints.reshape(data_size,3,4).permute(0,2,1).reshape(-1,3),cam_fs,cam_cs,cam_ks,cam_ps)
					k2d_loss=(projs.reshape(data_size,4,2)-tar_js[:,[9,3,8,2],:2]).norm(p=2,dim=-1).mean(1).sum()
					depth_regu_loss=torch.pow(ori_Ts[:,2]-Ts[:,2],2).sum()
					loss=k2d_loss+100.*depth_regu_loss
					optimizerInitial.zero_grad()
					loss.backward()
					optimizerInitial.step()
					schedulerInitial.step()
					if times%log_internal==0:
						print('{}th total mean loss:{:.6f}\nk2d_loss mean:{:.6f},depth_regu_loss:{:.6f}'.format(times,loss.item()/data_size,k2d_loss.item()/data_size,depth_regu_loss.item()/data_size))
						print('Rangle mean:{:.6f}, Ts mean:{:.6f}'.format(ori_rRs.norm(2,1).mean().item()/3.1415926*180.0,ori_Ts.norm(2,1).mean().item()))
					times+=1
				ori_rRs=ori_rRs.detach()
				ori_Ts=ori_Ts.detach()
			# for pose_ind,initp in enumerate([init_p0,init_p1]):
			for pose_ind,initp in enumerate([init_p0,init_p1]):
				print('start optimize with pose {:d} {:d}th initial rotation...'.format(pose_ind,i))
				rRs,rTs,rss,rps,record_loss=optimize_with_initp(ori_rRs,ori_Ts,initp,ss,tar_js,data_size,max_iters,log_internal)
				# re_k2d_loss=100.*re_k2d_loss+re_pose_loss
				temp_check=(final_k2d_loss>record_loss)
				final_Rs[temp_check,:]=rRs[temp_check,:]
				final_oriRs[temp_check,:]=ori_rRs[temp_check,:]
				final_Ts[temp_check,:]=rTs[temp_check,:]
				final_ss[temp_check,:]=rss[temp_check,:]
				final_ps[temp_check,:]=rps[temp_check,:]
				final_k2d_loss[temp_check]=record_loss[temp_check]

				# (ske_sfs,sfs),(ske_pfs,pfs)=sp_vae.decoder(rss,rps)
				# rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,6890)
				# rec_points=RT_points_Tensor(rec_points,rRs,ori_rRs,rTs)
				# temp_save_root=os.path.join(save_root,folder+'_ms{:d}_{:d}'.format(pose_ind,i))
				# if not os.path.isdir(temp_save_root):
				# 	os.mkdir(temp_save_root)
				# snames=[str(name) for name in np.arange(start_id,end_id)*INTERVAL]
				# D.save_obj_files(rec_points.detach().cpu().numpy().T,ref_tris,temp_save_root,[name+'.obj' for name in snames])
		(ske_pfs,pfs)=bodyExtRep(final_ss,final_ps)
		rec_points=convert_f_points(U.unscale_features(pfs,scales)).permute(0,2,1).reshape(-1,6890)
		rec_points=RT_points_Tensor(rec_points,final_Rs,final_oriRs,final_Ts)
		RTs=np.concatenate((RTs,torch.cat((final_Rs,final_Ts),1).detach().cpu().numpy()),0)
		rec_shapes=np.concatenate((rec_shapes,final_ss.detach().cpu().numpy()),0)
		rec_poses=np.concatenate((rec_poses,final_ps.detach().cpu().numpy()),0)
		rec_apts=np.concatenate((rec_apts,rec_points.detach().cpu().numpy()),0)
		snames=[str(name) for name in np.arange(start_id,end_id)*INTERVAL]
		U.save_obj_files(rec_points.detach().cpu().numpy().T,convert_f_points.face_index.cpu().numpy(),fsave_root,[name+'.obj' for name in snames])
		start_id=end_id
	with open(os.path.join(fsave_root,'rec_data.pkl'),'wb') as ff:
		pickle.dump({'RTs':RTs,'rec_shapes':rec_shapes,'rec_poses':rec_poses,'rec_points':rec_apts,'folder':folder,'frames':np.arange(fnum)*INTERVAL},ff)




rec_scene_names=['S9_SittingDown']
for folder in rec_scene_names:
	est_js=read_deepcut_js(folder)
	optimize_for_deepcut_j2ds(folder,est_js,50,800)
print('all done.')


