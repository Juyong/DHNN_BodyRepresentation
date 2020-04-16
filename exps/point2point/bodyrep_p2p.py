import sys
sys.path.append('../../src')
import model as M
import util as U
# import BodyDeform as D
import argparse
import torch
from torch import optim
import numpy as np
import os
import os.path as osp
from glob import glob
import pickle
parser = argparse.ArgumentParser(description='optimize for consistent connectivity meshs')
parser.add_argument('--gpu-id',type=int,default=0,metavar='ID',
                    help='if cuda enable, which gpu to use')
args = parser.parse_args()

shapenum=50
posenum=72
vnum=12500
torch.manual_seed(10000)
device = torch.device("cuda:"+str(args.gpu_id))
torch.cuda.set_device(args.gpu_id)

bodyrep=M.initial_BodyRep('../../models/bodyTem',shapenum,posenum,vnum).to(device)
convert_f_points=M.AcapDecodeLayer('../../models/bodyTem',anchor_id=6263).to(device)

names=glob('bodyrep_data/*.obj')
save_root='bodyrep_rec'
if not os.path.isdir(save_root):
	os.system('mkdir '+save_root)

num=len(names)
batch_size=200
start_id=0
thred=0.001
max_iters=2500
log_internal=50
rec_weight=1.e6
# r_ss_w=100.0
# r_ps_w=1.e6
weights=list(zip([100,50,10,1],
[10,5,1,0],))
def get_weight(time):
	if time<200:
		return weights[0]
	elif time<800:
		return weights[1]
	elif time<1600:
		return weights[2]
	else:
		return weights[3]
print('total {} meshes...'.format(num))
while start_id<num:
	end_id=start_id+batch_size
	if end_id>num:
		end_id=num
	tar_points=U.read_target_points(names[start_id:end_id])
	tar_points=torch.from_numpy(tar_points).to(device)
	data_size=end_id-start_id
	ss=torch.zeros((data_size,shapenum)).to(device)
	ps=torch.zeros((data_size,posenum)).to(device)
	Rs=torch.zeros((data_size,3))
	Rs.data.uniform_(-0.01,0.01)
	Rs=Rs.to(device)
	Ts=torch.zeros((data_size,3)).to(device)
	ss.requires_grad=True
	ps.requires_grad=True
	Rs.requires_grad=True
	Ts.requires_grad=True
	optimizer=optim.Adam([ss,ps,Rs,Ts], lr=0.04)
	scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[800,1600,2100], gamma=0.4)
	rec_loss=(torch.ones((1))*100.0).to(device)
	times=0
	print('start optimize {} to {} meshes...'.format(start_id,end_id))
	while rec_loss.item()>thred and times<max_iters:
		(r_ps_w,r_ss_w)=get_weight(times)
		(ske_pfs,pfs)=bodyrep(ss,ps)
		rec_points=convert_f_points(U.unscale_features(pfs,bodyrep.vs_scale)).permute(0,2,1).reshape(-1,vnum)
		rec_points=U.RT_points_Tensor(rec_points,Rs,Ts)
		rec_loss=U.GeometryEqualLoss(rec_points,tar_points)
		regular_shape_loss=torch.pow(ss,2).mean()
		regular_pose_loss=torch.pow(ps,2).mean()		
		loss=rec_weight*rec_loss+r_ss_w*regular_shape_loss+r_ps_w*regular_pose_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		if times%log_internal==0:
			print('{}th total mean loss:{:.6f}'.format(times,loss.item()))
			print('rec_loss mean:{:.6f}'.format(rec_loss.item()))
			print('regular_shape_loss:{:.6f}, regular_pose_loss:{:.6f}'.format(regular_shape_loss.item(),regular_pose_loss.item()))
			print('Rangle mean:{:.6f}, Ts mean:{:.6f}'.format(Rs.norm(2,1).mean().item()/3.1415926*180.0,Ts.norm(2,1).mean().item()))
		times+=1
	(ske_pfs,pfs)=bodyrep(ss,ps)
	rec_points=convert_f_points(U.unscale_features(pfs,bodyrep.vs_scale)).permute(0,2,1).reshape(-1,vnum)
	rec_points=U.RT_points_Tensor(rec_points,Rs,Ts)

	snames=names[start_id:end_id]
	snames=[osp.splitext(osp.basename(name))[0] for name in snames]
	U.save_obj_files(rec_points.detach().cpu().numpy().T,convert_f_points.face_index.detach().cpu().numpy(),save_root,[name+'.obj' for name in snames])

	start_id=end_id
print('all done.')
