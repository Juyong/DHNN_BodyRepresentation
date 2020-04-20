import sys
sys.path.append('../../src')
import model as M
import util as U
import torch
import argparse
import numpy as np
import os
import os.path as osp

parser = argparse.ArgumentParser(description='transfer')
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

sours=[0,1,2,3]
tars=[4,5,6,7]
shapes=[]
poses=[]
for sid,tid in zip(sours,tars):
	sd=np.load('meshes/%d.npz'%sid)
	td=np.load('meshes/%d.npz'%tid)
	shapes.append(sd['shape'])
	poses.append(td['pose'])
shapes=torch.from_numpy(np.stack(shapes)).to(device)
poses=torch.from_numpy(np.stack(poses)).to(device)
(ske_pfs,pfs)=bodyrep(shapes,poses)
rec_points=convert_f_points(U.unscale_features(pfs,bodyrep.vs_scale)).permute(0,2,1).reshape(-1,vnum)
if not osp.isdir('results'):
	os.system('mkdir results')
U.save_obj_files(rec_points.detach().cpu().numpy().T,convert_f_points.face_index.cpu().numpy(),'results',['%d_%d.obj'%(sid,tid) for sid,tid in zip(sours,tars)])

print('all done.')