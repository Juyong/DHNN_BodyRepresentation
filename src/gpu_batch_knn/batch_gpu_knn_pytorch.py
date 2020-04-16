# from torch.utils.cpp_extension import load
import os
# batch_knn=load(name='batch_knn',sources=[os.path.join(os.path.dirname(__file__),'BatchKnnGpu.cpp'),os.path.join(os.path.dirname(__file__),'CudaKernels.cu')])
import torch
import batch_knn
import numpy as np
#batch_pointcloud(b,n,maxPnum)
#batch_query(n,all_query_size)
def batch_knn_gpu_pytorch(batch_pointcloud,batch_query,pc_vnums,query_vnums,k=1):
	assert(type(batch_pointcloud)==torch.Tensor)
	assert(type(batch_query)==torch.Tensor)
	assert(batch_pointcloud.is_cuda)
	assert(batch_query.is_cuda)
	assert(batch_pointcloud.device==batch_query.device)
	assert(len(batch_pointcloud.shape)==3)
	assert(len(batch_query.shape)==2)
	assert(batch_pointcloud.shape[1]==batch_query.shape[0])
	pc_vnums=pc_vnums.reshape(-1)
	query_vnums=query_vnums.reshape(-1)
	assert(batch_pointcloud.shape[0]==pc_vnums.shape[0])	
	assert(pc_vnums.shape[0]==query_vnums.shape[0])
	bnum,ndim,maxPnum=batch_pointcloud.shape
	ndim,all_query_size=batch_query.shape
	assert(np.sum(query_vnums)==all_query_size)
	for vn in pc_vnums:
		assert(vn<=maxPnum)
	# start=time()
	query_to_batchIds=np.concatenate([np.array([i]*vn,dtype=np.int32) for i,vn in enumerate(query_vnums)],0)
	query_to_pc_sizes=np.concatenate([np.array([pc_vnums[i]]*vn,dtype=np.int32) for i,vn in enumerate(query_vnums)],0)
	query_to_pc_sizes=torch.from_numpy(query_to_pc_sizes).to(batch_query.device)
	query_to_batchIds=torch.from_numpy(query_to_batchIds).to(batch_query.device)
	assert(query_to_batchIds.shape[0]==all_query_size)
	assert(query_to_pc_sizes.shape[0]==all_query_size)	
	# stop=time()	
	# print('gpu time:{:.4f}'.format(stop-start))
	knn_dists,knn_indexs=batch_knn.batch_knn_gpu(batch_pointcloud,maxPnum,batch_query,all_query_size,query_to_batchIds,query_to_pc_sizes,ndim,k)
	return knn_dists,knn_indexs

def random_generate_test_data(bnum,ndim,maxPnum,maxQuerynum):
	b_pc=torch.randn(bnum,ndim,maxPnum)*2.
	pc_vnums=np.random.randint(max(maxPnum//2,1),maxPnum,bnum)
	query_vnums=np.random.randint(max(maxQuerynum//2,1),maxQuerynum,bnum)
	b_query=torch.randn(ndim,query_vnums.sum())*2.
	return {'b_pc':b_pc,'pc_vnums':pc_vnums,'b_query':b_query,'query_vnums':query_vnums}
import pickle
def save_test_data(name,data_dict):
	with open(name,'wb') as f:
		pickle.dump(data_dict,f)

def read_test_data(name):
	with open(name,'rb') as f:
		return pickle.load(f)

def compute_groundtruth_on_cpu(data_dict,k=1):
	b_pc=data_dict['b_pc']
	pc_vnums=data_dict['pc_vnums']
	b_query=data_dict['b_query']
	query_vnums=data_dict['query_vnums']
	knn_dists=torch.zeros(query_vnums.sum(),k)
	knn_indexs=torch.zeros(query_vnums.sum(),k,dtype=torch.long)
	for bid in range(b_pc.shape[0]):
		print('{:d}th data start...'.format(bid))
		pc=b_pc[bid,:,0:pc_vnums[bid]]
		q_s=query_vnums[0:bid].sum()
		q_e=query_vnums[0:bid+1].sum()
		query=b_query[:,q_s:q_e]
		dists=torch.zeros(q_e-q_s,pc_vnums[bid])
		for qid in range(query.shape[1]):
			dists[qid,:]=(query[:,qid].view(-1,1)-pc).norm(p=2,dim=0)
		dists,indexs=torch.sort(dists)
		knn_dists[q_s:q_e,0:k]=dists[:,0:k]
		knn_indexs[q_s:q_e,0:k]=indexs[:,0:k]
	return {'knn_dists':knn_dists,'knn_indexs':knn_indexs}

def check_result(knn_dists,knn_indexs,gt_dict):
	knn_dists,indexs=torch.sort(knn_dists)
	knn_indexs=torch.gather(knn_indexs,1,indexs)
	temp=knn_dists-gt_dict['knn_dists'][:,0:knn_dists.shape[1]]
	nonzero_indexs=(temp.abs()>0.0001).nonzero()
	if nonzero_indexs.numel()>0:
		print('{:d} not equal data:'.format(nonzero_indexs.numel()))
		for index in nonzero_indexs:
			print('{:d},{:d}:{:4f} {:4f}; {:d} {:d}'.format(index[0],index[1],knn_dists[index[0],index[1]],gt_dict['knn_dists'][index[0],index[1]],knn_indexs[index[0],index[1]],gt_dict['knn_indexs'][index[0],index[1]]))
	max_d=temp.abs().max()
	mean_d=temp.abs().mean()
	temp=knn_indexs-gt_dict['knn_indexs'][:,0:knn_indexs.shape[1]]
	sum_i=(temp!=0).sum()
	return max_d,mean_d,sum_i

import os.path as path
from time import time
if __name__=='__main__':
	device_id=3
	#check computation correct
	test_root='test_data/'
	name_base=test_root+'permute_data'
	if path.isfile(name_base+'.pkl') and path.isfile(name_base+'_gt.pkl'):
		print('read '+name_base+' data.')
		permute_data=read_test_data(name_base+'.pkl')
		permute_data_gt=read_test_data(name_base+'_gt.pkl')
	else:
		print('generate '+name_base+' data.')
		b_pc=torch.randn(20,3,5000)*2.
		pc_vnums=np.random.randint(max(5000//2,1),5000,20)
		query_vnums=np.array([1000]*20,dtype=np.int64)
		b_query=torch.randn(3,query_vnums.sum())*2.
		permute_data={'b_pc':b_pc,'pc_vnums':pc_vnums,'b_query':b_query,'query_vnums':query_vnums}
		save_test_data(name_base+'.pkl',permute_data)
		permute_data_gt=compute_groundtruth_on_cpu(permute_data,5)
		save_test_data(name_base+'_gt.pkl',permute_data_gt)


	name_base=test_root+'small_data'
	if path.isfile(name_base+'.pkl') and path.isfile(name_base+'_gt.pkl'):
		print('read '+name_base+' data.')
		small_data=read_test_data(name_base+'.pkl')
		small_data_gt=read_test_data(name_base+'_gt.pkl')
	else:
		print('generate '+name_base+' data.')
		small_data=random_generate_test_data(10,3,2000,100)
		save_test_data(name_base+'.pkl',small_data)
		small_data_gt=compute_groundtruth_on_cpu(small_data,100)
		save_test_data(name_base+'_gt.pkl',small_data_gt)

	name_base=test_root+'media_data'
	if path.isfile(name_base+'.pkl') and path.isfile(name_base+'_gt.pkl'):
		print('read '+name_base+' data.')
		media_data=read_test_data(name_base+'.pkl')
		media_data_gt=read_test_data(name_base+'_gt.pkl')
	else:
		print('generate '+name_base+' data.')
		media_data=random_generate_test_data(50,3,8000,2000)
		save_test_data(name_base+'.pkl',media_data)
		media_data_gt=compute_groundtruth_on_cpu(media_data,500)
		save_test_data(name_base+'_gt.pkl',media_data_gt)

	name_base=test_root+'big_data'
	if path.isfile(name_base+'.pkl') and path.isfile(name_base+'_gt.pkl'):
		print('read '+name_base+' data.')
		big_data=read_test_data(name_base+'.pkl')
		big_data_gt=read_test_data(name_base+'_gt.pkl')
	else:
		print('generate '+name_base+' data.')
		big_data=random_generate_test_data(100,3,50000,10000)
		save_test_data(name_base+'.pkl',big_data)
		big_data_gt=compute_groundtruth_on_cpu(big_data,100)
		save_test_data(name_base+'_gt.pkl',big_data_gt)
	print('read data done.')

	device=torch.device(device_id)

	b_pc=permute_data['b_pc'].to(device)
	pc_vnums=permute_data['pc_vnums']
	b_query=permute_data['b_query'].to(device)
	query_vnums=permute_data['query_vnums']
	print('start compute permute data with k 1...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,1)	
	stop=time()
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,permute_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	qvnum=b_query.shape[1]//b_pc.shape[0]
	bnum=b_pc.shape[0]
	b_pc_permute=torch.zeros(bnum*3,b_pc.shape[-1]).to(device)
	b_query_permute=torch.zeros(bnum*3,qvnum).to(device)
	for i in range(bnum):
		b_pc_permute[3*i:3*i+3,:]=b_pc[i,:,:]
		b_query_permute[3*i:3*i+3,:]=b_query[:,i*qvnum:(i+1)*qvnum]
	print('start compute permute data after permute with k 1...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc_permute.reshape(bnum,3,-1),b_query_permute.reshape(bnum,3,qvnum).permute(1,0,2).reshape(3,-1),pc_vnums,query_vnums,1)	
	stop=time()
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,permute_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	b_pc=small_data['b_pc'].to(device)
	pc_vnums=small_data['pc_vnums']
	b_query=small_data['b_query'].to(device)
	query_vnums=small_data['query_vnums']
	print('start compute small data with k 1...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,1)	
	stop=time()
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,small_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	print('start compute small data with k 20...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,20)
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	stop=time()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,small_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	b_pc=media_data['b_pc'].to(device)
	pc_vnums=media_data['pc_vnums']
	b_query=media_data['b_query'].to(device)
	query_vnums=media_data['query_vnums']
	print('start compute media data with k 1...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,1)
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	stop=time()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,media_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	print('start compute media data with k 20...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,20)
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	stop=time()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,media_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	b_pc=big_data['b_pc'].to(device)
	pc_vnums=big_data['pc_vnums']
	b_query=big_data['b_query'].to(device)
	query_vnums=big_data['query_vnums']
	print('start compute big data with k 1...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,1)
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	stop=time()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,big_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	print('start compute big data with k 20...')
	start=time()
	knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,20)
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()
	stop=time()
	print('elapse time:{:.4f}.'.format(stop-start))
	max_d,mean_d,sum_i=check_result(knn_dists,knn_indexs,big_data_gt)
	print('max delta d:{:.4f}, mean delta d:{:.4f}, nonzero indexs:{:d}'.format(max_d,mean_d,sum_i))

	#test computation time...
	device=torch.device(device_id)
	compute_time=5
	batch=100
	b_pc=(torch.randn(batch,3,10000)*.2).to(device)
	b_query=(torch.randn(3,2500*batch)*.2).to(device)
	pc_vnums=np.array([10000]*batch,dtype=np.int32)
	query_vnums=np.array([2500]*batch,dtype=np.int32)
	print('compute 2500,10000,{:d}batch k=1...'.format(batch))
	start=time()
	for i in range(compute_time):
		knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,1)
	stop=time()
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()	
	print('mean elapse time:{:.4f}'.format((stop-start)/float(compute_time)))

	batch=100
	b_pc=(torch.randn(batch,3,50000)*.2).to(device)
	b_query=(torch.randn(3,12500*batch)*.2).to(device)
	pc_vnums=np.array([50000]*batch,dtype=np.int32)
	query_vnums=np.array([12500]*batch,dtype=np.int32)
	print('compute 12500,50000,{:d}batch k=1...'.format(batch))
	start=time()
	for i in range(compute_time):
		knn_dists,knn_indexs=batch_knn_gpu_pytorch(b_pc,b_query,pc_vnums,query_vnums,1)
	stop=time()
	print('mean elapse time:{:.4f}'.format((stop-start)/float(compute_time)))
	knn_dists=knn_dists.cpu()
	knn_indexs=knn_indexs.cpu()		


	print('done.')


