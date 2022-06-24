import torch
import sys
import numpy as np
sys.path.append("/home/ruichentie/mani/ChamferDistancePytorch")
import chamfer3D.dist_chamfer_3D
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
points1 = torch.rand(1, 1000, 3).cuda()
points2 = torch.rand(1, 2000, 3, requires_grad=True).cuda()
dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
print(torch.mean(dist1).item(),torch.mean(dist2).item())

points1=points1.cpu().detach().numpy()
points2=points2.cpu().detach().numpy()
source_index = range(0, 1000)
target_index = range(0, 1000)
source_index, target_index = np.meshgrid(source_index, target_index)
# print(obs[0,source_index].shape)
dis = np.linalg.norm(points1[0, source_index]-points2[0, target_index], axis=2)
dis1 = dis.min(0)
dis2 = dis.min(1)
print(np.mean(dis1)**2,np.mean(dis2)**2)
# print(np.mean(dis1)+np.mean(dis2))