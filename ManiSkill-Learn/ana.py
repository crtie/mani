import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import sys
sys.path.append("/home/ruichentie/mani/ChamferDistancePytorch")
import chamfer3D.dist_chamfer_3D
obs0=np.loadtxt('obs0.txt')
obs1=np.loadtxt('obs1.txt')
obs2=np.loadtxt('obs2.txt')
nextobs0=np.loadtxt('nextobs0.txt')
nextobs1=np.loadtxt('nextobs1.txt')
nextobs2=np.loadtxt('nextobs2.txt')

predobs0=np.loadtxt('predobs0.txt')
predobs1=np.loadtxt('predobs1.txt')
predobs2=np.loadtxt('predobs2.txt')
rew=np.loadtxt('rew.txt')
seg=np.loadtxt('seg.txt')
print(seg.shape)
seg1=np.sum(seg,axis=0)
print(seg1)
obs=np.stack((obs0,obs1,obs2),axis=2)
predobs=np.stack((predobs0,predobs1,predobs2),axis=2)
nextobs=np.stack((nextobs0,nextobs1,nextobs2),axis=2)

obs=torch.from_numpy(obs).cuda()
nextobs=torch.from_numpy(nextobs).cuda()
predobs=torch.from_numpy(predobs).cuda()

# for i in range(30,50):
#     index=i
#     fig=plt.figure(dpi=300)
#     ax=fig.add_subplot(111,projection='3d')
#     # ax.scatter(obs0[index],obs1[index],obs2[index],linewidth=0)
#     ax.scatter(nextobs0[index],nextobs1[index],nextobs2[index],linewidth=0)
#     ax.scatter(predobs0[index],predobs1[index],predobs2[index],linewidth=0)


#     ax.view_init(elev=36, azim=-159)
#     plt.savefig(f'./visualization/{i} 1.png')
#     ax.view_init(elev=65, azim=-90)
#     plt.savefig(f'./visualization/{i} 2.png')
#     plt.close()

# plt.show()
# print(obs.shape)
print(rew)
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
dist1, dist2, idx1, idx2 = chamLoss(predobs.float(),nextobs.float())
loss3 = (torch.mean(dist1)) + (torch.mean(dist2))
print(loss3)