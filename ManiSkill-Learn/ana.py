
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
print("hello")
import sys
sys.path.append("/home/ruichentie/mani/ChamferDistancePytorch")
import chamfer3D.dist_chamfer_3D

obs0=np.loadtxt('visualization/obs0.txt')
obs1=np.loadtxt('visualization/obs1.txt')
obs2=np.loadtxt('visualization/obs2.txt')
nextobs0=np.loadtxt('visualization/nextobs0.txt')
nextobs1=np.loadtxt('visualization/nextobs1.txt')
nextobs2=np.loadtxt('visualization/nextobs2.txt')

predobs0=np.loadtxt('visualization/predobs0.txt')
predobs1=np.loadtxt('visualization/predobs1.txt')
predobs2=np.loadtxt('visualization/predobs2.txt')
rew=np.loadtxt('visualization/rew.txt')
seg=np.loadtxt('visualization/seg.txt')
print(seg.shape)

obs=np.stack((obs0,obs1,obs2),axis=2)
predobs=np.stack((predobs0,predobs1,predobs2),axis=2)
nextobs=np.stack((nextobs0,nextobs1,nextobs2),axis=2)



# for i in range(50):
#     index=i
#     fig=plt.figure(dpi=300)
#     ax=fig.add_subplot(111,projection='3d')
#     # ax.scatter(obs0[index],obs1[index],obs2[index],linewidth=0)
#     ax.scatter(nextobs0[index],nextobs1[index],nextobs2[index],linewidth=0,label="next")
#     ax.scatter(predobs0[index],predobs1[index],predobs2[index],linewidth=0,label="pred")


#     ax.view_init(elev=36, azim=-159)
#     plt.savefig(f'./visualization/{i} 1.png')
#     ax.view_init(elev=65, azim=-90)
#     plt.savefig(f'./visualization/{i} 2.png')
#     plt.close()

# plt.show()
# print(obs.shape)

# source_index = range(0, 1200)
# target_index = range(0, 1200)
# source_index, target_index = np.meshgrid(source_index, target_index)
# print(obs[0,source_index].shape)
# dis = np.linalg.norm(-predobs[0, source_index]+nextobs[0, target_index], axis=2)
# dis1 = dis.min(0)
# dis2 = dis.min(1)
# print(np.mean(dis1),np.mean(dis2))
# print(np.mean(dis1)+np.mean(dis2))

obs=torch.from_numpy(obs).cuda()
nextobs=torch.from_numpy(nextobs).cuda()
predobs=torch.from_numpy(predobs).cuda()

print(rew)
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
for i in range(99):
    print(f'idx {i}')
    dist1, dist2, idx1, idx2 = chamLoss(predobs[i:i+1].float(),nextobs[i:i+1].float())
    loss3 = (torch.mean(dist1)) + (torch.mean(dist2))
    # print(torch.mean(dist1).item(),torch.mean(dist2).item())
    print(f'dist pred to next is {loss3}')

    dist1, dist2, idx1, idx2 = chamLoss(obs[i:i+1].float(),nextobs[i:i+1].float())
    loss3 = (torch.mean(dist1)) + (torch.mean(dist2))
    # print(torch.mean(dist1).item(),torch.mean(dist2).item())
    print(f'dist obs to next is {loss3}')
    print('')
