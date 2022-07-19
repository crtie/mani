
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

obs0 = np.loadtxt('visualization/obs0.txt')
obs1 = np.loadtxt('visualization/obs1.txt')
obs2 = np.loadtxt('visualization/obs2.txt')
seg0 = np.loadtxt('visualization/seg0.txt')
seg1 = np.loadtxt('visualization/seg1.txt')
seg2 = np.loadtxt('visualization/seg2.txt')

# nextobs0 = np.loadtxt('visualization/nextobs0.txt')
# nextobs1 = np.loadtxt('visualization/nextobs1.txt')
# nextobs2 = np.loadtxt('visualization/nextobs2.txt')
# nextseg0 = np.loadtxt('visualization/nextseg0.txt')
# nextseg1 = np.loadtxt('visualization/nextseg1.txt')
# nextseg2 = np.loadtxt('visualization/nextseg2.txt')


nextobs_computed0 = np.loadtxt('visualization/next_obs_computed0.txt')
nextobs_computed1 = np.loadtxt('visualization/next_obs_computed1.txt')
nextobs_computed2 = np.loadtxt('visualization/next_obs_computed2.txt')
nextseg_computed0 = np.loadtxt('visualization/computed_nextseg0.txt')
nextseg_computed1 = np.loadtxt('visualization/computed_nextseg1.txt')
nextseg_computed2 = np.loadtxt('visualization/computed_nextseg2.txt')
predobs0 = np.loadtxt('visualization/predobs0.txt')
predobs1 = np.loadtxt('visualization/predobs1.txt')
predobs2 = np.loadtxt('visualization/predobs2.txt')


obs = np.stack((obs0, obs1, obs2), axis=2)
predobs = np.stack((predobs0, predobs1, predobs2), axis=2)
# nextobs = np.stack((nextobs0, nextobs1, nextobs2), axis=2)
next_obs_computed = np.stack((nextobs_computed0, nextobs_computed1, nextobs_computed2), axis=2)


for i in range(66,99):
    index = i
    computed_next_obs_seg0 = np.where(nextseg_computed0[i]==True)[0]
    computed_next_obs_seg1 = np.where(nextseg_computed1[i]==True)[0]
    computed_next_obs_seg2 = np.where(nextseg_computed2[i]==True)[0]
    # next_obs_seg0 = np.where(nextseg0[i]==True)[0]
    # next_obs_seg1 = np.where(nextseg1[i]==True)[0]
    # next_obs_seg2 = np.where(nextseg2[i]==True)[0]
    obs_seg0 = np.where(seg0[i]==True)[0]
    obs_seg1 = np.where(seg1[i]==True)[0]
    obs_seg2 = np.where(seg2[i]==True)[0]
    # print(np.sum(np.abs(next_obs_computed[index][computed_next_obs_seg1]-obs[index][obs_seg1])))
    # print(np.sum(np.abs(next_obs_computed[index][computed_next_obs_seg2]-obs[index][obs_seg2])))

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_title(f'pic {i}') 
    # print(i)
    ax.scatter(obs[index, :, 0], obs[index, :, 1],
               obs[index, :, 2], c='b', label='obs')
    # ax.scatter(nextobs[index, :, 0], nextobs[index, :, 1],
    #            nextobs[index, :, 2], c='r', label="next")
    ax.scatter(predobs[index, :, 0], predobs[index, :, 1],
               predobs[index, :, 2], c='g', label="pred")
    ax.scatter(next_obs_computed[index, :, 0], next_obs_computed[index, :, 1],
               next_obs_computed[index, :, 2], c='m', label="next_computed")
    # print(next_obs_computed[index][computed_next_obs_seg0].shape)

    # ax.scatter(next_obs_computed[index][computed_next_obs_seg1][:,0], next_obs_computed[index][computed_next_obs_seg1][:,1],
    #            next_obs_computed[index][computed_next_obs_seg1][:,2], c='b', label="next_computed")
    # ax.scatter(next_obs_computed[index][computed_next_obs_seg2][:,0], next_obs_computed[index][computed_next_obs_seg2][:,1],
    #            next_obs_computed[index][computed_next_obs_seg2][:,2], c='r', label="next_computed")
    # ax.scatter(next_obs_computed[index][computed_next_obs_seg0][:,0], next_obs_computed[index][computed_next_obs_seg0][:,1],
    #            next_obs_computed[index][computed_next_obs_seg0][:,2], c='g', label="next_computed")

    # ax.scatter(nextobs[index][next_obs_seg0][:,0], nextobs[index][next_obs_seg0][:,1],
    #            nextobs[index][next_obs_seg0][:,2], c='b', label="next")
    # ax.scatter(nextobs[index][next_obs_seg1][:,0], nextobs[index][next_obs_seg1][:,1],
    #            nextobs[index][next_obs_seg1][:,2], c='g', label="next")
    # ax.scatter(nextobs[index][next_obs_seg2][:,0], nextobs[index][next_obs_seg2][:,1],
    #            nextobs[index][next_obs_seg2][:,2], c='m', label="next")

    # ax.scatter(obs[index][obs_seg2][:,0], obs[index][obs_seg2][:,1],
    #            obs[index][obs_seg2][:,2], c='r', label="obs")
    # ax.scatter(obs[index][obs_seg1][:,0], obs[index][obs_seg1][:,1],
    #            obs[index][obs_seg1][:,2], c='r', label="obs")
    # ax.scatter(obs[index][obs_seg0][:,0], obs[index][obs_seg0][:,1],
    #            obs[index][obs_seg0][:,2], c='b', label="obs")
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    tit= "frame "+str(i)
    print(tit)
    ax.legend(loc='best')

    plt.show()
    plt.close()

