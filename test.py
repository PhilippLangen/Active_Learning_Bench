import numpy as np
from scipy import spatial

np.random.seed(2)
u_idx = np.arange(0,10)
s_idx = np.random.choice(u_idx,3,replace=False)
u_idx = np.setdiff1d(u_idx,s_idx)

print(u_idx)
print(s_idx)
dat = np.random.randint(0,10,(10,2))
print(dat)

u_dat = dat[u_idx] #u_dat , s_dat are in modified index order
s_dat = dat[s_idx]
print(u_dat)
print(s_dat)
# create distance matrix axis 0 is udat 1 is sdat
dist = spatial.distance.cdist(u_dat,s_dat)
print(dist)
# for each unsampled date find min distance to labelled
min_dist = np.min(dist, axis=1)
print(min_dist)

budget = 7
for i in range(budget):
    idx = np.argmax(min_dist) #find index of largest min_dist entry
    add_sample = u_idx[idx] #min_dist is in index modified order, so to find original order index use index list
    move_dat = u_dat[idx] #get data of unsampled date we have chosen for sampling
    u_idx = np.delete(u_idx, idx)   # we delete this index out of u_idx and add it to s_idx this changes modified idx order
    s_idx = np.append(s_idx, add_sample)
    u_dat = np.delete(u_dat, idx, axis=0) # we also delete the data row, this makes same change to modified idx order balancing out
    min_dist = np.delete(min_dist, idx, axis=0) #Finally delete out of distance list, same change to mod idx order

    print(min_dist)

    # now we need to see if sampling has minimized any min distances to labelled samples
    min_dist=\
        np.min(     # finally calc min over 2 values
            np.append(
                np.reshape(min_dist,(-1,1)),    # second compare to old min dist values shape(x,1) -> (x,2)
                spatial.distance.cdist(u_dat, np.reshape(move_dat, (1, -1))),   #first calc distance from unlabelled to new sample
                axis=1),
            axis=1)

    print(min_dist)

    print(add_sample)