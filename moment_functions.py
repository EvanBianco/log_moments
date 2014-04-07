import numpy as np
from las import LASReader
import matplotlib.pyplot as plt
from scipy.stats import skew

well = LASReader('Panuke_B-90.las', null_subs= 0) #np.nan)

DT = well.data['DT']
RHOB = well.data['RHOB']
GR = well.data['GR']
z = well.data['DEPTH']
dz = well.step

start = 1000
stop = 25000
n = (stop-start)/1000
print n
GR = GR[start:stop]
z = z[start:stop]
ncols = 50

def moment1(GR, dz, moment=1, max_window= 820, ncols=50):
    """
    GR is log
    dz is sample rate (m)
    max_window = 1640 (m)
    moment: is the nth order moment to calculate
    """
    w = max_window / dz  # integer with log base 2 max num windows
    print w
    m = np.ones( (len(GR), ncols))
    num = int(np.log2(w))
    
    # first moment 
    for j, window in enumerate(np.logspace(0, num, ncols,
                                endpoint=True, base=2)):
        print j, window
        for i in np.arange(int(window),len(GR)-int(window)):
            m[i,j] = np.mean(GR[ i-window: i+window+1 ] )
        m[:,j] /= np.amax(m[:,j])         
    
    return m
    
GRimg = np.repeat(np.expand_dims(GR,1),ncols,1)
m1 = moment1(GR,dz, ncols)
print "finished calculating first Moment"

fig  = plt.figure(figsize = [6,12], facecolor='white')
ax1 = fig.add_axes([0.05, 0.1, 0.1, 0.85])
ax2 = fig.add_axes([0.25, 0.10, 0.2, 0.85])
ax3 = fig.add_axes([0.55, 0.10, 0.55, 0.85])

# plot log
ax1.plot(GR[::n],z[::-n],'k' )
ax1.axis('tight')

# plot image of log
ax2.imshow(GRimg, cmap='gist_earth_r')

scGR = ncols*GR/np.amax(GR)

maxpt = np.argmax(np.transpose(m1),axis=0)
maxpt = ncols*maxpt/(2**13)
plt.figure()
plt.plot(maxpt)

plt.show()

# with log on top
ax2.plot(scGR,np.arange(len(scGR)),'k')

# make the color fill clip
ax2.fill_betweenx(np.arange(len(scGR)), scGR, x2=100, color='white')
ax2.axis('off')
ax2.axis('tight')
# plot scale dependent image log
ax3.imshow(m1[::n,:], cmap='gist_earth_r')
ax3.plot(maxpt[::n],np.arange(len(maxpt[::n])),'r')
ax3.axis('off')
ax3.axis('tight')
fig.show()