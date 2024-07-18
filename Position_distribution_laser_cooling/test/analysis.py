import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import norm
import matplotlib.gridspec as gridspec

pd.options.mode.chained_assignment = None  # default='warn'

Labs     =  0.005
Lfront   =  0.197
Lhex     =  0.390
Lback    =  0.127 - 0.04
Llc      =  0.15
Ldetect  =  0.60
Lsp      =  3.50 - Ldetect - Llc - Lback - Lhex - Lfront

L        = [Lfront-Labs, Lhex, Lback, Llc, Ldetect, Lsp]

df=pd.read_csv('nn1e6_nj50_s1.7_ff037_d0_67_nolc.csv')
#df=np.loadtxt('nn1e3_nj50_s5_ff0232_d2_13_nolc.csv_nj50_s5',delimiter=',',skiprows=1,usecols=[1, 2, 3, 4,5,6,7,8])
#print(df.shape)
print('Import successful')



x=df['x']
y=df['y']
z=df['z']
vx=df['vx']
vy=df['vy']
ax=df['ax']
ay=df['ay']
#xx=yy=zz=vvx=vvy=aax=aay=[0 for i in range(len(x))]
print(len(x))
for i in range(len(x)):
    x[i]=json.loads(x[i])
    y[i]=json.loads(y[i])
    z[i]=json.loads(z[i])
    vx[i]=json.loads(vx[i])
    vy[i]=json.loads(vy[i])
    ax[i]=json.loads(ax[i])
    ay[i]=json.loads(ay[i])
    if np.mod(i,1000)==0:
        print(i)
print('Successful change of class')
def xy_distribution(xpoints,ypoints,zpoints):
    #First we want the x and y points that are at a distances z. This
    #distance z is Ldetec
    Lcooling=L[0]+L[1]+L[2]+L[3]+L[4]
    n=len(zpoints)
    x=[]
    y=[]
    z=[]
    for i in range(n):
        if Lcooling in zpoints[i]:
            index=zpoints[i].index(Lcooling)
            x.append(xpoints[i][index])
            y.append(ypoints[i][index])
            z.append(zpoints[i][index])
    return x,y,z
#print(zz)

xcooling,ycooling,zcooling=xy_distribution(x,y,z)
print('Successful distribution analysis')
x_edges=np.linspace(-0.025,0.025,50)
y_edges=np.linspace(-0.025,0.025,50)
H, xedges, yedges = np.histogram2d(xcooling, ycooling, bins=[x_edges, y_edges])

print(len(zcooling))

(mu, sigma) = norm.fit(xcooling)
n,bins,patches=plt.hist(xcooling,1000,range=(-0.025,0.025))
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

(mu2, sigma2) = norm.fit(ycooling)
n2,bins2,patches2=plt.hist(ycooling,1000,range=(-0.025,0.025))

yy = ((1 / (np.sqrt(2 * np.pi) * sigma2)) *
     np.exp(-0.5 * (1 / sigma2 * (bins2 - mu2))**2))


'''
#print(len(zz))
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 2, 1])

# Create the subplots
ax_main = plt.subplot(gs[1, 0])
ax_right = plt.subplot(gs[1, 1])
ax_top = plt.subplot(gs[0, 0])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
im=ax_main.imshow(H.T, origin='lower', aspect='auto', cmap='inferno',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
fig.colorbar(im, orientation="horizontal", pad=0.2)

ax_right.plot(yy,bins2)
ax_top.plot(bins,y)
ax_top.set_xticks([])
ax_right.set_yticks([])

plt.tight_layout()
# Show the plot
plt.show()
'''
fig=plt.figure()
axs=fig.add_subplot(111)
pos=axs.imshow(H.T, origin='lower', aspect='auto', cmap='inferno',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
fig.colorbar(pos, ax=axs)

axs.set_xlabel('X Axis')
axs.set_ylabel('Y Axis')

axs.set_title('2D Heatmap')
fig.tight_layout()
plt.show()
plt.clf()



fig.savefig('s1_7_ff037_d0_67_nolc.png')

plt.clf()
plt.plot(bins, y, '--', color ='red')
plt.show()

plt.clf()
plt.plot(bins2, yy, '--', color ='red')
plt.show()

