import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def Gauss(x, a, sigma):
    return a*np.exp(-(x)**2/(2*sigma**2))
xbeam    =  4.    # size laserbeam
X=[-0.5*xbeam,0,0.5*xbeam]
Y=[1/np.e**2,1,1/np.e**2]
par,cov=curve_fit(Gauss, X, Y)
Labs     =  0.005
Lfront   =  0.197
Lhex     =  0.390
Lback    =  0.127 - 0.04
Llc      =  0.15
Ldetect  =  0.60
Lsp      =  3.50 - Ldetect - Llc - Lback - Lhex - Lfront

L        = [Lfront-Labs, Lhex, Lback, Llc, Ldetect, Lsp]

L1=L[0]+L[1]+L[2]
L2=L[0]+L[1]+L[2]+L[3]
Lc_region=L2-L1
centers=[Lc_region*float(i+1)/float(18) for i in range(18)]
z=0
z_positions=[]
hit=True
intensity=[]
print(Lc_region)
while((0 <= z < Lc_region) and hit == True):
        distance_center=abs(z- min(centers, key=lambda x:abs(x-z)))
        alpha0=Gauss(distance_center*1e3,*par)
        z_positions.append(z)
        intensity.append(alpha0)
        z+=0.000001
plt.scatter(centers,np.ones(len(centers)))
plt.plot(z_positions,intensity)
plt.plot(centers,[1/np.e**2 for _ in centers])
plt.show()