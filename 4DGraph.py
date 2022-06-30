import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Data=np.loadtxt('Data.txt')

dat=Data[:-2]
N=int(Data[-2])
Taille=int(Data[-1])
dat=dat.reshape(np.ones(N,dtype=int)*Taille)

a=b=c=np.linspace(-np.pi,np.pi,Taille)
A,B,C=np.meshgrid(a,b,c)

fig=plt.figure(figsize=(16,14))
ax=plt.subplot(projection='3d')


cmap=matplotlib.colors.ListedColormap(['red','yellow','green'])
boundaries=[-10,-5e-2,5e-2,10]
norm=matplotlib.colors.BoundaryNorm(boundaries,cmap.N,clip=True)

im=ax.scatter(A,B,C,s=400,alpha=0.8,c=dat,cmap=cmap,norm=norm)



fig.colorbar(im)
plt.show()