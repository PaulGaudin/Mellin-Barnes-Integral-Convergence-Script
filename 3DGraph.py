import numpy as np
import matplotlib.pyplot as plt
import matplotlib

Data=np.loadtxt('Data.txt')


dat=Data[:-2]
N=int(Data[-2])
Taille=int(Data[-1])
dat=dat.reshape(np.ones(N,dtype=int)*Taille)


a=b=np.linspace(-np.pi,np.pi,Taille)
A,B=np.meshgrid(a,b)

fig=plt.figure(figsize=(16,14))
"""
ax = plt.axes(projection='3d')
ax.plot_surface(A,B,dat, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Min')
"""
cmap=matplotlib.colors.ListedColormap(['red','yellow','green'])
boundaries=[-10,-5e-2,5e-2,10]
norm=matplotlib.colors.BoundaryNorm(boundaries,cmap.N,clip=True)


im=plt.pcolor(A, B, dat,cmap=cmap,norm=norm)
fig.colorbar(im)
plt.show()
