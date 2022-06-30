import numpy as np
from scipy.optimize import minimize
import joblib
import timeit

N=2
"""
Instructions : 
Taille : Number of steps for the parameters (which go between -pi and pi)
params1 : n and m appearing in the PochHammer in the NUMERATOR
params2 : n and m... appearing in the PochHammer in the DENOMINATOR
NbCoeur : Number of hearts used for the multi-threading (default : Number of heart in your processor - 2)
"""

Taille=1000
#Example : Convergence of the Mellin-Barnes representation of the F4 Appell function
params1=[(1,1),(1,1),(1,0),(0,1)]
params2=[(1,0),(0,1)]
NbCoeur=joblib.cpu_count()-2


#TIMER
def timer(function):
    def inner(*args, **kwargs):
        start = timeit.default_timer()
        result = function(*args, **kwargs)
        end = timeit.default_timer()
        time = end - start
        print(f"Function {function.__name__} executed in {time} seconds.")
        return result
        
    return inner

a=np.linspace(-np.pi,np.pi,Taille)
b=np.linspace(-np.pi,np.pi,Taille)

#SCRIPT : COBYLA
def contrainte1(p):
    return p[0]**2+p[1]**2-1

def contrainte2(p):
    return 1-(p[0]**2+p[1]**2)

con=[{'type':'ineq','fun':contrainte1},{'type':'ineq','fun':contrainte2}]


class FuncGen:
    def __init__(self,params,params2,a,b):
        self.params=params
        self.params2=params2
        self.a=a
        self.b=b

    def fun(self,p):
        tot=0
        for i in self.params:
            tot+=abs(i[0]*p[0]+i[1]*p[1])
        for i in self.params2:
            tot-=abs(i[0]*p[0]+i[1]*p[1])
        tot*=np.pi/2
        tot-=abs(self.a*p[0]+self.b*p[1])

        return tot


def func(i,j,a,b,con,params1,params2):
    Gen=FuncGen(params1,params2,a[i],b[j])
    res=minimize(Gen.fun,[0,0],method='COBYLA',constraints=con)
    return Gen.fun(res.x)

@timer
def resolv(a,b,con,params1,params2):
    return joblib.Parallel(n_jobs=14)(joblib.delayed(func)(i,j,a,b,con,params1,params2) for i in np.arange(Taille) for j in np.arange(Taille))


#WARNING : ARRAY NOT RESHAPED TO THE CORRECT SHAPE FOR CONSISTENCY WITH THE ND SOLVER (see N dim solver)
#PLZ ADD dat=dat.reshape(np.ones(N,dtype=int)*Taille) AT THE BEGINNING OF YOUR CODE AFTER LOADING DATA AND EXTRACTING N AND Taille
dat = np.array(resolv(a,b,con,params1,params2))
dat=np.append(dat,[N,Taille])
np.savetxt('Data.txt',dat)