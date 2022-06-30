import numpy as np
from scipy.optimize import minimize
import joblib
import timeit

"""
Instructions : 
Taille : Number of steps for the parameters (which go between -pi and pi)
N : Number of parameters
params1 : n1,n2,... appearing in the PochHammer in the NUMERATOR
params2 : n1,n2,... appearing in the PochHammer in the DENOMINATOR
NbCoeur : Number of hearts used for the multi-threading (default : Number of heart in your processor - 2)
"""

N=3
Taille=10
#Example : Convergence of the Mellin-Barnes representation of the Lauricella's Fa function, depending of the arguments of x,y,z.
params1=[(1,1,1),(1,0,0),(0,1,0),(0,0,1),(1,0,0),(0,1,0),(0,0,1)]
params2=[(1,0,0),(0,1,0),(0,0,1)]
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

#VARIABLE CREATION
Const=np.array([np.linspace(-np.pi,np.pi,Taille) for i in range(N)])

#RECURSIVE FONCTION FOR VARYING NUMBER OF LOOPS
def for_recursive(number_of_loops, range_list, execute_function, current_index=0, iter_list = []):
    if iter_list == []:
        iter_list = [0]*number_of_loops

    if current_index == number_of_loops-1:
        for iter_list[current_index] in range_list[current_index]:
            execute_function(iter_list)
    else:
        for iter_list[current_index] in range_list[current_index]:
            for_recursive(number_of_loops, iter_list = iter_list, range_list = range_list,  current_index = current_index+1, execute_function = execute_function) 


#CREATION OF MINIMIZATION USING COBYLA MINIMIZATION SCRIPT IN SCIPY.MINIMIZE

#CREATION OF CONSTRAINT : ||z||=1, z={z1,...,zn}
def contrainte(p):
    r=0
    for i in p:
        r+=i**2
    return (r-1)

def contrainte2(p):
    r=0
    for i in p:
        r+=i**2
    return (1-r)

con=[{'type':'ineq','fun':contrainte},{'type':'ineq','fun':contrainte2}]

#CREATION OF CONVERGENCE CRITERIA
class FuncGen:
    def __init__(self,params,params2,P,N):
        self.params=params
        self.params2=params2
        self.P=P
        self.N=N

    def fun(self,p):
        tot=0
        temp=0
        for i in self.params:
            for j in range(len(i)):
                temp+=i[j]*p[j]
            tot+=abs(temp)
            temp=0
        for i in self.params2:
            for j in range(len(i)):
                temp+=i[j]*p[j]
            tot-=abs(temp)
            temp=0
        tot*=np.pi/2
        for i in range(len(self.N)):
            temp+=self.P[i][self.N[i]]*p[i]
        tot-=abs(temp)

        return tot

#MINIMIZATION OF THE CONVERGENCE CRITERIA
def func(N,Const,con,params1,params2):
    Gen=FuncGen(params1,params2,Const,N)
    res=minimize(Gen.fun,np.zeros(len(Const)),method='COBYLA',constraints=con)
    return Gen.fun(res.x)


#MULTI-THREADING INIT
class Multi:
    def __init__(self,N,B,Taille,func,Params):
        self.B=B
        self.N=N
        self.Taille=Taille
        self.func=func
        self.Const=Params[0]
        self.con=Params[1]
        self.params1=Params[2]
        self.params2=Params[3]

    def f(self,arg):
        tot=0
        for l in range(len(arg)):
            tot+=int((arg[l]*Taille**l))
            arg[l]=int(arg[l])
        self.B[tot]=joblib.delayed(self.func)(arg[:],self.Const,self.con,self.params1,self.params2)
    
C=[[j for j in range(Taille)] for i in range(N)]

Arr=np.zeros(Taille**N)
Arr=Arr.tolist()

for_recursive(range_list=C, execute_function = Multi(N,Arr,Taille,func,[Const,con,params1,params2]).f, number_of_loops=N)

#SOLVE FUNCTION USING MULTI-THREADING
@timer
def resolv(Arr):
    return joblib.Parallel(n_jobs=NbCoeur)(Arr)

#EXECUTION
dat = np.array(resolv(Arr))
dat=np.append(dat,[N,Taille])
#WARNING : ARRAY NOT RESHAPED TO THE CORRECT SHAPE IN ORDER TO ALLOW EXECUTION OF SAVETXT
#PLZ ADD dat=dat.reshape(np.ones(N,dtype=int)*Taille) AT THE BEGINNING OF YOUR CODE AFTER LOADING DATA AND EXTRACTING N AND Taille
np.savetxt('Data.txt',(dat))
#N and Taille PARAMETERS WILL BE PLACED AT THE END OF THE Data.txt FILE