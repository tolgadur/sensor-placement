import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import time
import copy

from numpy.linalg import inv
from numpy import linalg as LA


import math
from scipy.sparse.linalg import svds


import sys
sys.path.append('fluidity-master')
import vtktools


ntime = 536
ntimelocal = 536





indLoc = np.loadtxt('./solutions/LSBU32/area_6_8/solution_6.txt') # indexes of nodes

NindLoc = len(indLoc)

uvwTot = np.array([])
for i in range(ntimelocal):
    print(i)

    filename = './data/raw_data/sub-domains/LSBU_*_6/LSBU_'+str(i+1)+'_6.vtu' # path to all vtu files
    ug=vtktools.vtu(filename)
    ug.GetFieldNames()
    xFluidity = ug.GetScalarField('TracerGeorge') # Output variable
    xMLocal = np.array([])
    for j in range(NindLoc):
        indexLocal = indLoc[j]
        indexLocal = int(indexLocal)
        xMpoint = xFluidity[indexLocal]
        xMLocal = np.append(xMLocal,xMpoint)
    uvwTot = np.append(uvwTot,xMLocal)

n = NindLoc


dimuvwTot = len(uvwTot)

m = np.array([])
m = np.zeros(dimuvwTot/ntime)
for j in range(n):
        for i in range(1,ntime+1):
                m[j] = np.copy(m[j] + uvwTot[j+(i-1)*n])
        m[j] = m[j]/ntime



err = np.array([])
err = np.zeros(dimuvwTot)

for j in range(n):
        for i in range(1,ntime+1):
                err[j+(i-1)*j] = abs(uvwTot[j+(i-1)*j]-m[j])

W =  np.transpose(np.reshape(err, (ntime,n)))

trnc=n-1
Utrunc, strunc, Wtrunc = svds(W, k=trnc)
X = Utrunc.dot(np.diag(np.sqrt(strunc)))
np.savetxt("./validation/matrixVpreclocal"+str(trnc)+".txt", X)

V = X.copy()

lam = 1e-60


#put the observation file (time step 988)

ugg=vtktools.vtu('./data/raw_data/sub-domains/LSBU_*_6/LSBU_536_6.vtu') #the last one
ugg.GetFieldNames()
uvwVecobstot = ugg.GetScalarField('n2_TracerFluidity_WT')
uvwVecobs = np.array([])
for i in range(NindLoc):
    indexLocal = indLoc[i]
    indexLocal = int(indexLocal)
    xMpointobs = uvwVecobstot[indexLocal]
    uvwVecobs = np.append(uvwVecobs,xMpointobs)


#put the background (time step 100)

nstobs = len(uvwVecobs)
ug=vtktools.vtu('./data/raw_data/sub-domains/LSBU_*_6/LSBU_200_6.vtu.vtu') # time step 100 or 200
ug.GetFieldNames()
uvwVectot = ug.GetScalarField('Tracer1')
nRec = len(uvwVectot)
uvwVec = np.array([])
for i in range(NindLoc):
    indexLocal = indLoc[i]
    indexLocal = int(indexLocal)
    xMpointFl = uvwVectot[indexLocal]
    uvwVec = np.append(uvwVec,xMpointFl)

nst = len(uvwVec)
pos=ug.GetLocations()
z=pos[:,2]
n = len(uvwVec)

m = trnc
xB = uvwVec.copy()
y = uvwVecobs.copy()
R = lam * 0.9

x0 = uvwVec

Vin = np.linalg.pinv(V)
v0 = np.dot(Vin,x0)
VT = np.transpose(V)
HxB = xB.copy()
d = np.subtract(y,HxB)

# Cost function J
def J(v):
        vT = np.transpose(v)
        vTv = np.dot(vT,v)
        Vv = np.dot(V,v)
        Jmis = np.subtract(Vv,d)
        invR = 1/R
#       invR = 1e+60
        JmisT = np.transpose(Jmis)
        RJmis = JmisT.copy()
        J1 = invR*np.dot(Jmis,RJmis)
        Jv = (vTv + J1) / 2
        return Jv

# Gradient of J
def gradJ(v):
        Vv = np.dot(V,v)
        Jmis = np.subtract(Vv,d)
        invR = 1/R
#       invR = 1e+60
        g1 = Jmis.copy()
        VT = np.transpose(V)
        g2 = np.dot(VT,g1)
        gg2 = np.multiply(invR , g2)
        ggJ = v + gg2
        return ggJ

# Compute the minimum
t = time.time()

res = minimize(J, v0, method='L-BFGS-B', jac=gradJ,
                options={'disp': True})


vDA = np.array([])
vDA = res.x
deltaxDA = np.dot(V,vDA)
xDA = xB + deltaxDA

elapsed = time.time() - t
#print 'elapsed' , elapsed , '\n'

errxB = y - xB
MSExb = LA.norm(errxB, 2)/LA.norm(y, 2)
print 'L2 norm of the background error' , MSExb , '\n'

errxDA = y - xDA
MSExDA = LA.norm(errxDA, 2)/LA.norm(y, 2)
print 'L2 norm of the error in DA solution' , MSExDA , '\n'

errxBtot = np.array([])
errxBtot = np.zeros(nRec)

for j in range(NindLoc):
    indexLocal = indLoc[j]
    indexLocal = int(indexLocal)
    errxBtot[indexLocal] = errxB[j]

abserrxBtot = np.absolute(errxBtot)

ug.AddScalarField('u_0^M - u_C', abserrxBtot)
ug.Write('./validation/abserruM400-local.vtu') # create result folder and


errxDAtot = np.array([])
errxDAtot = np.zeros(nRec)

for j in range(NindLoc):
    indexLocal = indLoc[j]
    indexLocal = int(indexLocal)
    errxDAtot[indexLocal] = errxDA[j]


abserrxDAtot = np.absolute(errxDAtot)

ug.AddScalarField('u^DA - u_C', abserrxDAtot)
ug.Write('./validation/abserruDA400-local.vtu')


errxDAtot = np.array([])
errxDAtot = np.zeros(nRec)

for j in range(NindLoc):
    indexLocal = indLoc[j]
    indexLocal = int(indexLocal)
    errxDAtot[indexLocal] = errxDA[j]


abserrxDAtot = np.absolute(errxDAtot)

ug.AddScalarField('u^DA - u_C', abserrxDAtot)
ug.Write('./validation/abserruDA400-local.vtu')


xDAtot = uvwVectot.copy()

for j in range(NindLoc):
    indexLocal = indLoc[j]
    indexLocal = int(indexLocal)
    xDAtot[indexLocal] = xDA[j]

ug.AddScalarField('uDA', xDAtot)
ug.Write('./validation/uDA400-local.vtu')


xBtot = uvwVectot.copy()

# ug.AddScalarField('uM', xBtot)
# ug.Write('/home') # result folder
