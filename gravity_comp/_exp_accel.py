#!/usr/bin/env python3
"""对比残差网络在不同特征集下的测试集 RMSE。"""
import csv, numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
np.random.seed(42)

def skew(v): return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

rows=[]
with open('datasets/data1.csv') as f:
    for r in csv.DictReader(f):
        q=np.array([float(r['qx']),float(r['qy']),float(r['qz']),float(r['qw'])]); q/=np.linalg.norm(q)
        if q[3]<0: q=-q
        rows.append(dict(q=q, Rm=R.from_quat(q).as_matrix(),
            w=np.array([float(r[k]) for k in ['fx','fy','fz','tx','ty','tz']]),
            qp=np.array([float(r[f'joint_{i}_pos']) for i in range(1,7)]),
            qv=np.array([float(r[f'joint_{i}_vel']) for i in range(1,7)]),
            t=float(r['time_sec'])))
N=len(rows); t=np.array([r['t'] for r in rows])
QV=np.vstack([r['qv'] for r in rows])
QVs=savgol_filter(QV,11,3,axis=0)
QA=savgol_filter(np.gradient(QVs,t,axis=0),11,3,axis=0)
for i,r in enumerate(rows): r['qa']=QA[i]; r['qvs']=QVs[i]

# 物理辨识
def identify(rr):
    n=len(rr); A=np.zeros((3*n,6)); B=np.zeros(3*n)
    for i,r in enumerate(rr):
        A[3*i:3*i+3,:3]=r['Rm']; A[3*i:3*i+3,3:]=np.eye(3); B[3*i:3*i+3]=r['w'][:3]
    X,*_=np.linalg.lstsq(A,B,rcond=None); G,Fb=X[:3],X[3:]
    A=np.zeros((3*n,6)); B=np.zeros(3*n)
    for i,r in enumerate(rr):
        V=r['Rm']@G; A[3*i:3*i+3,:3]=-skew(V); A[3*i:3*i+3,3:]=np.eye(3); B[3*i:3*i+3]=r['w'][3:]
    X,*_=np.linalg.lstsq(A,B,rcond=None)
    return dict(G=G,Fb=Fb,CoM=X[:3],Tb=X[3:])
def phys(r,p):
    return np.concatenate([r['Rm']@p['G']+p['Fb'], -skew(r['Rm']@p['G'])@p['CoM']+p['Tb']])

def feat(r,p,acc):
    g=r['Rm']@p['G']; g/=np.linalg.norm(g)+1e-9
    f=[*g,*r['q'],*np.sin(r['qp']),*np.cos(r['qp']),*r['qvs']]
    if acc: f+=[*r['qa'],*(r['qvs']**2)]
    return np.array(f)

class MLP:
    def __init__(s,din,dh=64):
        rng=np.random.default_rng(0); sc=lambda a,b:rng.standard_normal((a,b))*np.sqrt(2/a)
        s.P=[sc(din,dh),np.zeros(dh),sc(dh,dh),np.zeros(dh),sc(dh,6),np.zeros(6)]
    def fwd(s,X,c=False):
        z1=X@s.P[0]+s.P[1]; a1=np.tanh(z1); z2=a1@s.P[2]+s.P[3]; a2=np.tanh(z2); y=a2@s.P[4]+s.P[5]
        if c: s.c=(X,a1,a2)
        return y
    def bwd(s,dy):
        X,a1,a2=s.c; n=X.shape[0]
        gW3=a2.T@dy/n; gb3=dy.mean(0); da2=dy@s.P[4].T; dz2=da2*(1-a2**2)
        gW2=a1.T@dz2/n; gb2=dz2.mean(0); da1=dz2@s.P[2].T; dz1=da1*(1-a1**2)
        gW1=X.T@dz1/n; gb1=dz1.mean(0); return [gW1,gb1,gW2,gb2,gW3,gb3]

def train(X,Y,mask,ep=2500,lr=1e-3):
    net=MLP(X.shape[1]); m=[np.zeros_like(p) for p in net.P]; v=[np.zeros_like(p) for p in net.P]
    idx=np.where(mask)[0]; np.random.shuffle(idx); ntr=int(len(idx)*0.8); tr,va=idx[:ntr],idx[ntr:]
    best=None; bl=1e9
    for e in range(1,ep+1):
        p=net.fwd(X[tr],True); dy=2*(p-Y[tr]); g=net.bwd(dy)
        for i in range(6):
            m[i]=.9*m[i]+.1*g[i]; v[i]=.999*v[i]+.001*g[i]**2
            net.P[i]-=lr*(m[i]/(1-.9**e))/(np.sqrt(v[i]/(1-.999**e))+1e-8)
        vl=np.mean((net.fwd(X[va])-Y[va])**2)
        if vl<bl: bl=vl; best=[p.copy() for p in net.P]
    for i in range(6): net.P[i][...]=best[i]
    return net

perm=np.arange(N); np.random.shuffle(perm); ntr=int(N*0.8)
mask=np.zeros(N,bool); mask[perm[:ntr]]=True; test=perm[ntr:]
p=identify([rows[i] for i in perm[:ntr]])
meas=np.vstack([r['w'] for r in rows]); ph=np.vstack([phys(r,p) for r in rows])
Yres=meas-ph

for acc in [False,True]:
    X=np.vstack([feat(r,p,acc) for r in rows])
    xm,xs=X[mask].mean(0),X[mask].std(0)+1e-8; Xn=(X-xm)/xs
    ym,ys=Yres[mask].mean(0),Yres[mask].std(0)+1e-8; Yn=(Yres-ym)/ys
    net=train(Xn,Yn,mask); pred=net.fwd(Xn)*ys+ym; ext=Yres-pred
    rm=np.sqrt(np.mean(ext[test]**2,0))
    tag='vel+acc+vel^2' if acc else 'vel only (当前)'
    print(f'[{tag:16s}] dim={X.shape[1]:2d}  F:{rm[0]:.4f},{rm[1]:.4f},{rm[2]:.4f}  T:{rm[3]:.4f},{rm[4]:.4f},{rm[5]:.4f}')
