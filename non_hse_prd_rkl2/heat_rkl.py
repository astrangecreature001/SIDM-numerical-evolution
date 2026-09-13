"""Explicit second-order Runge-Kutta-Legendre thermal increment.

Geometry and lapse are fixed during a hydrodynamic step; conductivity is
recomputed at each thermal stage.
RKL2 recurrence: Meyer et al. 2014; Vaidya et al. 2017 Appendix A."""
import numpy as np
from numba import njit

@njit(cache=True)
def rhs(ep,ep0,P0,R,rho,E,Ga,A,R0,sigma,fq,a,b,C):
    n=len(ep);q=np.zeros(n+1);rate=np.zeros(n);P=P0*ep/ep0
    ec=.5*(E[:-1]+E[1:])
    for i in range(1,n+1):
        left=i-1;right=i
        if i==n:left=n-2;right=n-1
        # Use the outer one-sided gradient and endpoint coefficient.
        ef=.5*(ep[left]+ep[right]);pf=.5*(P[left]+P[right]);vf=.5*(1/rho[left]+1/rho[right])
        if i==n:ef=ep[-1];pf=P[-1];vf=1/rho[-1]
        grad=(ec[right]*ep[right]-ec[left]*ep[left])/(.5*(A[right+1]-A[left]))
        den=1/C+a/b*sigma**2/(4*np.pi*R0**4)*pf
        q[i]=fq*Ga[i]*np.sqrt(ef)*pf/vf*R[i]**2/E[i]*grad/den
    L=4*np.pi*R**2*E**2*q
    for j in range(n):rate[j]=-(L[j+1]-L[j])/(A[j+1]-A[j])*sigma/R0**3.5/ec[j]
    return rate,L[-1]-L[0]

@njit(cache=True)
def stage_count(ep,P,R,rho,E,Ga,A,R0,sigma,fq,a,b,C,dt,safety=4.):
    n=len(ep);rates=np.zeros(n);ec=.5*(E[:-1]+E[1:]);W=np.diff(A)*ec*R0**3.5/sigma
    for i in range(1,n):
        ef=.5*(ep[i-1]+ep[i]);pf=.5*(P[i-1]+P[i]);vf=.5*(1/rho[i-1]+1/rho[i])
        den=1/C+a/b*sigma**2/(4*np.pi*R0**4)*pf
        conduct=-fq*Ga[i]*np.sqrt(ef)*pf/vf*R[i]**2/E[i]/den
        g=4*np.pi*R[i]**2*E[i]**2*conduct/(.5*(A[i+1]-A[i-1]))
        rates[i-1]+=g*ec[i-1]/W[i-1];rates[i]+=g*ec[i]/W[i]
    # Interior diffusion bound; outer extrapolating boundary not an M-matrix guarantee.
    demand=safety*dt*np.max(rates);s=3
    while (s*s+s-2)/4<demand:s+=2
    if s>10001:raise ValueError('RKL stage budget exceeded')
    return s,float(dt*np.max(rates))

@njit(cache=True)
def advance(ep0,P0,R,rho,E,Ga,A,R0,sigma,fq,a,b,C,dt,safety=4.):
    stages,courant=stage_count(ep0,P0,R,rho,E,Ga,A,R0,sigma,fq,a,b,C,dt,safety)
    w1=4./(stages*stages+stages-2);B=np.full(stages+1,1/3.)
    for j in range(2,stages+1):B[j]=(j*j+j-2.)/(2*j*(j+1.))
    f0,flux0=rhs(ep0,ep0,P0,R,rho,E,Ga,A,R0,sigma,fq,a,b,C)
    ym2=ep0.copy();ym1=ep0+B[1]*w1*dt*f0
    if np.min(ym1)<=0 or not np.all(np.isfinite(ym1)):raise ValueError('Nonpositive RKL stage 1')
    z0=0.;zm2=0.;zm1=B[1]*w1*dt*flux0
    for j in range(2,stages+1):
        mu=(2*j-1.)/j*B[j]/B[j-1];nu=-(j-1.)/j*B[j]/B[j-2]
        mut=mu*w1;gam=-(1-B[j-1])*mut
        f,flux=rhs(ym1,ep0,P0,R,rho,E,Ga,A,R0,sigma,fq,a,b,C)
        y=mu*ym1+nu*ym2+(1-mu-nu)*ep0+mut*dt*f+gam*dt*f0
        z=mu*zm1+nu*zm2+mut*dt*flux+gam*dt*flux0
        if np.min(y)<=0 or not np.all(np.isfinite(y)):raise ValueError('Nonpositive RKL intermediate stage')
        ym2=ym1;ym1=y;zm2=zm1;zm1=z
    W=np.diff(A)*.5*(E[:-1]+E[1:])*R0**3.5/sigma
    balance=np.sum(W*(ym1-ep0))+zm1
    norm=np.sum(W*np.abs(ym1-ep0))+abs(zm1)
    relative=abs(balance)/max(norm,1e-300)
    return ym1-ep0,stages,courant,relative
