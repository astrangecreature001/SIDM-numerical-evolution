"""Native-units family adapter. All new initial states are independently rebalanced."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import kernel as k
import main as native
from physics import Baryons,Units,GYR
ROOT=Path(__file__).resolve().parent
PROFILES={'none':0,'feng2021-softened':1,'plummer':2,'hernquist':3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def ctx(c):
    h=c['halo'];M=h['M10_Msun'];rs=h['rs_kpc']
    p=dict(M=M,Rs=rs,sigma0=5.,gm=5/3,a=2.26,b=1.38,C=.75,ephib=1.)
    R0=(rs/2.6)/(M/6.3e9)*8.5e6
    sig=5*2e33*M/(1.48e5*M)**2
    t0=1.35e12/5*(M/6.3e9)**(-2.5)*(rs/2.6)**3.5
    return p,R0,sig,t0
def evolve(s,A,dt,prev,c):
    p,R0,sig,_=ctx(c)
    k.evolve_kernel_unified(*s,A,dt,1.,R0,sig,-1.5*(p['gm']-1)**1.5*p['a'],p['a'],p['b'],p['gm'],1.,999.,False,True,True,.75,prev,PROFILES[c['profile']],c['mu_b'],c['eta_b'] or 1.)
def build(c):
    p,R0,sig,t0=ctx(c);oldA=np.load(ROOT/'reference/A.npy').ravel();old=np.load(ROOT/'reference/initial.npy')
    n=1000;first=c['inner_shell_mass_Msun']/p['M'];lo=0.;hi=.1
    for _ in range(100):
        mid=(lo+hi)/2
        if np.sum(first*np.exp(mid*np.arange(n)))>oldA[-1]:hi=mid
        else:lo=mid
    logq=(lo+hi)/2;dm=first*np.exp(logq*np.arange(n));A=np.r_[0,np.cumsum(dm)]
    R=np.cbrt(np.interp(A,oldA,old[1]**3));R[0]=0.;R[-1]=old[1,-1]
    bary=Baryons(c['profile'],c['mu_b'],c['eta_b'] or 1.);mb=bary.mass(R)
    vol=4*np.pi/3*np.diff(R**3);Ga=np.ones(n+1);m=A.copy();eps=np.full(n,old[3,-2]);rho=dm/vol;P=(p['gm']-1)*rho*eps
    for it in range(500):
        prevP=P.copy();prevm=m.copy();prevGa=Ga.copy()
        Ga[1:]=np.sqrt(1-2*(m[1:]+mb[1:])/(R[1:]*R0));gc=.5*(Ga[:-1]+Ga[1:]);rho=gc*dm/vol
        eps=P/((p['gm']-1)*rho);w=1+(eps+P/rho)/R0
        wf=.5*(w[:-1]+w[1:]);pf=.5*(P[:-1]+P[1:])
        g=(m[1:-1]+mb[1:-1])/R[1:-1]**2+4*np.pi*pf*R[1:-1]/R0
        drop=.5*(dm[1:]+dm[:-1])*wf*g/(Ga[1:-1]*4*np.pi*R[1:-1]**2)
        P[-1]=(p['gm']-1)*rho[-1]*old[3,-2];P[:-1]=P[-1]+np.cumsum(drop[::-1])[::-1]
        eps=P/((p['gm']-1)*rho);m=np.r_[0,np.cumsum(gc*dm*(1+eps/R0))]
        err=max(np.max(abs(P/prevP-1)),np.max(abs(m[1:]/prevm[1:]-1)),np.max(abs(Ga/prevGa-1)))
        if err<1e-13:break
    else:raise RuntimeError('HSE initial solve did not converge')
    w=1+(eps+P/rho)/R0;pad=lambda v:np.r_[v,v[-1]]
    gp=-k.deriv_c2f(P,A)/(k.avg_c2f(rho)*k.avg_c2f(w));phi=np.r_[-np.cumsum((.5*(gp[:-1]+gp[1:])*dm)[::-1])[::-1],0.];E=np.exp(phi/R0)
    q=k.evaluate_heat(R,pad(rho),pad(eps),pad(P),E,Ga,A,R0,sig,-1.5*(p['gm']-1)**1.5*p['a'],p['a'],p['b'],p['C'])
    rf=k.avg_c2f(rho);H=np.zeros(n+1);H[1:]=q[1:]/(4*np.pi*R[1:]**2*rf[1:]**2)
    s=np.array([np.zeros(n+1),R,pad(rho),pad(eps),pad(P),pad(w),E,m,Ga,H,q]);native.validate(A,s)
    termP=Ga[1:-1]*4*np.pi*R[1:-1]**2*k.deriv_c2f(P,A)[1:-1]/k.avg_c2f(w)[1:-1]
    grav=(m[1:-1]+mb[1:-1])/R[1:-1]**2+4*np.pi*k.avg_c2f(P)[1:-1]*R[1:-1]/R0
    residual=float(np.max(abs(termP+grav)/abs(grav)))
    assert residual<1e-9 and abs(dm[0]*p['M']/c['inner_shell_mass_Msun']-1)<1e-12
    assert np.max(abs(np.diff(A)[1:]/np.diff(A)[:-1]/np.exp(logq)-1))<1e-10
    assert abs(bary.depth()-c['target_potential_ratio'])<1e-10
    report=dict(case_id=c['case_id'],hse_iterations=it+1,force_residual=residual,grid_q=float(np.exp(logq)),first_shell_Msun=float(dm[0]*p['M']),rout_rs=float(R[-1]),total_rest_mass_Msun=float(A[-1]*p['M']),positive_finite_ordered=True,actual_potential_ratio_10rs=bary.depth(),native_scales=dict(R0=R0,sigma=sig,t0=t0),physical_reference=Units(p['M'],p['Rs']).metadata(),initialization='native cumulative rest-mass to volume relation conservatively remapped; new discrete equilibrium per model; not analytic NFW resampling',outer_epsilon=float(old[3,-2]))
    return A,s,report
def run():
    configs=json.loads((ROOT/'parameter_anchors.json').read_text(encoding='utf-8'))['configurations'];reports=[]
    for d in ('prepared','runs','pilots'): (ROOT/d).mkdir(exist_ok=True)
    for c in configs:
        A,s,r=build(c);out=ROOT/'prepared'/c['case_id'];out.mkdir(exist_ok=False)
        np.save(out/'A.npy',A);np.save(out/'initial.npy',s)
        c['initial_sha256']=sha(out/'initial.npy');c['grid_sha256']=sha(out/'A.npy')
        dump(out/'run_config.json',c);dump(out/'initial_state_validation.json',r);reports.append(r)
        print(c['case_id'],r['force_residual'],flush=True)
    dump(ROOT/'run_manifest.json',dict(unique_programs=len(configs),configurations=configs))
    dump(ROOT/'initial_state_validation.json',reports)
    dump(ROOT/'source_hashes.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
if __name__=='__main__':run()
