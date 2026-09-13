"""Non-HSE hydrodynamic runner and shared state validation utilities.

The default policy uses the acoustic CFL timestep. The upstream-fixed policy
uses the configured fixed timestep. Thermal evolution uses explicit RKL2."""
import os,sys,json,time,hashlib,argparse,traceback
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import kernel as k
ROOT=Path(__file__).resolve().parent
def read_config(name):
    result={}
    for line in (ROOT/name).read_text().splitlines():
        line=line.split('#')[0].strip()
        if '=' not in line:continue
        key,v=map(str.strip,line.split('=',1))
        if v.lower() in ('true','false'):result[key]=v.lower()=='true'
        else:
            try:result[key]=float(v)
            except ValueError:result[key]=v
    return result
def context():
    c,p=read_config('config.txt'),read_config('parameter.txt')
    R0=(p['Rs']/2.6)/(p['M']/6.3e9)*8.5e6
    sig=p['sigma0']*2e33*p['M']/(1.48e5*p['M'])**2
    t0=1.35e12/p['sigma0']*(p['M']/6.3e9)**(-2.5)*(p['Rs']/2.6)**3.5
    return c,p,R0,sig,t0
def load():
    A=np.load(ROOT/'A.npy').astype(float).flatten();raw=np.load(ROOT/'initial.npy')
    if raw.ndim!=2 or raw.shape[0]<11 or raw.shape[1]!=len(A):raise ValueError('Grid/state shape mismatch')
    return A,[np.ascontiguousarray(a.copy()) for a in raw[:11]]
def validate(A,s):
    if not np.isfinite(s).all():raise ValueError('Nonfinite state')
    if np.min(np.diff(A))<=0 or np.min(np.diff(s[1]))<=0:raise ValueError('Shell crossing')
    for i in (2,3,4,5,6,8):
        if np.min(s[i])<=0:raise ValueError('Nonpositive state field '+str(i))
    if np.min(np.diff(s[7]))<=0:raise ValueError('Nonmonotone gravitational mass')
    if s[0][0]!=0 or s[1][0]!=0 or s[7][0]!=0:raise ValueError('Invalid centre')
def cfl_dt(s,R0,cfl):
    speed=np.maximum(abs(s[0][:-1]),abs(s[0][1:]))/R0+np.sqrt(s[4][:-1]/s[2][:-1]/R0)
    dt=cfl*np.min(np.diff(s[1])*R0/(np.maximum(s[6][:-1],s[6][1:])*speed))
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid CFL')
    return float(dt)
def call(s,A,dt,prev,p,R0,sig,fp):
    k.evolve_kernel_unified(*s,A,dt,1.,R0,sig,-1.5*(p['gm']-1)**1.5*p['a'],p['a'],p['b'],p['gm'],p['ephib'],999.,False,fp,True,p['C'],prev)
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def run():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--policy',choices=['cfl','upstream-fixed'],default='cfl');ap.add_argument('--steps',type=int,default=1000000);ap.add_argument('--cfl',type=float,default=.0625);ap.add_argument('--fp',choices=['divergence','off'],default='divergence');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if Path(args.name).name!=args.name:raise ValueError('Invalid run name')
    out=ROOT/'runs'/args.name;c,p,R0,sig,t0=context();A,s=load();validate(A,s)
    if c['FLAG_HSE'] or not c['FLAG_HEAT'] or c['freeze_radius']!=999:
        raise ValueError('This entry requires dynamical evolution with heat enabled and no frozen subdomain')
    source={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'kernel.py',ROOT/'heat_rkl.py',ROOT/'main.py',ROOT/'parameter.txt',ROOT/'config.txt',ROOT/'A.npy',ROOT/'initial.npy')}
    config=dict(policy=args.policy,cfl=args.cfl,fp=args.fp,steps_target=args.steps,parameters=p,R0_native=R0,sigma_native=sig,t0_native_geometric=t0,
        time_label='tau_native=t_geometric/t0_approx; approximate time normalization',sources=source,seed=None)
    elapsed=0.;n=0;t=0.;prev=0.
    if args.resume:
        if json.loads((out/'run_config.json').read_text())!=config:raise ValueError('Resume configuration/source mismatch')
        with np.load(out/'latest_checkpoint.npz') as d:
            s=[a.copy() for a in d['state']];A=d['A'].copy();n=int(d['steps']);t=float(d['time']);prev=float(d['dt_previous']);elapsed=float(d['wall_seconds'])
        validate(A,s)
    else:
        out.mkdir(parents=True,exist_ok=False);(out/'snapshots').mkdir();atomic(out/'run_config.json',config)
    start=time.monotonic();lastsave=start;initial_m=s[7].copy();initial_E=s[6].copy()
    def record(final=False):
        kn=np.sqrt(4*np.pi*s[2][:-1]/R0**3)/(sig*s[2][:-1]/R0**3*np.sqrt((p['gm']-1)*s[3][:-1]/R0))
        row=dict(steps=n,tau_native=t/t0,time_geometric=t,dt_previous=prev,wall_seconds=elapsed+time.monotonic()-start,pid=os.getpid(),updated_unix=time.time(),rho_c_native=float(s[2][0]),eps_c_native=float(s[3][0]),m_outer=float(s[7][-1]),lapse_c=float(s[6][0]),central_Kn=float(kn[0]),min_width=float(np.min(np.diff(s[1]))),max_compactness=float(np.max(2*s[7][1:]/(s[1][1:]*R0))),max_mass_change_from_segment_start=float(np.max(abs(s[7]-initial_m))),max_lapse_change_from_segment_start=float(np.max(abs(s[6]-initial_E))),finite=True)
        tmp=out/'latest_checkpoint.tmp.npz';np.savez_compressed(tmp,A=A,state=np.array(s),steps=n,time=t,dt_previous=prev,wall_seconds=row['wall_seconds']);os.replace(tmp,out/'latest_checkpoint.npz')
        snap=out/'snapshots'/f'step_{n:012d}.npz'
        if (final or n==0 or n%10000==0) and not snap.exists():np.savez_compressed(snap,A=A,state=np.array(s),steps=n,time=t,dt_previous=prev,wall_seconds=row['wall_seconds'])
        atomic(out/'latest_progress.json',row)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        return row
    record();exitcode=0;reason='requested_steps_completed'
    try:
        while n<args.steps:
            dt=cfl_dt(s,R0,args.cfl) if args.policy=='cfl' else c['dt_t0']*t0
            candidate=[a.copy() for a in s]
            call(candidate,A,dt,prev,p,R0,sig,args.fp=='divergence');validate(A,candidate)
            s=candidate;n+=1;t+=dt;prev=dt
            if n%10000==0 or time.monotonic()-lastsave>30:record();lastsave=time.monotonic()
    except Exception as exc:
        exitcode=1;reason=str(exc)
        (out/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc())
        # s is always the last accepted valid state; no partial kernel mutation.
    row=record(final=True);atomic(out/'run_summary.json',dict(exit_code=exitcode,completion_reason=reason,last_valid=row));(out/'exit_code').write_text(str(exitcode)+'\n');sys.exit(exitcode)
if __name__=='__main__':run()
