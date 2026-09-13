"""Production native RKL2 driver: checkpointed, peak-gated, never floors failed states."""
import os,json,time,hashlib,argparse,traceback,fcntl,shutil
from pathlib import Path
import numpy as np
import main as native
from prepare import ROOT,ctx,evolve,sha
from physics import Baryons,F10
STOP_POLICY = dict(name='postpeak_0p1_tau_native_v1', postpeak_interval_tau_native=0.1,
    min_postpeak_records=8, min_tau_native=10., density_growth_100_required=False,
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def atomic(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,p)
def diag(s,A,R0,sig):
    # Same lambda/H convention as the native test runner; central connected SMFP only.
    rho=s[2][:-1];v=np.sqrt((2/3)*s[3][:-1]/R0)
    kn=np.sqrt(4*np.pi*rho/R0**3)/(sig*rho/R0**3*v)
    centres=.5*(A[:-1]+A[1:]);j=0
    if kn[0]>1:mass=0.;boundary=0.
    else:
        while j<len(kn) and kn[j]<=1:j+=1
        if j==len(kn):mass=float(A[-1]);boundary=float(s[1][-1])
        else:
            f=-np.log(kn[j-1])/(np.log(kn[j])-np.log(kn[j-1]))
            mass=float(centres[j-1]+f*(centres[j]-centres[j-1]))
            rc=.5*(s[1][:-1]+s[1][1:]);boundary=float(rc[j-1]+f*(rc[j]-rc[j-1]))
    return dict(rho_c_over_rho_s=float(rho[0]*4*np.pi*F10),eps_c_native=float(s[3][0]),central_Kn=float(kn[0]),M_SMFP_over_M10=mass,r_smfp_over_rs=boundary,smfp_cells=j,smfp_phase='SMFP' if mass>0 else 'LMFP',max_dm_compactness=float(np.max(2*s[7][1:]/(s[1][1:]*R0))),min_width=float(np.diff(s[1]).min()))
def run(args):
    cfgpath=ROOT/'prepared'/args.case/'run_config.json';cfg=json.loads(cfgpath.read_text());p,R0,sig,t0=ctx(cfg)
    source=json.loads((ROOT/'source_hashes.json').read_text())
    for name,digest in source.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed: '+name)
    inp=cfgpath.parent
    assert sha(inp/'A.npy')==cfg['grid_sha256'] and sha(inp/'initial.npy')==cfg['initial_sha256']
    base=ROOT/('pilots' if args.pilot_steps else 'runs');out=base/args.case
    out.mkdir(exist_ok=True);lock=(out/'process.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'snapshots').mkdir(exist_ok=True);A=np.load(inp/'A.npy');s=[a.copy() for a in np.load(inp/'initial.npy')]
    native.validate(A,s);n=0;t=0.;prev=0.;elapsed=0.;best=None;minrho=float(s[2][0]*4*np.pi*F10);post=0;confirmed=False;lastsample=-np.inf
    startup=time.time();start=time.monotonic();lastwall=start
    if args.resume:
        if json.loads((out/'run_config.json').read_text())!=cfg:raise RuntimeError('Config mismatch')
        meta=json.loads((out/'latest_checkpoint.json').read_text());path=out/meta['snapshot']
        if sha(path)!=meta['sha256']:raise RuntimeError('Resume hash mismatch')
        with np.load(path) as d:
            s=[a.copy() for a in d['state']];t=float(d['time']);n=int(d['steps']);prev=float(d['dt_previous'])
        elapsed=meta['wall_seconds'];best=meta['best'];minrho=meta['minrho'];post=meta['post'];confirmed=meta['peak_confirmed'];lastsample=t/t0
        native.validate(A,s)
        for name in ('run_summary.json','exit_code'):
            if (out/name).exists():(out/name).rename(out/(name+'.prior_'+str(time.time_ns())))
    else:
        if (out/'history.jsonl').exists() or (out/'run_config.json').exists():raise RuntimeError('Refusing duplicate fresh run')
        atomic(out/'run_config.json',cfg);atomic(out/'source_hashes.json',source)
    segment=dict(started_unix=startup,pid=os.getpid(),resumed=args.resume,seed=None,stop_policy=STOP_POLICY)
    with (out/'segments.jsonl').open('a') as f:f.write(json.dumps(segment)+'\n')
    def save(check_peak=True):
        nonlocal best,minrho,post,confirmed,lastsample
        d=diag(s,A,R0,sig);tau=t/t0;minrho=min(minrho,d['rho_c_over_rho_s'])
        if check_peak:
            if d['M_SMFP_over_M10']>0 and (best is None or d['M_SMFP_over_M10']>best['M_SMFP_over_M10']):
                best=dict(tau_native=tau,steps=n,M_SMFP_over_M10=d['M_SMFP_over_M10'],rho_c_over_rho_s=d['rho_c_over_rho_s'],snapshot=f'snapshots/step_{n:012d}.npz');post=0;confirmed=False
            elif best and tau>best['tau_native'] and d['M_SMFP_over_M10']<best['M_SMFP_over_M10']:post+=1
            guard=max(10.,best['tau_native']+0.1) if best else np.inf
            if best and best['tau_native']>0 and post>=8 and tau>=guard and d['rho_c_over_rho_s']>1.1*minrho:confirmed=True
        row=dict(**d,steps=n,tau_native=tau,time_geometric=t,dt_previous=prev,wall_seconds=elapsed+time.monotonic()-start,pid=os.getpid(),updated_unix=time.time(),best=best,minrho=minrho,post=post,peak_confirmed=confirmed,stop_policy=STOP_POLICY,density_half_events=0,accepted_nonmonotonic_half_steps=0,rejected_steps=0,finite=True)
        snap=out/'snapshots'/f'step_{n:012d}.npz'
        if not snap.exists():
            tmp=snap.with_suffix('.tmp.npz');np.savez_compressed(tmp,A=A,state=np.array(s),steps=n,time=t,dt_previous=prev);os.replace(tmp,snap)
        row.update(snapshot=str(snap.relative_to(out)),sha256=sha(snap))
        # Immutable snapshot is canonical; latest JSON pointer is committed after history.
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        atomic(out/'latest_checkpoint.json',row);atomic(out/'latest_progress.json',row)
        if confirmed:atomic(out/'peak_record.json',dict(stop_policy=STOP_POLICY,peak=best,confirmed_at_tau_native=tau,postpeak_samples=post,continue_after_peak=cfg['post_peak_continue'],qualification='maximum over sampled checkpoint sequence; not a proof about the infinite future'))
        lastsample=tau;return row
    code=0;reason='confirmed_m_smfp_peak';row=save(check_peak=not args.resume)
    try:
        while True:
            if confirmed and not cfg['post_peak_continue'] and not args.pilot_steps:break
            if args.pilot_steps and n>=args.pilot_steps:reason='pilot_steps_completed';break
            if n%1000==0 and shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('Free disk below 10 GiB: preserved checkpoint, needs capacity review')
            dt=native.cfl_dt(s,R0,.3);candidate=[a.copy() for a in s]
            evolve(candidate,A,dt,prev,cfg);native.validate(A,candidate)
            s=candidate;n+=1;t+=dt;prev=dt
            # Do not shorten CFL steps to hit output times; record actual cadence overshoot.
            if n%10==0:
                d=diag(s,A,R0,sig);cadence=.001 if d['M_SMFP_over_M10']>0 or best else .05
                if t/t0-lastsample>=cadence:row=save()
            if time.monotonic()-lastwall>=30:row=save();lastwall=time.monotonic()
    except Exception as e:
        code=1;reason='needs_diagnosis';(out/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc())
    row=save(check_peak=False)
    summary=dict(stop_policy=STOP_POLICY,case_id=args.case,exit_code=code,completion_reason=reason,peak=best,peak_confirmed=confirmed,last_valid=row,wall_seconds=row['wall_seconds'],source_hashes=source)
    atomic(out/'run_summary.json',summary);(out/'exit_code').write_text(str(code)+'\n');print(json.dumps(summary),flush=True)
    return code
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--pilot-steps',type=int,default=0)
    raise SystemExit(run(ap.parse_args()))
