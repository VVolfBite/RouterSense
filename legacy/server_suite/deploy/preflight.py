#!/usr/bin/env python3
"""SSH and software preflight for a deployment YAML."""
from __future__ import annotations
import argparse,json,shlex,subprocess
from pathlib import Path
from routersense_sched.pipeline.config import DeploymentConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True);a=p.parse_args();dep=DeploymentConfig.from_file(a.deployment);rows=[]
    python=shlex.quote(dep.python)
    check=(f"{python} --version; command -v nvidia-smi >/dev/null; "
           f"{python} -c 'import torch; print(torch.__version__, torch.cuda.device_count(), torch.distributed.is_nccl_available())'")
    for host in dep.hosts:
        target=f"{dep.ssh_user}@{host}" if dep.ssh_user else host
        r=subprocess.run(['ssh',*dep.ssh_options,target,check],text=True,capture_output=True)
        rows.append({'host':host,'ok':r.returncode==0,'stdout':r.stdout,'stderr':r.stderr,'python':dep.python})
    print(json.dumps({'ok':all(x['ok'] for x in rows),'hosts':rows},indent=2));raise SystemExit(0 if all(x['ok'] for x in rows) else 2)
if __name__=='__main__':main()
