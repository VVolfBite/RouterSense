from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from scripts.analysis.sweep_oracle_large import _tasks_from_matrix, _problem, _result_row, _coordinates
from rs_sim.trace import load_fixture
from rs_sim.runtime.config.profiles import load_runtime_profile_bundle_json
from rs_sim.runtime.core.engine import _default_topology_for_fixture, _rscf_wire_cost_model_from_runtime
from rs_sim.scheduler.core.oracle import solve_exact_wire


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-windows',type=int,default=8)
    ap.add_argument('--local-limit-ms',type=int,default=1000)
    ap.add_argument('--joint-limit-ms',type=int,default=750)
    ns=ap.parse_args()
    started=time.monotonic()
    fixture=load_fixture(ns.fixture)
    profile=load_runtime_profile_bundle_json(ns.profile)
    topology=_default_topology_for_fixture(fixture,topology_id=f'oracle-sweep:{fixture.fixture_id}')
    hardware=profile.transport_profile.hardware_profile
    model,ep,seq,step=_coordinates(ns.fixture)
    rows=[]
    window_count=min(max(0,len(fixture.windows)-1),ns.max_windows)
    for wi in range(window_count):
        current=fixture.windows[wi]; following=fixture.windows[wi+1]
        p1_matrix=current.payload_matrix('COMBINE'); p2_matrix=following.payload_matrix('DISPATCH')
        wire=_rscf_wire_cost_model_from_runtime(
            topology=topology, hardware_profile=hardware, fixture_input=fixture,
            base_layer_index=int(current.layer_id),
            predicted_p2_matrix=tuple(tuple(int(v) for v in row) for row in p2_matrix),
            predicted_p2_confidence_ppm=1_000_000, timing_profile=None)
        base={'fixture_path':str(ns.fixture),'fixture_id':fixture.fixture_id,'model':model,'ep':ep,
              'sequence_length':seq,'step':step,'window_index':wi,'anchor_layer_id':int(current.layer_id),
              'target_layer_id':int(following.layer_id),'rank_count':int(fixture.world_size)}
        p1=_tasks_from_matrix(fixture_id=fixture.fixture_id,window_index=wi,phase=1,matrix=p1_matrix)
        p2=_tasks_from_matrix(fixture_id=fixture.fixture_id,window_index=wi,phase=2,matrix=p2_matrix)
        for phase,sem,tasks in [('P1',1,p1),('P2',2,p2)]:
            r=solve_exact_wire(_problem(fixture.world_size,tasks,tag=f'LOCAL:{phase}'),wire_cost_model=wire,
                               time_limit_ms=ns.local_limit_ms,relative_gap=.02,release_mode='PHASE_BARRIER',
                               semantic_phase_ordinal=sem)
            rows.append(_result_row(base,'LOCAL',phase,r))
        if ep==8 and seq==128 and wi==0 and model != 'OLMoE-1B-7B-0924':
            r=solve_exact_wire(_problem(fixture.world_size,p1+p2,tag='JOINT'),wire_cost_model=wire,
                               time_limit_ms=ns.joint_limit_ms,relative_gap=.02,release_mode='RANK_LOCAL',
                               semantic_phase_ordinal=None)
            rows.append(_result_row(base,'JOINT','P12',r))
    payload={'status':'PASS','fixture':str(ns.fixture),'fixture_id':fixture.fixture_id,'model':model,'ep':ep,
             'sequence_length':seq,'step':step,'window_count':window_count,'elapsed_s':time.monotonic()-started,
             'rows':rows}
    ns.output.parent.mkdir(parents=True,exist_ok=True)
    ns.output.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    print(json.dumps({k:payload[k] for k in ('status','fixture_id','ep','sequence_length','window_count','elapsed_s')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    code=0
    try: main()
    except BaseException as e:
        code=1
        print(json.dumps({'status':'FAILED','error':f'{type(e).__name__}: {e}'},ensure_ascii=False),flush=True)
    finally:
        import sys
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
