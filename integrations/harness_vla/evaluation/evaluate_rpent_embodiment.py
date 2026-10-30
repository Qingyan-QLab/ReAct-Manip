#!/usr/bin/env python3
"""Run a separate RPent embodiment experiment, or a bounded RGB-D diagnostic.

This generates fresh scenes; it does not replay or overwrite the PiPER-X bank.
Results are experimental and cannot be reported as the original bank score.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def diagnostic_session(env,context,memory):
    from integrations.rpent_embodiments.control import hold_native,normalize
    while not env.cancelled.is_set():
        obs=env.current.observation
        for camera,depth in obs['depth_m'].items():
            if not obs['depth_valid'][camera].any():raise ValueError(f'No valid depth in {camera}')
        _,_,term,trunc,_=env.step(normalize(hold_native(obs),obs['control_contract']))
        if term or trunc:break


def main(argv=None):
    from integrations.rpent_embodiments.catalog import EMBODIMENTS
    from integrations.rpent_embodiments.catalog import SUPPORTED_TASKS as ROBOT_TASKS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',required=True,choices=tuple(EMBODIMENTS))
    parser.add_argument('--task',required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--smoke-steps',type=int)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():parser.error('--out must be a new path')
    if args.smoke_steps is not None and args.smoke_steps<1:parser.error('--smoke-steps must be positive')
    if args.task not in ROBOT_TASKS[args.robot]:parser.error(f'Available tasks for this robot: {ROBOT_TASKS[args.robot]}')
    config=json.loads(args.config.read_text()) if args.config else {}
    if args.smoke_steps is None:
        if not config.get('planner_model') or not config.get('vla',{}).get('checkpoint_id'):
            parser.error('Real evaluation requires planner_model and vla.checkpoint_id; no dummy policy fallback')
        if config.get('vla',{}).get('robot_id')!=args.robot:parser.error('Config VLA robot_id must match --robot')
        if config.get('rpent_root'):sys.path.insert(0,str(Path(config['rpent_root']).expanduser().resolve()))
        config['vla_factory']='integrations.rpent_embodiments.policy:build'
    from integrations.rpent_embodiments.environment import EmbodimentEnv
    from integrations.harness_vla.bridge import HarnessVLAPolicy
    from react_manip.core.runner import BenchmarkRunner
    from react_manip.evaluation.protocols import _EpisodePolicy
    from react_manip.instances.runtime import _model_digest
    from react_manip.instances.records import SUPPORTED_TASKS
    if args.task not in SUPPORTED_TASKS:parser.error('Unknown benchmark task')
    output=dict(track='rpent_embodiments_experimental_rgbd',robot_id=args.robot,task_id=args.task,
                mode='smoke' if args.smoke_steps else 'practice_exam',valid=False,
                official_score=None,protocol_score=None,experimental_exam_score=None,episodes=[],
                limitations=['Fresh scenes, not the frozen PiPER-X candidate bank',
                             'Model/control ports do not establish task solvability or checkpoint performance'])
    paths=list((ROOT/'integrations/rpent_embodiments').glob('*.py'))+list((ROOT/'integrations/harness_vla').glob('*.py'))
    paths += [ROOT/'integrations/rpent_embodiments/assets.lock.json',Path(__file__)]
    paths += list((ROOT/'react_manip').rglob('*.py'))+list((ROOT/'tasks'/args.task).rglob('*.py'))
    paths += [ROOT/'tasks'/args.task/'task_spec.json',ROOT/'tasks'/args.task/'instance_contract.json',ROOT/'integrations/rpent_embodiments/tools/build_assets.py']
    output['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    policy=None;env=None
    try:
        if args.smoke_steps:
            policy=HarnessVLAPolicy(diagnostic_session,action_timeout_s=30)
            policy.policy_id='rpent-embodiment-hold-diagnostic'
        else:
            from integrations.harness_vla.rpent_session import build
            directory=args.out.parent/(args.out.stem+'_sessions')
            if directory.exists():raise FileExistsError(directory)
            policy=build(config,directory)
            output['policy_provenance']=policy.provenance
            output['execution_settings']={k:config['vla'].get(k) for k in ('control_hz','n_action_steps','expected_metadata')}
        scenario=None
        contract=json.loads((ROOT/'tasks'/args.task/'instance_contract.json').read_text())
        for index in range(1 if args.smoke_steps else 15):
            env=EmbodimentEnv(args.task,args.robot,seed=args.seed+index,scenario=deepcopy(scenario))
            if index==0:
                scenario=deepcopy(getattr(env,'scenario',None))
                output['model_digest']=_model_digest(env.env)
                output['control_hz']=float(env.env.control_freq)
                output['robot_base_position']=env.env.robots[0].base_pos.copy()
                output['scenario']=getattr(scenario,'to_dict',lambda:repr(scenario))()
            phase='practice' if index<10 else 'exam'
            session=_EpisodePolicy(policy,phase,index if index<10 else index-10,index)
            spec=env.definition.spec
            row=BenchmarkRunner().evaluate(env.definition,policy=session,robot_id=args.robot,
                seed=args.seed+index,env=env,stop_on_done=True,
                horizon_sec=args.smoke_steps/spec.control_hz if args.smoke_steps else spec.horizon_sec,
                score_key=contract['evaluation']['score_key']).to_dict()
            row.update(phase=phase,episode_index=index)
            output['episodes'].append(row)
            env.close();env=None
            if 'error' in row:raise RuntimeError(row['error'])
        output['valid']=True
        if not args.smoke_steps:
            output['experimental_exam_score']=sum(x['episode_score'] for x in output['episodes'][10:])/5
    except Exception as exc:
        output['error']=f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if env is not None:env.close()
            if policy is not None:
                if hasattr(policy,'shutdown'):policy.shutdown()
                else:policy.close()
        except Exception as exc:
            output.update(valid=False,error=f'Cleanup failed: {exc}',experimental_exam_score=None)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(output,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
    print(json.dumps(dict(valid=output['valid'],out=str(args.out),error=output.get('error')),ensure_ascii=False))
    return 0 if output['valid'] else 1


if __name__=='__main__':raise SystemExit(main())
