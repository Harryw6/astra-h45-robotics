"""Offline dataset operations. This CLI contains no robot-control endpoint."""
import argparse
import json
from pathlib import Path

from .data import read_json, write_json


def compile_library_contexts(library, *, progress=print):
    from .contexts import compile_demo
    library=Path(library).resolve()
    manifest=read_json(library/'manifest.json')
    if not manifest.get('episodes'):raise ValueError('Not a populated library')
    for episode in manifest['episodes']:
        source=episode.get('source_demo')
        if not source or episode.get('context'):continue
        progress('Compiling native GPT-Policy context: '+episode['episode_id'])
        destination=library/'contexts'/episode['episode_id']
        result=compile_demo(library/source,destination,
            'Review proposed Piper actions against current observations and historical demonstrations. '
            'Identify visible task deviations. Historical measured states are numeric references, not issued controls. '
            'Task outcomes and hardware acknowledgements are unverified. Do not invent missing observations.')
        episode['context']={key:str(Path(value).relative_to(library)) for key,value in result.items()}
        write_json(library/'episodes'/episode['episode_id']/'episode.json',episode)
        manifest['compiled_context_count']=sum(bool(e.get('context')) for e in manifest['episodes'])
        write_json(library/'manifest.json',manifest)
    return {'contexts':manifest['compiled_context_count']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build');build.add_argument('--source',required=True);build.add_argument('--output',required=True)
    build.add_argument('--no-contexts',action='store_true');build.add_argument('--no-images',action='store_true')
    for name in ('index','compile-contexts','stats'):
        p=sub.add_parser(name);p.add_argument('--library',required=True)
    review=sub.add_parser('review',help='Make a real Astra call through native GPT-Policy; offline output only')
    review.add_argument('--request',required=True);review.add_argument('--output',required=True)
    review.add_argument('--effort',choices=['low','medium','high','xhigh','max'],default='high')
    review.add_argument('--timeout-s',type=float,default=600.)
    memory_review=sub.add_parser('prepare-memory-review',help='Prepare current images with retrieved native demonstrations; no model call')
    memory_review.add_argument('--library',required=True);memory_review.add_argument('--memory',required=True)
    memory_review.add_argument('--task-id',required=True);memory_review.add_argument('--current-request',required=True)
    memory_review.add_argument('--output',required=True);memory_review.add_argument('-k',type=int,default=1)
    memory_review.add_argument('--multiview-inspection-root',help='Optional existing per-episode inspection sidecars; unavailable episodes retain original RGB')
    for command in ('query','select'):
        query=sub.add_parser(command);query.add_argument('--library',required=True)
        query.add_argument('--joints',nargs=6,type=float);query.add_argument('--episode');query.add_argument('--date')
        query.add_argument('--event');query.add_argument('--at-s',type=float);query.add_argument('-k',type=int,default=5)
        if command=='select':
            query.add_argument('--output',required=True)
            query.add_argument('--instruction',default='Review proposed Piper actions against current observations and these historical measured demonstrations. Identify visible task deviations; do not treat historical measurements as issued controls or infer task success.')
    args=parser.parse_args()
    if args.command=='build':
        from .builder import build_library
        result=build_library(args.source,args.output,extract_images=not args.no_images,
                             compile_contexts=not args.no_contexts,progress=lambda s:print(s,flush=True))
    elif args.command=='index':
        from .retrieval import build_index
        result=build_index(args.library)
    elif args.command=='compile-contexts':
        result=compile_library_contexts(args.library,progress=lambda s:print(s,flush=True))
    elif args.command=='review':
        from .astra import run_review
        result=run_review(args.request,args.output,effort=args.effort,timeout_s=args.timeout_s)
    elif args.command=='prepare-memory-review':
        from .memory import TaskMemory
        from .memory_request import prepare_memory_review
        result=prepare_memory_review(TaskMemory(args.library,args.memory),args.task_id,
                                     args.current_request,args.output,k=args.k,
                                     multiview_inspection_root=args.multiview_inspection_root)
    elif args.command in ('query','select'):
        from .retrieval import query
        result=query(args.library,joint_positions=args.joints,episode_id=args.episode,date=args.date,
                     event=args.event,at_s=args.at_s,k=args.k)
        if args.command=='select':
            from .selection import export_selection
            result=export_selection(args.library,result,args.output,args.instruction)
    else:
        manifest=read_json(Path(args.library)/'manifest.json')
        result={key:manifest[key] for key in ('source_files','source_bytes','block_count','image_count',
                                            'compiled_context_count','source_verification')}
        result['episodes']=len(manifest['episodes'])
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
