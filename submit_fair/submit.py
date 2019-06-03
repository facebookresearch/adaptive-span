import os
import argparse
import uuid
from pathlib import Path
import submitit
from main import get_parser, SubmititMain

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='')
parser.add_argument('--folder', type=str, default='', help='')
parser.add_argument('--partition', type=str, default='learnfair', help='')
parser.add_argument('--ngpus', type=int, default=8, help='')
parser.add_argument('--nodes', type=int, default=1, help='')
parser.add_argument('--constraint', type=str, default='volta32gb', help='')
parser.add_argument('--args', type=str, default='', help='')
args = parser.parse_args()

job_args = args.args.split()
folder = Path(args.folder)
os.makedirs(str(folder), exist_ok=True)
init_file = folder / f"{uuid.uuid4().hex}_init" #not used when nodes=1
job_args += ['--submitit', '--dist-init', init_file.as_uri()]
job_parser = get_parser()
job_args = job_parser.parse_args(job_args)

executor = submitit.AutoExecutor(folder=folder / "%j", max_num_timeout=10)
executor.update_parameters(
    mem_gb=128,
    gpus_per_node=args.ngpus,
    tasks_per_node=args.ngpus,  # one task per GPU
    cpus_per_task=2,
    nodes=args.nodes,
    timeout_min=4320,
    # Below are cluster dependent parameters
    partition=args.partition,
    signal_delay_s=120,
    constraint=args.constraint,
)
if args.partition == 'priority':
    executor.update_parameters(
        comment='NIPS deadline May 23rd'
    )

executor.update_parameters(name=args.name)
main = SubmititMain()
job = executor.submit(main, job_args)
print('submited {} {}'.format(job.job_id, args.name))
