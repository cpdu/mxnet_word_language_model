import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description='Mxnet scheduler or worker for distributed training.')
parser.add_argument('--role', type=str, default='server',
                    help='scheduler IP')
parser.add_argument('--ip', type=str, default='127.0.0.1',
                    help='scheduler IP')
parser.add_argument('--port', type=str, default='9000',
                    help='scheduler port')
parser.add_argument('--num-server', type=str, default='1',
                    help='number of servers')
parser.add_argument('--num-worker', type=str, default='1',
                    help='number of workers')
parser.add_argument('--verbose', type=str, default='2',
                    help='log verbose')
args = parser.parse_args()

env = os.environ.copy()
env.update({"DMLC_ROLE": args.role,
            "DMLC_PS_ROOT_PORT": args.port,
            "DMLC_PS_ROOT_URI": args.ip,
            "DMLC_NUM_SERVER": args.num_server,
            "DMLC_NUM_WORKER": args.num_worker,
            "PS_VERBOSE": args.verbose})
proc = subprocess.Popen("python -c 'import mxnet'", shell=True, env=env)
proc.wait()
