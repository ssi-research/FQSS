import subprocess as sp
import sys
import time
import torch
from train_env.tasnet_musdbhq.musdbhq_utils import free_port


def train(yml_path, device):

    # Fill args
    args = ['--yml_path', yml_path,
            '--device', device]

    # ------------------------------------------
    # Multi GPUs
    # ------------------------------------------
    gpus = torch.cuda.device_count()
    port = free_port()
    args += ["--world_size", str(gpus), "--master", f"127.0.0.1:{port}"]

    tasks = []
    for gpu in range(gpus):
        kwargs = {}
        if gpu > 0:
            kwargs['stdin'] = sp.DEVNULL
            kwargs['stdout'] = sp.DEVNULL
            # We keep stderr to see tracebacks from children.
        # Create GPU task
        tasks.append(sp.Popen(["python3", "train_env/tasnet_musdbhq/musdbhq_train.py"] + args + ["--rank", str(gpu)], **kwargs))
        tasks[-1].rank = gpu

    failed = False
    try:
        while tasks:
            for task in tasks:
                try:
                    exitcode = task.wait(0.1)
                except sp.TimeoutExpired:
                    continue
                else:
                    tasks.remove(task)
                    if exitcode:
                        print(f"Task {task.rank} died with exit code "
                              f"{exitcode}",
                              file=sys.stderr)
                        failed = True
            if failed:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        for task in tasks:
            task.terminate()
        raise
    if failed:
        for task in tasks:
            task.terminate()
        sys.exit(1)

