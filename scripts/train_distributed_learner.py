"""
Run the learner on one machine and accept remote collectors over TCP.
"""

import argparse
import ctypes
import queue
import shutil
from pathlib import Path

import torch.multiprocessing as mp


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


if __name__ == "__main__":
    copy_configuration_file()

    from config_files import config_copy
    from trackmania_rl.device import configure_torch_runtime, resolve_torch_device
    from trackmania_rl.distributed.training_hub import RemoteLearnerHub
    from trackmania_rl.multiprocess.learner_process import learner_process_fn

    parser = argparse.ArgumentParser(description="Run the Linesight learner and accept remote TrackMania collectors.")
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind the learner server to.")
    parser.add_argument("--port", type=int, default=9100, help="TCP port for remote collectors.")
    parser.add_argument("--auth-token", required=True, help="Shared secret required by collectors.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device for the learner.")
    args = parser.parse_args()

    learner_device = resolve_torch_device(args.device)
    configure_torch_runtime(learner_device)

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / config_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_base_dir = base_dir / "tensorboard"

    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queue = queue.Queue(max(1, config_copy.gpu_collectors_count * config_copy.max_rollout_queue_size))

    hub = RemoteLearnerHub(rollout_queue=rollout_queue, auth_token=args.auth_token)
    hub.start(args.host, args.port)
    print(f"Remote learner listening on {args.host}:{args.port}")

    try:
        learner_process_fn(
            [rollout_queue],
            None,
            None,
            shared_steps,
            base_dir,
            save_dir,
            tensorboard_base_dir,
            learner_device=learner_device,
            network_publisher=hub.publish_network,
        )
    finally:
        hub.close()
