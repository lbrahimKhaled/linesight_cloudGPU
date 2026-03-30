"""
Run TrackMania collectors locally and stream rollouts to a remote learner.
"""

import argparse
import ctypes
import os
import shutil
import signal
import socket
import sys
from pathlib import Path

import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock


def copy_configuration_file():
    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "config.py",
        base_dir / "config_files" / "config_copy.py",
    )


def clear_tm_instances(is_linux: bool):
    if is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill /F /IM TmForever.exe")


if __name__ == "__main__":
    copy_configuration_file()

    from config_files import config_copy
    from trackmania_rl.multiprocess.collector_process import collector_process_fn

    parser = argparse.ArgumentParser(description="Run Linesight TrackMania collectors and connect them to a remote learner.")
    parser.add_argument("--learner-host", required=True, help="IP address or hostname of the learner machine.")
    parser.add_argument("--learner-port", type=int, default=9100, help="TCP port exposed by the learner.")
    parser.add_argument("--auth-token", required=True, help="Shared secret expected by the learner.")
    parser.add_argument(
        "--collector-name-prefix",
        default=socket.gethostname(),
        help="Name prefix used for collector registrations on the learner.",
    )
    args = parser.parse_args()

    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    game_spawning_lock = Lock()
    collector_processes = []

    def signal_handler(sig, frame):
        print("Received SIGINT signal. Killing all open Trackmania instances.")
        clear_tm_instances(config_copy.is_linux)
        for child in mp.active_children():
            child.kill()
        tprint("Bye bye!", font="tarty1")
        sys.exit()

    signal.signal(signal.SIGINT, signal_handler)

    clear_tm_instances(config_copy.is_linux)

    base_dir = Path(__file__).resolve().parents[1]

    shutil.copyfile(
        base_dir / "trackmania_rl" / "tmi_interaction" / "Python_Link.as",
        config_copy.target_python_link_path,
    )

    if config_copy.is_linux:
        os.system(f"chmod +x {config_copy.linux_launch_game_path}")

    remote_collector_config = {
        "host": args.learner_host,
        "port": args.learner_port,
        "auth_token": args.auth_token,
        "collector_name_prefix": args.collector_name_prefix,
    }

    print("Run:\n\n")
    tprint(config_copy.run_name, font="tarty4")
    print("\n" * 2)
    tprint("Linesight", font="tarty1")
    print("\n" * 2)
    print("Remote collector mode is starting!")

    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                None,
                None,
                None,
                game_spawning_lock,
                shared_steps,
                base_dir,
                base_dir / "save" / config_copy.run_name,
                config_copy.base_tmi_port + process_number,
                process_number,
                remote_collector_config,
            ),
        )
        for process_number in range(config_copy.gpu_collectors_count)
    ]

    for collector_process in collector_processes:
        collector_process.start()

    for collector_process in collector_processes:
        collector_process.join()
