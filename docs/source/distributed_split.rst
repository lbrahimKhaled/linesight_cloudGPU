=========================
Distributed Split Deploy
=========================

This page documents the split deployment for Linesight where:

- the Windows Lenovo runs TrackMania and the collector processes locally
- the Apple Silicon Mac runs the learner on the `mps` backend
- the two machines exchange weights and rollout data over TCP

This setup is intended for a trusted local network. The transport is authenticated with a shared token, but it is not encrypted. Do not expose the learner port to the public internet.

Recommended hardware split
--------------------------

The Mac should be used for the learner because Apple Silicon can run the learner on the `mps` backend.

If you later move the learner to an Nvidia machine, CUDA is still the stronger option. On the Mac side, `mps` is the best native backend for this repo.

Setup assumptions
-----------------

Before starting, make sure both machines have:

- the same Linesight checkout and the same commit
- the same `config_files/config.py`
- the same `run_name`
- access to the same maps and the same relative map paths referenced by the config

Machine-specific settings still need to be correct in `config_files/user_config.py`:

- on Windows, point to the local TrackMania installation, TMInterface plugin path, and TMLoader setup
- on macOS, keep the repo paths valid for the local user account; TrackMania itself is not required on the Mac

The code copies `config_files/config.py` to `config_files/config_copy.py` at launch. If you change training hyperparameters, restart both machines so they reload the same config snapshot.

Mac learner
-----------

Start the learner first on the Mac:

.. code-block:: bash

    python scripts/train_distributed_learner.py --device mps --host 0.0.0.0 --port 9100 --auth-token YOUR_SHARED_TOKEN

Notes:

- `--device mps` forces the learner to use Apple Silicon GPU acceleration
- `--host 0.0.0.0` binds the learner server to all interfaces on the Mac
- `--port 9100` is the default listener port used by the collector machine
- `--auth-token` must match the token you pass to the Windows collector command

The Mac process owns:

- the online and target networks
- replay buffer sampling and training
- checkpoints in `save/{run_name}/`
- TensorBoard logs in `tensorboard/`

Windows Lenovo collectors
-------------------------

On the Windows machine, start the collector runner after the learner is listening:

.. code-block:: bash

    python scripts/run_distributed_collectors.py --learner-host MAC_IP_ADDRESS --learner-port 9100 --auth-token YOUR_SHARED_TOKEN

Notes:

- `MAC_IP_ADDRESS` should be the Mac's LAN IP address or a hostname that the Windows machine can resolve
- the auth token must be identical to the one used on the Mac
- the Windows machine keeps running TrackMania instances locally
- each collector uses its own local TMInterface port from `base_tmi_port`

The Windows process owns:

- game launch and restart
- TMInterface interaction
- frame capture and action execution
- rollout generation

Network and firewall requirements
---------------------------------

Open the learner port on the Mac so the Windows machine can reach it.

Keep the deployment on a trusted LAN or VPN. The transport uses authenticated Python serialization and should not be exposed to the public internet.

Config sync expectations
------------------------

The following settings should match on both machines:

- `run_name`
- map cycle and map paths
- training hyperparameters that affect rollout generation or learning
- `gpu_collectors_count`

The following settings are machine-specific:

- Windows TrackMania, TMInterface, and TMLoader paths
- the Mac learner bind address, TCP port, and shared auth token used to accept collectors

Verification
------------

Use this sequence to confirm the split is working:

1. Start the Mac learner and confirm it reports `Learner device: mps`.
2. Confirm the Mac prints `Remote learner listening on ...:9100`.
3. Start the Windows collector command and confirm it launches TrackMania instances.
4. Check that rollouts begin and that the Mac starts writing TensorBoard scalars.
5. Verify that checkpoints appear under `save/{run_name}/` on the Mac.
6. If you are using more than one collector, confirm each game instance is using a distinct TMInterface port starting at `base_tmi_port`.

If the collector cannot connect, check the Mac firewall, the IP address, and the shared token first.

Operational notes
-----------------

- Start the learner before the collectors.
- Restart both machines after editing `config_files/config.py`.
- Treat the token as a secret and rotate it if it is ever exposed.
- If MPS hits an unsupported operator, PyTorch fallback is enabled automatically on the Mac.
