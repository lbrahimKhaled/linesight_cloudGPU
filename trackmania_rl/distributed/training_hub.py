"""
Network transport between remote collectors and the learner.
"""

from __future__ import annotations

import io
import pickle
import queue
import socket
import struct
import threading
import time
import zlib
from collections import deque
from typing import Any

import numpy as np
import torch

from trackmania_rl.device import state_dict_to_cpu

_HEADER = struct.Struct("!BQ")
_MESSAGE_FLAG_COMPRESSED = 1
_MESSAGE_COMPRESSION_LEVEL = 1
_MESSAGE_COMPRESSION_THRESHOLD_BYTES = 256 * 1024
_WEIGHTS_PULL_INTERVAL_S = 0.25


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    buffer = bytearray(length)
    view = memoryview(buffer)
    bytes_received = 0
    while bytes_received < length:
        received_now = sock.recv_into(view[bytes_received:])
        if received_now == 0:
            raise EOFError("Socket closed while receiving a message.")
        bytes_received += received_now
    return buffer


def _send_message(sock: socket.socket, payload: Any):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    flags = 0
    if len(data) >= _MESSAGE_COMPRESSION_THRESHOLD_BYTES:
        compressed_data = zlib.compress(data, level=_MESSAGE_COMPRESSION_LEVEL)
        if len(compressed_data) < len(data):
            data = compressed_data
            flags |= _MESSAGE_FLAG_COMPRESSED
    sock.sendall(_HEADER.pack(flags, len(data)))
    sock.sendall(data)


def _recv_message(sock: socket.socket):
    flags, message_length = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    payload = _recv_exact(sock, message_length)
    if flags & _MESSAGE_FLAG_COMPRESSED:
        payload = zlib.decompress(payload)
    return pickle.loads(payload)


def _serialize_state_dict(state_dict: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict_to_cpu(state_dict), buffer)
    return buffer.getvalue()


def _deserialize_state_dict(state_dict_bytes: bytes) -> dict:
    return torch.load(io.BytesIO(state_dict_bytes), map_location="cpu", weights_only=False)


def _pad_rollout_frames(frames, target_length: int) -> np.ndarray:
    stacked_frames = np.ascontiguousarray(np.stack(frames))
    if stacked_frames.shape[0] == target_length:
        return stacked_frames

    padded_frames = np.zeros((target_length, *stacked_frames.shape[1:]), dtype=stacked_frames.dtype)
    padded_frames[: stacked_frames.shape[0]] = stacked_frames
    return padded_frames


def _pad_rollout_scalars(values, target_length: int, dtype, pad_value) -> np.ndarray:
    padded_values = np.empty(target_length, dtype=dtype)
    packed_values = np.asarray(values, dtype=dtype)
    padded_values[: len(packed_values)] = packed_values
    if len(packed_values) < target_length:
        padded_values[len(packed_values) :] = pad_value
    return padded_values


def _pack_rollout_payload(payload):
    rollout_results, end_race_stats, fill_buffer, is_explo, map_name, map_status, rollout_duration, loop_number = payload

    race_finished = "race_time" in rollout_results
    actual_frame_count = len(rollout_results["frames"]) - int(race_finished)
    if actual_frame_count <= 0:
        return payload

    transport_frame_count = actual_frame_count + int(race_finished)
    packed_rollout_results = {
        "_packed_transport_version": 2,
        "furthest_zone_idx": rollout_results["furthest_zone_idx"],
        "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
        "current_zone_idx": np.ascontiguousarray(np.asarray(rollout_results["current_zone_idx"], dtype=np.int32)),
        "frames": _pad_rollout_frames(rollout_results["frames"][:actual_frame_count], transport_frame_count),
        "actions": _pad_rollout_scalars(
            rollout_results["actions"][:actual_frame_count],
            transport_frame_count,
            np.int32,
            -1,
        ),
        "action_was_greedy": _pad_rollout_scalars(
            rollout_results["action_was_greedy"][:actual_frame_count],
            transport_frame_count,
            np.bool_,
            False,
        ),
        "meters_advanced_along_centerline": np.ascontiguousarray(np.asarray(
            rollout_results["meters_advanced_along_centerline"],
            dtype=np.float32,
        )),
        "state_float": np.ascontiguousarray(np.asarray(rollout_results["state_float"], dtype=np.float32)),
        "q_values": np.ascontiguousarray(np.asarray(rollout_results["q_values"], dtype=np.float32)),
    }
    if race_finished:
        packed_rollout_results["race_time"] = rollout_results["race_time"]

    return (
        packed_rollout_results,
        end_race_stats,
        fill_buffer,
        is_explo,
        map_name,
        map_status,
        rollout_duration,
        loop_number,
    )


def _unpack_rollout_payload(payload):
    rollout_results, end_race_stats, fill_buffer, is_explo, map_name, map_status, rollout_duration, loop_number = payload
    transport_version = rollout_results.get("_packed_transport_version")
    if transport_version == 2:
        return payload

    if not rollout_results.get("_packed_transport_v1"):
        return payload

    race_finished = rollout_results["race_time"] is not None
    frames = [frame for frame in rollout_results["frames"]]
    actions = rollout_results["actions"].tolist()
    action_was_greedy = rollout_results["action_was_greedy"].tolist()

    if race_finished:
        frames.append(np.nan)
        actions.append(np.nan)
        action_was_greedy.append(np.nan)

    unpacked_rollout_results = {
        "current_zone_idx": rollout_results["current_zone_idx"],
        "frames": frames,
        "actions": actions,
        "action_was_greedy": action_was_greedy,
        "q_values": rollout_results["q_values"],
        "meters_advanced_along_centerline": rollout_results["meters_advanced_along_centerline"],
        "state_float": rollout_results["state_float"],
        "furthest_zone_idx": rollout_results["furthest_zone_idx"],
        "worker_time_in_rollout_percentage": rollout_results["worker_time_in_rollout_percentage"],
    }

    if race_finished:
        unpacked_rollout_results["race_time"] = rollout_results["race_time"]

    return (
        unpacked_rollout_results,
        end_race_stats,
        fill_buffer,
        is_explo,
        map_name,
        map_status,
        rollout_duration,
        loop_number,
    )


class RemoteLearnerHub:
    def __init__(self, rollout_queue, auth_token: str):
        self.rollout_queue = rollout_queue
        self.auth_token = auth_token
        self._lock = threading.Lock()
        self._weights_version = -1
        self._weights_payload = None
        self._shared_steps = 0
        self._server_socket = None
        self._stop_event = threading.Event()
        self._accept_thread = None
        self._seen_rollout_ids = set()
        self._seen_rollout_ids_order = deque()
        self._max_seen_rollout_ids = 4096

    def start(self, host: str, port: int):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((host, port))
        self._server_socket.listen()
        self._accept_thread = threading.Thread(target=self._accept_loop, name="remote-learner-hub", daemon=True)
        self._accept_thread.start()

    def close(self):
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass

    def publish_network(self, state_dict: dict, shared_steps: int):
        payload = _serialize_state_dict(state_dict)
        with self._lock:
            self._weights_version += 1
            self._weights_payload = payload
            self._shared_steps = shared_steps

    def _accept_loop(self):
        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except OSError:
                if self._stop_event.is_set():
                    return
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket):
        authenticated = False
        try:
            while not self._stop_event.is_set():
                request = _recv_message(conn)
                request_type = request.get("type")

                if request_type == "hello":
                    authenticated = request.get("auth_token") == self.auth_token
                    _send_message(conn, {"ok": authenticated})
                    if not authenticated:
                        return
                    continue

                if not authenticated:
                    _send_message(conn, {"ok": False, "error": "Not authenticated"})
                    return

                if request_type == "pull_weights":
                    known_version = request.get("known_version", -1)
                    with self._lock:
                        if self._weights_payload is None:
                            response = {
                                "ok": True,
                                "ready": False,
                                "shared_steps": self._shared_steps,
                                "weights_version": self._weights_version,
                            }
                        else:
                            response = {
                                "ok": True,
                                "ready": True,
                                "shared_steps": self._shared_steps,
                                "weights_version": self._weights_version,
                            }
                            if known_version != self._weights_version:
                                response["state_dict"] = self._weights_payload
                    _send_message(conn, response)
                elif request_type == "submit_rollout":
                    rollout_id = request.get("rollout_id")
                    should_enqueue = True
                    if rollout_id is not None:
                        with self._lock:
                            should_enqueue = rollout_id not in self._seen_rollout_ids
                            if should_enqueue:
                                self._seen_rollout_ids.add(rollout_id)
                                self._seen_rollout_ids_order.append(rollout_id)
                                while len(self._seen_rollout_ids_order) > self._max_seen_rollout_ids:
                                    oldest_rollout_id = self._seen_rollout_ids_order.popleft()
                                    self._seen_rollout_ids.discard(oldest_rollout_id)
                    if should_enqueue:
                        self.rollout_queue.put(_unpack_rollout_payload(request["payload"]))
                    _send_message(conn, {"ok": True})
                else:
                    _send_message(conn, {"ok": False, "error": f"Unsupported request: {request_type}"})
                    return
        except (EOFError, OSError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass


class RemoteCollectorSession:
    def __init__(
        self,
        host: str,
        port: int,
        auth_token: str,
        collector_name: str,
        connect_timeout_s: int = 30,
        request_timeout_s: int = 120,
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.collector_name = collector_name
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s
        self.sock = None
        self._submit_sock = None
        self._next_rollout_id = 0
        self._last_shared_steps = 0
        self._latest_weights_version = -1
        self._latest_state_dict = None
        self._submit_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._weights_condition = threading.Condition()
        self._submit_thread = threading.Thread(
            target=self._submit_loop,
            name=f"remote-rollout-submit-{collector_name}",
            daemon=True,
        )
        self._weights_thread = threading.Thread(
            target=self._weights_loop,
            name=f"remote-weights-pull-{collector_name}",
            daemon=True,
        )
        self._submit_thread.start()
        self._weights_thread.start()

    def close(self):
        self._stop_event.set()
        try:
            self._submit_queue.put_nowait(None)
        except queue.Full:
            pass
        self._close_socket("control")
        self._close_socket("submit")
        with self._weights_condition:
            self._weights_condition.notify_all()

    def _close_socket(self, channel: str):
        socket_name = "sock" if channel == "control" else "_submit_sock"
        sock = getattr(self, socket_name)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            setattr(self, socket_name, None)

    def wait_for_initial_weights(self):
        while True:
            with self._weights_condition:
                if self._latest_state_dict is not None:
                    state_dict = self._latest_state_dict
                    self._latest_state_dict = None
                    return self._latest_weights_version, state_dict, self._last_shared_steps
                if self._stop_event.is_set():
                    raise RuntimeError("Remote collector session stopped before initial weights arrived.")
                self._weights_condition.wait(timeout=1)

    def pull_weights(self, known_version: int, max_attempts: int | None = 2):
        del max_attempts
        with self._weights_condition:
            latest_version = max(known_version, self._latest_weights_version)
            if self._latest_state_dict is None or self._latest_weights_version <= known_version:
                return latest_version, None, self._last_shared_steps

            state_dict = self._latest_state_dict
            self._latest_state_dict = None
            return self._latest_weights_version, state_dict, self._last_shared_steps

    def submit_rollout(self, payload):
        rollout_id = f"{self.collector_name}:{self._next_rollout_id}"
        self._next_rollout_id += 1
        self._submit_queue.put((rollout_id, payload))

    def _connect(self, channel: str):
        self._close_socket(channel)
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        sock.settimeout(self.request_timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_message(
            sock,
            {
                "type": "hello",
                "auth_token": self.auth_token,
                "collector_name": self.collector_name,
            },
        )
        response = _recv_message(sock)
        if not response.get("ok"):
            sock.close()
            raise RuntimeError("Authentication to remote learner failed.")
        if channel == "control":
            self.sock = sock
        else:
            self._submit_sock = sock

    def _request(self, payload, max_attempts: int | None = 2, channel: str = "control"):
        attempts = 0
        socket_name = "sock" if channel == "control" else "_submit_sock"
        while True:
            attempts += 1
            try:
                if getattr(self, socket_name) is None:
                    self._connect(channel)
                current_sock = getattr(self, socket_name)
                _send_message(current_sock, payload)
                response = _recv_message(current_sock)
                if not response.get("ok", False):
                    raise RuntimeError(response.get("error", "Unknown remote learner error"))
                return response
            except (EOFError, OSError):
                self._close_socket(channel)
                if max_attempts is not None and attempts >= max_attempts:
                    raise
                time.sleep(1)

    def _submit_loop(self):
        while not self._stop_event.is_set():
            queued_item = self._submit_queue.get()
            if queued_item is None:
                return
            rollout_id, payload = queued_item
            self._request(
                {
                    "type": "submit_rollout",
                    "rollout_id": rollout_id,
                    "payload": _pack_rollout_payload(payload),
                },
                max_attempts=None,
                channel="submit",
            )

    def _weights_loop(self):
        known_version = -1
        while not self._stop_event.is_set():
            response = self._request(
                {"type": "pull_weights", "known_version": known_version},
                max_attempts=None,
                channel="control",
            )

            state_dict = None
            response_version = response["weights_version"]
            if response.get("ready") and "state_dict" in response:
                state_dict = _deserialize_state_dict(response["state_dict"])
                known_version = response_version
            elif response_version > known_version:
                known_version = response_version

            with self._weights_condition:
                self._last_shared_steps = response["shared_steps"]
                if state_dict is not None and response_version >= self._latest_weights_version:
                    self._latest_weights_version = response_version
                    self._latest_state_dict = state_dict
                self._weights_condition.notify_all()

            if self._stop_event.wait(_WEIGHTS_PULL_INTERVAL_S):
                return
