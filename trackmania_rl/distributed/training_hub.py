"""
Network transport between remote collectors and the learner.
"""

from __future__ import annotations

import io
import pickle
import socket
import struct
import threading
import time
from collections import deque
from typing import Any

import torch

from trackmania_rl.device import state_dict_to_cpu

_HEADER = struct.Struct("!Q")


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        data = sock.recv(length - len(chunks))
        if not data:
            raise EOFError("Socket closed while receiving a message.")
        chunks.extend(data)
    return bytes(chunks)


def _send_message(sock: socket.socket, payload: Any):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def _recv_message(sock: socket.socket):
    message_length = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, message_length))


def _serialize_state_dict(state_dict: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict_to_cpu(state_dict), buffer)
    return buffer.getvalue()


def _deserialize_state_dict(state_dict_bytes: bytes) -> dict:
    return torch.load(io.BytesIO(state_dict_bytes), map_location="cpu", weights_only=False)


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
                        self.rollout_queue.put(request["payload"])
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
        self._next_rollout_id = 0
        self._last_shared_steps = 0

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def wait_for_initial_weights(self):
        while True:
            weights_version, state_dict, shared_steps = self.pull_weights(-1, max_attempts=None)
            if state_dict is not None:
                return weights_version, state_dict, shared_steps
            time.sleep(1)

    def pull_weights(self, known_version: int, max_attempts: int | None = 2):
        response = self._request(
            {"type": "pull_weights", "known_version": known_version},
            max_attempts=max_attempts,
        )
        if not response["ready"]:
            self._last_shared_steps = response["shared_steps"]
            return response["weights_version"], None, response["shared_steps"]
        state_dict = None
        if "state_dict" in response:
            state_dict = _deserialize_state_dict(response["state_dict"])
        self._last_shared_steps = response["shared_steps"]
        return response["weights_version"], state_dict, response["shared_steps"]

    def submit_rollout(self, payload):
        rollout_id = f"{self.collector_name}:{self._next_rollout_id}"
        self._next_rollout_id += 1
        self._request(
            {
                "type": "submit_rollout",
                "rollout_id": rollout_id,
                "payload": payload,
            },
            max_attempts=None,
        )

    def _connect(self):
        self.close()
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
        self.sock = sock

    def _request(self, payload, max_attempts: int | None = 2):
        attempts = 0
        while True:
            attempts += 1
            try:
                if self.sock is None:
                    self._connect()
                _send_message(self.sock, payload)
                response = _recv_message(self.sock)
                if not response.get("ok", False):
                    raise RuntimeError(response.get("error", "Unknown remote learner error"))
                return response
            except (EOFError, OSError):
                self.close()
                if max_attempts is not None and attempts >= max_attempts:
                    raise
                time.sleep(1)
