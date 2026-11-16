"""Socket manager for inter-component communication.

Supports dual-pattern communication:
- REQ/REP: Non-blocking ROUTER (server) / DEALER (client) pattern for async task dispatch and status updates
- PUB/SUB: Publish-Subscribe pattern for status/metrics broadcasting
"""

import time
import json
import queue
import dataclasses
from threading import Thread, Lock, Event
from typing import List, Dict, Any, Optional

import zmq

from core.logger import get_logger
from src.network.message_types import (
    Message,
    MessageTypes,
    TaskMessage,
    TensorMessage,
    ControlMessage,
    MetricMessage,
    WorkerRegistration,
)


def deserialize_message(data: Dict[str, Any]) -> Message:
    """Deserialize raw dictionary into strongly-typed Message subclass."""
    msg_type = data.get("message_type")

    cls_map = {
        MessageTypes.TASK_ASSIGNMENT: TaskMessage,
        MessageTypes.TASK_STATUS: TaskMessage,
        MessageTypes.TENSOR_DATA: TensorMessage,
        MessageTypes.CONTROL: ControlMessage,
        MessageTypes.METRIC: MetricMessage,
        MessageTypes.WORKER_REGISTER: WorkerRegistration,
    }

    target_cls = cls_map.get(msg_type, Message)
    valid_fields = {f.name for f in dataclasses.fields(target_cls)}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}
    return target_cls(**filtered_data)


class SocketManager:
    """Manages ZeroMQ sockets for inter-component communication."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        logger=None,
        context: Optional[zmq.Context] = None,
        pattern: str = "reqrep",
        is_server: bool = True,
        identity: Optional[str] = None,
    ):
        """
        Initialize socket manager with specified communication pattern.

        Args:
            host: Host address to bind/send from
            port: Port number
            logger: Logger instance
            context: Optional pre-existing ZeroMQ context
            pattern: Communication pattern - 'reqrep' (ROUTER/DEALER) or 'pubsub' (PUB/SUB)
            is_server: True to bind as server/publisher, False to act as client/subscriber
            identity: Optional unique string identity for the socket
        """
        self.host = host
        self.port = port
        self.logger = logger or get_logger(name="socket_manager")
        self.pattern = pattern
        self.is_server = is_server
        self.identity = identity
        self.connected_address: Optional[str] = None

        # Manage ZeroMQ context ownership
        self._owns_context = context is None
        self.context = context or zmq.Context()

        # Socket and identity tracking
        self._socket = None
        self._client_identities: Dict[str, bytes] = {}

        # Thread synchronization
        self._lock = Lock()
        self._running = False
        self._stop_event = Event()
        self._receive_thread: Optional[Thread] = None

        # Queues for async communication
        self._send_queue = queue.Queue()
        self._receive_queue = queue.Queue()

        # Initialize socket setup
        self._bind()

    def _bind(self):
        """Setup and configure the ZeroMQ socket."""
        if self.pattern == "reqrep":
            if self.is_server:
                # ROUTER socket on server: accepts concurrent asynchronous worker connections
                self._socket = self.context.socket(zmq.ROUTER)
                self._socket.setsockopt(zmq.LINGER, 0)
                if self.identity:
                    self._socket.setsockopt_string(zmq.IDENTITY, self.identity)
                self._socket.bind(f"tcp://{self.host}:{self.port}")
                self.logger.info(f"REQ/REP ROUTER socket bound to tcp://{self.host}:{self.port}")
            else:
                # DEALER socket on client: non-blocking asynchronous connection to ROUTER
                self._socket = self.context.socket(zmq.DEALER)
                self._socket.setsockopt(zmq.LINGER, 0)
                if self.identity:
                    self._socket.setsockopt_string(zmq.IDENTITY, self.identity)
                self.logger.info(f"REQ/REP DEALER socket initialized (identity: {self.identity})")

        elif self.pattern == "pubsub":
            if self.is_server:
                # PUB socket on server
                self._socket = self.context.socket(zmq.PUB)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.bind(f"tcp://{self.host}:{self.port}")
                self.logger.info(f"PUB/SUB socket bound to tcp://{self.host}:{self.port}")
            else:
                # SUB socket on client
                self._socket = self.context.socket(zmq.SUB)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self.logger.info("PUB/SUB subscriber socket initialized")

        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}. Use 'reqrep' or 'pubsub'")

    def connect(self, address: str):
        """
        Connect to a remote socket (client mode).

        Args:
            address: Remote address in format 'tcp://host:port' or 'host:port'
        """
        if not address.startswith("tcp://"):
            address = f"tcp://{address}"

        with self._lock:
            if self._socket is None:
                raise RuntimeError("Socket is not initialized")
            self.logger.info(f"Connecting to {address}")
            self._socket.connect(address)
            self.connected_address = address

    def send_message(self, message: Message, timeout_ms: Optional[int] = None):
        """
        Send a message to a recipient.

        Args:
            message: Message to send
            timeout_ms: Optional send timeout in milliseconds
        """
        try:
            message_dict = dataclasses.asdict(message)
            message_json = json.dumps(message_dict)
            payload_bytes = message_json.encode("utf-8")
            self.logger.debug(f"Sending message: {message.message_type} to {message.recipient_id}")

            with self._lock:
                if self._socket is None:
                    self.logger.warning("Attempted to send on closed socket")
                    return

                if timeout_ms is not None:
                    self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

                if self.pattern == "reqrep":
                    if self.is_server:
                        # Server routing to worker(s)
                        recipients = []
                        if message.recipient_id in ("all", "*"):
                            recipients = list(self._client_identities.values())
                        elif message.recipient_id:
                            rid = self._client_identities.get(
                                message.recipient_id,
                                message.recipient_id.encode("utf-8"),
                            )
                            recipients = [rid]
                        elif len(self._client_identities) == 1:
                            recipients = list(self._client_identities.values())
                        else:
                            self.logger.warning(
                                f"Cannot route message without recipient_id on ROUTER socket: {message.message_type}"
                            )
                            return

                        for rid in recipients:
                            self._socket.send_multipart([rid, b"", payload_bytes])
                    else:
                        # Client DEALER sends framed message [empty, body]
                        self._socket.send_multipart([b"", payload_bytes])

                elif self.pattern == "pubsub":
                    self._socket.send_string(message_json)

        except zmq.Again:
            self.logger.error("Send timeout - message not delivered")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    def send_message_async(self, message: Message):
        """
        Send a message asynchronously via background thread.

        Args:
            message: Message to send
        """
        self._send_queue.put(message)
        if not self._running:
            self._start_background_thread()

    def listen(self, timeout_ms: Optional[int] = 100) -> List[Message]:
        """
        Listen for incoming messages.

        Args:
            timeout_ms: Timeout in milliseconds (None or integer, defaults to 100ms)

        Returns:
            List of received messages
        """
        messages = []

        with self._lock:
            if self._socket is None:
                return []

            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)

            try:
                poll_timeout = 0 if timeout_ms is None else max(0, timeout_ms)
                sockets = dict(poller.poll(poll_timeout))

                if sockets.get(self._socket) == zmq.POLLIN:
                    # Drain all pending messages
                    while True:
                        try:
                            if self.pattern == "reqrep":
                                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                                if not frames:
                                    break

                                body_str = frames[-1].decode("utf-8")
                                message_data = json.loads(body_str)
                                message = deserialize_message(message_data)

                                if self.is_server and len(frames) >= 2:
                                    # frames[0] is client routing identity
                                    sender_routing_id = frames[0]
                                    self._client_identities[message.sender_id] = sender_routing_id

                                messages.append(message)
                                self.logger.debug(
                                    f"Received message: {message.message_type} from {message.sender_id}"
                                )

                            elif self.pattern == "pubsub":
                                message_json = self._socket.recv_string(flags=zmq.NOBLOCK)
                                message_data = json.loads(message_json)
                                message = deserialize_message(message_data)
                                messages.append(message)
                                self.logger.debug(
                                    f"Received pubsub message: {message.message_type} from {message.sender_id}"
                                )

                        except zmq.Again:
                            # Queue drained
                            break
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse JSON message: {e}")
                            break

            except Exception as e:
                self.logger.error(f"Error during socket receive: {e}")

        return messages

    def listen_async(self):
        """Listen for messages asynchronously in background thread."""
        if not self._running:
            self._start_background_thread()

    def _start_background_thread(self):
        """Start background thread for async communication."""
        self._running = True
        self._receive_thread = Thread(
            target=self._background_listener,
            daemon=True,
        )
        self._receive_thread.start()
        self.logger.info("Background listener started")

    def _background_listener(self):
        """Background listener loop for message handling."""
        while not self._stop_event.is_set():
            try:
                messages = self.listen(timeout_ms=50)

                for msg in messages:
                    self._receive_queue.put(msg)

                while not self._send_queue.empty():
                    try:
                        msg = self._send_queue.get_nowait()
                        self.send_message(msg)
                    except Exception as e:
                        self.logger.error(f"Error processing send queue: {e}")

            except Exception as e:
                self.logger.error(f"Error in background listener: {e}")
                time.sleep(0.05)

    def get_next_message(self, timeout_ms: int = 1000) -> Optional[Message]:
        """
        Get next message from receive queue with timeout.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Next message or None if timeout
        """
        try:
            msg = self._receive_queue.get(timeout=timeout_ms / 1000.0)
            return msg
        except queue.Empty:
            return None

    def stop(self):
        """Stop the socket manager and cleanly release resources."""
        self.logger.info("Socket manager stopping...")
        self._stop_event.set()
        self._running = False

        # Clean up socket
        with self._lock:
            if self._socket:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
                self._socket = None

        if self._owns_context and self.context:
            try:
                self.context.term()
            except Exception:
                pass
            self.context = None

        self.logger.info("Socket manager stopped")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except Exception:
            pass


__all__ = ["SocketManager", "deserialize_message"]
