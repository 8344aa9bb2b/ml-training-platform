"""Socket manager for inter-component communication.

Supports dual-pattern communication:
- REQ/REP: Request-Reply pattern for task assignment and status updates
- PUB/SUB: Publish-Subscribe pattern for status/metrics broadcasting
"""

import zmq
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from threading import Thread, Lock, Event
import queue

from core.logger import get_logger
from src.network.message_types import Message


class SocketManager:
    """Manages ZeroMQ sockets for inter-component communication."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        logger=None,
        context: Optional[zmq.Context] = None,
        pattern: str = "reqrep"
    ):
        """
        Initialize socket manager with specified communication pattern.

        Args:
            host: Host address to bind/send from
            port: Port number
            logger: Logger instance
            context: Optional pre-existing ZeroMQ context
            pattern: Communication pattern - 'reqrep' (Request-Reply) or 'pubsub' (Publish-Subscribe)
        """
        self.host = host
        self.port = port
        self.logger = logger or get_logger(name="socket_manager")
        self.pattern = pattern

        # Use provided context or create new one
        self.context = context or zmq.Context()

        # Socket setup
        self._socket = None
        self._bind()

        # Thread synchronization
        self._lock = Lock()
        self._running = False
        self._stop_event = Event()

        # Queues for async communication
        self._send_queue = queue.Queue()
        self._receive_queue = queue.Queue()

    def _bind(self):
        """Bind socket to address based on pattern."""
        if self.pattern == "reqrep":
            # Request-Reply pattern (Controller <-> Worker)
            self._socket = self.context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 0)
            # self._socket.setsockopt(zmq.REQ_RETRIES, 3)  # Not available in pyzmq 27.1.0
            # self._socket.setsockopt(zmq.REQ_TIMEOUT, 5000)  # Not available in pyzmq 27.1.0
            self._socket.bind(f"tcp://{self.host}:{self.port}")
            self.logger.info(f"REQ/REP socket bound to tcp://{self.host}:{self.port}")

        elif self.pattern == "pubsub":
            # Publish-Subscribe pattern (Controller -> Workers)
            self._socket = self.context.socket(zmq.PUB)
            self._socket.bind(f"tcp://{self.host}:{self.port}")
            self.logger.info(f"PUB/SUB socket bound to tcp://{self.host}:{self.port}")

        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}. Use 'reqrep' or 'pubsub'")

    def connect(self, address: str):
        """
        Connect to a remote socket (REQ/REP pattern only).

        Args:
            address: Remote address in format "tcp://host:port"
        """
        if self.pattern != "reqrep":
            self.logger.warning("Connect only works with REQ/REP pattern")
            return

        # Normalize address
        if not address.startswith("tcp://"):
            address = f"tcp://{address}"

        self.logger.info(f"Connecting to {address}")
        # TODO: Implement actual connection logic

    def send_message(self, message: Message, timeout_ms: Optional[int] = None):
        """
        Send a message to a recipient.

        Args:
            message: Message to send
            timeout_ms: Optional timeout for send operation (REQ/REP only)
        """
        try:
            message_dict = asdict(message)

            # Serialize to JSON
            message_json = json.dumps(message_dict)
            self.logger.debug(f"Sending message: {message.message_type} to {message.recipient_id}")

            # Send via socket
            with self._lock:
                if self.pattern == "reqrep":
                    # REQ/REP pattern - send with timeout
                    if timeout_ms is not None:
                        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
                    self._socket.send_string(message_json)
                else:
                    # PUB/SUB pattern - fire and forget
                    self._socket.send_string(message_json)

        except zmq.Again:
            self.logger.error("Send timeout - message not delivered")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    def send_message_async(self, message: Message):
        """
        Send a message asynchronously (fire and forget).

        Args:
            message: Message to send
        """
        self._send_queue.put(message)
        if not self._running:
            self._start_background_thread()

    def listen(self, timeout_ms: Optional[int] = None) -> List[Message]:
        """
        Listen for incoming messages.

        Args:
            timeout_ms: Timeout in milliseconds (0 for non-blocking)

        Returns:
            List of received messages
        """
        messages = []

        with self._lock:
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)

            try:
                # Poll with timeout
                if timeout_ms is not None:
                    sockets = dict(poller.poll(timeout_ms))
                else:
                    sockets = dict(poller.poll())

                if sockets.get(self._socket) == zmq.POLLIN:
                    # Receive message
                    try:
                        message_json = self._socket.recv_string(flags=zmq.NOBLOCK)
                        message_data = json.loads(message_json)

                        # Reconstruct message object
                        message = Message(**message_data)
                        messages.append(message)
                        self.logger.debug(f"Received message: {message.message_type} from {message.sender_id}")

                    except zmq.Again:
                        # No message available
                        pass
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON message: {e}")

            except Exception as e:
                self.logger.error(f"Error during socket receive: {e}")

        return messages

    def listen_async(self):
        """
        Listen for messages asynchronously in background thread.

        Returns received messages via the receive queue.
        """
        if not self._running:
            self._start_background_thread()

    def _start_background_thread(self):
        """Start background thread for async communication."""
        self._running = True
        self._receive_thread = Thread(
            target=self._background_listener,
            daemon=True
        )
        self._receive_thread.start()
        self.logger.info("Background listener started")

    def _background_listener(self):
        """Background listener loop for message handling."""
        while not self._stop_event.is_set():
            try:
                # Listen for messages
                messages = self.listen(timeout_ms=100)

                # Queue received messages
                for msg in messages:
                    self._receive_queue.put(msg)

                # Process queued sends
                while not self._send_queue.empty():
                    try:
                        msg = self._send_queue.get_nowait()
                        self.send_message(msg)
                    except Exception as e:
                        self.logger.error(f"Error processing send queue: {e}")

            except Exception as e:
                self.logger.error(f"Error in background listener: {e}")
                time.sleep(0.1)

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
        """Stop the socket manager."""
        self.logger.info("Socket manager stopping...")
        self._stop_event.set()
        self._running = False

        # Clean up socket
        with self._lock:
            if self._socket:
                self._socket.close()
                self._socket = None

        self.context.term()
        self.logger.info("Socket manager stopped")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except Exception:
            pass


__all__ = ["SocketManager"]