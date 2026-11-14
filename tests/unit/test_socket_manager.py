"""Unit tests for the dual-pattern SocketManager."""

import json
import pytest

from src.network.message_types import Message, MessageTypes
from src.network.socket_manager import SocketManager


@pytest.fixture
def logger():
    """Quiet logger for tests."""
    import logging
    log = logging.getLogger("test_socket_manager")
    log.addHandler(logging.NullHandler())
    return log


class TestSocketManagerReqRep:
    """REQ/REP pattern tests."""

    def test_bind_and_send(self, logger):
        manager = SocketManager(host="127.0.0.1", port=15550, logger=logger, pattern="pubsub")
        msg = Message(
            message_type=MessageTypes.HEARTBEAT,
            sender_id="worker-1",
        )
        # Should not raise
        manager.send_message(msg)
        manager.stop()

    def test_invalid_pattern(self, logger):
        with pytest.raises(ValueError):
            SocketManager(host="127.0.0.1", port=15551, logger=logger, pattern="invalid")


class TestSocketManagerPubSub:
    """PUB/SUB pattern tests."""

    def test_publish_receive(self, logger):
        publisher = SocketManager(
            host="127.0.0.1", port=15552, logger=logger, pattern="pubsub"
        )

        # Subscriber using raw zmq to read from publisher
        import zmq
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect("tcp://127.0.0.1:15552")

        # Allow subscription to settle
        import time
        time.sleep(0.2)

        msg = Message(
            message_type=MessageTypes.METRIC,
            sender_id="worker-1",
            payload={"loss": 0.5},
        )
        publisher.send_message(msg)

        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        sockets = dict(poller.poll(2000))

        assert sub in sockets, "Subscriber did not receive published message"
        data = json.loads(sub.recv_string())
        assert data["message_type"] == "metric"
        assert data["payload"]["loss"] == 0.5

        sub.close()
        ctx.term()
        publisher.stop()

    def test_async_listener(self, logger):
        manager = SocketManager(
            host="127.0.0.1", port=15553, logger=logger, pattern="pubsub"
        )
        manager.listen_async()
        assert manager._running is True
        manager.stop()
        assert manager._running is False
