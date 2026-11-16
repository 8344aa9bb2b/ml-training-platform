"""Unit tests for the dual-pattern SocketManager."""

import time
import json
import pytest

from src.network.message_types import (
    Message,
    MessageTypes,
    TaskMessage,
    WorkerRegistration,
)
from src.network.socket_manager import SocketManager, deserialize_message


@pytest.fixture
def logger():
    """Quiet logger for tests."""
    import logging
    log = logging.getLogger("test_socket_manager")
    log.addHandler(logging.NullHandler())
    return log


class TestSocketManagerReqRep:
    """REQ/REP pattern tests (ROUTER server / DEALER client)."""

    def test_bind_and_send_basic(self, logger):
        manager = SocketManager(host="127.0.0.1", port=15570, logger=logger, pattern="reqrep", is_server=True)
        msg = Message(
            message_type=MessageTypes.HEARTBEAT,
            sender_id="worker-1",
        )
        # Should not raise even when no workers are connected
        manager.send_message(msg)
        manager.stop()

    def test_invalid_pattern(self, logger):
        with pytest.raises(ValueError):
            SocketManager(host="127.0.0.1", port=15571, logger=logger, pattern="invalid")

    def test_bidirectional_client_server_communication(self, logger):
        """Test true TCP loopback communication between ROUTER server and DEALER client."""
        server_port = 15572
        server = SocketManager(
            host="127.0.0.1",
            port=server_port,
            logger=logger,
            pattern="reqrep",
            is_server=True,
        )

        client = SocketManager(
            logger=logger,
            pattern="reqrep",
            is_server=False,
            identity="worker-alpha",
        )
        client.connect(f"tcp://127.0.0.1:{server_port}")
        time.sleep(0.05)

        # 1. Client registers with server
        reg_msg = WorkerRegistration(
            message_type=MessageTypes.WORKER_REGISTER,
            sender_id="worker-alpha",
            payload={"cores": 8, "gpu_available": False},
        )
        client.send_message(reg_msg)

        # 2. Server receives registration
        received = server.listen(timeout_ms=1000)
        assert len(received) == 1
        assert received[0].message_type == MessageTypes.WORKER_REGISTER
        assert received[0].sender_id == "worker-alpha"
        assert received[0].payload["cores"] == 8

        # 3. Server sends confirmation back to worker-alpha
        ack = Message(
            message_type=MessageTypes.CONTROL,
            sender_id="controller",
            recipient_id="worker-alpha",
            payload={"status": "registered"},
        )
        server.send_message(ack)

        # 4. Client receives confirmation
        client_received = client.listen(timeout_ms=1000)
        assert len(client_received) == 1
        assert client_received[0].message_type == MessageTypes.CONTROL
        assert client_received[0].payload["status"] == "registered"

        client.stop()
        server.stop()

    def test_multi_client_targeted_routing(self, logger):
        """Test ROUTER routing to specific client and broadcasting."""
        server_port = 15573
        server = SocketManager(
            host="127.0.0.1",
            port=server_port,
            logger=logger,
            pattern="reqrep",
            is_server=True,
        )

        client1 = SocketManager(
            logger=logger,
            pattern="reqrep",
            is_server=False,
            identity="worker-1",
        )
        client1.connect(f"tcp://127.0.0.1:{server_port}")

        client2 = SocketManager(
            logger=logger,
            pattern="reqrep",
            is_server=False,
            identity="worker-2",
        )
        client2.connect(f"tcp://127.0.0.1:{server_port}")
        time.sleep(0.1)

        # Both workers ping the server so server learns their routing identities
        client1.send_message(Message(message_type=MessageTypes.HEARTBEAT, sender_id="worker-1"))
        client2.send_message(Message(message_type=MessageTypes.HEARTBEAT, sender_id="worker-2"))

        deadline = time.time() + 2.0
        pings = []
        while len(pings) < 2 and time.time() < deadline:
            pings.extend(server.listen(timeout_ms=200))
        assert len(pings) == 2

        # Server sends task targeted ONLY to worker-2
        task_msg = TaskMessage(
            message_type=MessageTypes.TASK_ASSIGNMENT,
            sender_id="controller",
            recipient_id="worker-2",
            task_id="task-42",
        )
        server.send_message(task_msg)

        # Worker 2 should receive the task
        w2_msgs = client2.listen(timeout_ms=500)
        assert len(w2_msgs) == 1
        assert getattr(w2_msgs[0], "task_id", None) == "task-42"

        # Worker 1 should receive nothing
        w1_msgs = client1.listen(timeout_ms=200)
        assert len(w1_msgs) == 0

        # Broadcast to all workers
        bcast = Message(
            message_type=MessageTypes.CONTROL,
            sender_id="controller",
            recipient_id="all",
            payload={"sync": "step_complete"},
        )
        server.send_message(bcast)

        assert len(client1.listen(timeout_ms=500)) == 1
        assert len(client2.listen(timeout_ms=500)) == 1

        client1.stop()
        client2.stop()
        server.stop()


class TestSocketManagerPubSub:
    """PUB/SUB pattern tests."""

    def test_publish_receive(self, logger):
        publisher = SocketManager(
            host="127.0.0.1", port=15574, logger=logger, pattern="pubsub"
        )

        import zmq
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect("tcp://127.0.0.1:15574")

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
            host="127.0.0.1", port=15575, logger=logger, pattern="pubsub"
        )
        manager.listen_async()
        assert manager._running is True
        manager.stop()
        assert manager._running is False


class TestMessageDeserialization:
    """Tests for typed message deserialization."""

    def test_deserialize_task_message(self):
        data = {
            "message_type": "task_assignment",
            "sender_id": "controller",
            "task_id": "task-100",
            "priority": 2,
            "extra_unknown_field": "ignored",
        }
        msg = deserialize_message(data)
        assert isinstance(msg, TaskMessage)
        assert msg.task_id == "task-100"
        assert msg.priority == 2

    def test_deserialize_worker_registration(self):
        data = {
            "message_type": "worker_register",
            "sender_id": "worker-1",
            "worker_id": "worker-1",
            "capabilities": {"cpu": 4},
        }
        msg = deserialize_message(data)
        assert isinstance(msg, WorkerRegistration)
        assert msg.sender_id == "worker-1"
        assert msg.capabilities["cpu"] == 4
