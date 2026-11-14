"""Unit tests for network socket manager."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.network.message_types import Message, MessageTypes
from src.network.socket_manager import SocketManager


class TestSocketManager:
    """Test suite for socket manager."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.error = Mock()
        logger.warning = Mock()
        return logger

    @pytest.fixture
    def socket_manager(self, mock_logger):
        """Create a socket manager instance."""
        with patch('zmq.Context'):
            manager = SocketManager(
                host="127.0.0.1",
                port=5555,
                logger=mock_logger
            )
            yield manager
            manager.stop()

    def test_initialization(self, mock_logger):
        """Test socket manager initialization."""
        with patch('zmq.Context'):
            manager = SocketManager(
                host="127.0.0.1",
                port=5555,
                logger=mock_logger
            )
            assert manager.host == "127.0.0.1"
            assert manager.port == 5555
            manager.stop()

    @patch('zmq.Context')
    def test_send_message(self, mock_context, mock_logger):
        """Test message sending."""
        manager = SocketManager(
            host="127.0.0.1",
            port=5555,
            logger=mock_logger
        )

        message = Message(
            message_type=MessageTypes.HEARTBEAT,
            sender_id="test-worker",
            recipient_id="controller"
        )

        with patch.object(manager._socket, 'send_string'):
            manager.send_message(message)

        assert mock_logger.debug.called

        manager.stop()

    def test_listen_non_blocking(self, mock_logger):
        """Test non-blocking message listening."""
        manager = SocketManager(
            host="127.0.0.1",
            port=5555,
            logger=mock_logger
        )

        # should return empty list when no messages
        messages = manager.listen(timeout_ms=10)  # 10ms timeout
        assert messages == []

        manager.stop()

    def test_background_listener(self, mock_logger):
        """Test background listener thread."""
        manager = SocketManager(
            host="127.0.0.1",
            port=5555,
            logger=mock_logger
        )

        message_received = []
        handler = lambda msg: message_received.append(msg)

        manager.start_listener(message_handler=handler)

        # Wait for thread to start
        time.sleep(0.5)

        # Should have started successfully
        assert manager._running is True

        # Cleanup
        manager.stop()
        time.sleep(0.2)

    def test_stop_manager(self, mock_logger):
        """Test socket manager stop."""
        with patch('zmq.Context'):
            manager = SocketManager(
                host="127.0.0.1",
                port=5555,
                logger=mock_logger
            )

            assert manager._running is False
            manager.stop()
            assert manager._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])