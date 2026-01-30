"""ROS2 server for robot integration."""

import threading
from typing import Any, Optional

from ..server.base_server import AbstractServer


class ROS2Server(AbstractServer):
    """ROS2 node for VLA inference in robot systems."""

    def __init__(
        self,
        node_name: str = "vla_inference",
        host: str = "localhost",
        port: int = 0,  # Not used for ROS2
        model: Optional[Any] = None,
        action_topic: str = "/vla/action",
        observation_topic: str = "/vla/observation",
        service_name: str = "/vla/predict",
    ):
        """
        Args:
            node_name: ROS2 node name
            host: Not used
            port: Not used
            model: VLA model or policy
            action_topic: Topic to publish actions
            observation_topic: Topic to subscribe for observations
            service_name: Service name for inference
        """
        super().__init__(host=host, port=port)
        self.node_name = node_name
        self.model = model
        self.action_topic = action_topic
        self.observation_topic = observation_topic
        self.service_name = service_name

        self._node = None
        self._action_publisher = None
        self._observation_subscriber = None
        self._service = None
        self._spin_thread = None

    async def start(self) -> None:
        """Start ROS2 node."""
        try:
            import rclpy
            from rclpy.node import Node

            rclpy.init()
            self._node = Node(self.node_name)

            # Setup publishers, subscribers, services
            self._setup_ros2()

            self._is_running = True

            # Spin in background thread
            self._spin_thread = threading.Thread(
                target=rclpy.spin,
                args=(self._node,)
            )
            self._spin_thread.start()

        except ImportError:
            raise ImportError("rclpy required for ROS2 server")

    def _setup_ros2(self) -> None:
        """Setup ROS2 publishers, subscribers, and services."""
        try:
            from std_msgs.msg import Float32MultiArray

            # Action publisher
            self._action_publisher = self._node.create_publisher(
                Float32MultiArray,
                self.action_topic,
                10
            )

            # Observation subscriber
            self._observation_subscriber = self._node.create_subscription(
                Float32MultiArray,
                self.observation_topic,
                self._observation_callback,
                10
            )

        except ImportError:
            pass

    def _observation_callback(self, msg) -> None:
        """Handle incoming observation."""
        import numpy as np

        if self.model is None:
            return

        # Process observation and publish action
        observation = {"proprioception": np.array(msg.data)}

        if hasattr(self.model, 'get_action'):
            action = self.model.get_action(observation, "")
        else:
            action = np.zeros(7)

        # Publish action
        self._publish_action(action)

    def _publish_action(self, action) -> None:
        """Publish action to ROS2 topic."""
        try:
            from std_msgs.msg import Float32MultiArray

            msg = Float32MultiArray()
            msg.data = action.tolist()
            self._action_publisher.publish(msg)

        except ImportError:
            pass

    async def stop(self) -> None:
        """Stop ROS2 node."""
        try:
            import rclpy

            if self._node:
                self._node.destroy_node()
            rclpy.shutdown()

            if self._spin_thread:
                self._spin_thread.join()

        except ImportError:
            pass

        self._is_running = False

    async def predict(
        self,
        request: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Handle prediction request."""
        import numpy as np

        if self.model is None:
            return {"actions": []}

        observation = {
            "images": np.array(request.get("images", [])),
            "proprioception": np.array(request.get("proprioception", [])),
        }
        instruction = request.get("instruction", "")

        if hasattr(self.model, 'get_action'):
            action = self.model.get_action(observation, instruction)
        else:
            action = np.zeros(7)

        # Also publish to ROS2 topic
        self._publish_action(action)

        return {"actions": action.tolist()}

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load model."""
        try:
            import torch
            self.model = torch.load(model_path)
        except ImportError:
            pass

    def get_server_info(self) -> dict[str, Any]:
        return {
            "type": "ROS2",
            "node_name": self.node_name,
            "action_topic": self.action_topic,
            "observation_topic": self.observation_topic,
            "is_running": self._is_running,
        }


__all__ = ['ROS2Server']
