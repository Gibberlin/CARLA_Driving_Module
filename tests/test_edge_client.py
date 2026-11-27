"""End-to-end tests for Node 1 EdgeNavClient <-> Node 2 Edge Server + local fallback."""

import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

from navigation.edge_client import EdgeNavClient

NODE2_DIR = Path(r"C:\Users\hijbu\Desktop\Node2")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestEdgeClientAndFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _find_free_port()
        cls.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "edge_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
                "--log-level",
                "error",
            ],
            cwd=str(NODE2_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        cls.proc.terminate()
        raise RuntimeError("Node 2 edge_server failed to start in time")

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def test_live_node1_to_node2_and_offline_fallback(self):
        client = EdgeNavClient(
            base_url=f"http://127.0.0.1:{self.port}",
            step_timeout_s=0.5,
        )
        try:
            health = client.check_health()
            self.assertIsNotNone(health)
            self.assertEqual(health["status"], "ONLINE")

            # Vehicle standing at (92.0, 76.0) in Town01
            plan = client.plan_route(
                map_name="Town01",
                origin=(92.0, 76.0, 0.0),
                destination=(334.0, 252.0),
            )
            self.assertIsNotNone(plan)
            self.assertGreater(len(plan.polyline_2d), 50)

            # Live step from Node 2
            p0 = plan.polyline_2d[0]
            inst = client.step_navigation(x=p0[0], y=p0[1], yaw_deg=0.0, speed_kmh=15.0)
            self.assertFalse(inst.using_fallback)
            self.assertEqual(inst.nav_state, "FOLLOWING")
            self.assertLess(inst.latency_ms, 500.0)

            # Simulate Node 2 unreachable mid-drive -> automatic local fallback
            client.base_url = "http://127.0.0.1:1"
            if client._http is not None:
                client._http.close()
                import httpx
                client._http = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.05)

            inst_fb = client.step_navigation(x=p0[0], y=p0[1], yaw_deg=0.0, speed_kmh=15.0)
            self.assertTrue(inst_fb.using_fallback)
            self.assertEqual(inst_fb.maneuver, "FALLBACK")
            self.assertEqual(inst_fb.nav_state, "FOLLOWING")

            # Obstacle failsafe in fallback mode -> full brake
            inst_stop = client.step_navigation(
                x=p0[0], y=p0[1], yaw_deg=0.0, speed_kmh=15.0, failsafe_blocked=True
            )
            self.assertEqual(inst_stop.nav_state, "BLOCKED")
            self.assertEqual(inst_stop.brake_hint, 1.0)
            self.assertEqual(inst_stop.throttle_hint, 0.0)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
