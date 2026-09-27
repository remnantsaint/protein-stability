"""接口与生命周期测试：用可控的模拟模型，不下载或加载大型权重。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event
import unittest

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app

P00044 = (
    "MTEFKAGSAKKGATLFKTRCLQCHTVEKGGPHKVGPNLHGIFGRHSGQAEGYSYTDANIK"
    "KNVLWDENNMSEYLTNPKKYIPGTKMAFGGLKKEKDRNDLITYLKKACE"
)
VALID = {"sequence": P00044, "mutations": "F88Y_L91A"}


class FakeService:
    device = "test"

    def __init__(self):
        self.calls = []
        self.closed = False
        self.fail = False
        self.nonfinite = False

    def predict(self, sequence, mutations):
        self.calls.append((sequence, mutations))
        if self.fail:
            raise RuntimeError("internal-details-should-not-leak")
        return (float("nan") if self.nonfinite else -2.5934), mutations

    def close(self):
        self.closed = True


class WebTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.app = create_app(lambda: self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_pages_and_static_image(self):
        for path in ("/", "/ddg-predictor/"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Mutation set", response.text)
            self.assertIn("/ddg-predictor/static/figure.jpg", response.text)
            self.assertNotIn("{% raw %}", response.text)
            self.assertIn("{{ mutation }}", response.text)
        self.assertEqual(self.client.get("/ddg-predictor/static/figure.jpg").status_code, 200)
        self.assertEqual(self.client.get("/ddg-predictor", follow_redirects=False).status_code, 307)

    def test_both_prediction_routes(self):
        for path in ("/predict", "/ddg-predictor/predict"):
            result = self.client.post(path, json=VALID)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), {"ddg": -2.5934, "length": len(P00044),
                                             "mutation": "F88Y_L91A", "mutation_count": 2})

    def test_normalization(self):
        response = self.client.post("/predict", json={"sequence": " \n" + P00044.lower() + " ",
                                                     "mutations": " f88y_l91a "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.calls[-1], (P00044, "F88Y_L91A"))

    def test_legacy_single_point(self):
        for pos in (88, "88"):
            response = self.client.post("/predict", json={"sequence": P00044, "pos": pos, "wt": "f", "mt": "y"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mutation"], "F88Y")
            self.assertEqual(response.json()["mutation_count"], 1)

    def test_new_mutations_take_priority(self):
        response = self.client.post("/predict", json={**VALID, "pos": 999, "wt": "?", "mt": None})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mutation"], VALID["mutations"])

    def test_bad_inputs_never_reach_model(self):
        invalid = [[], [1], True, 42, None, {},
                   {"sequence": 42, "mutations": "F88Y"},
                   {"sequence": "  ", "mutations": "F88Y"},
                   {"sequence": "AXZ", "mutations": "A1C"},
                   {"sequence": P00044, "mutations": ["F88Y"]},
                   {"sequence": P00044, "mutations": "F88Y_"},
                   {"sequence": P00044, "mutations": "F88Y_F88A"},
                   {"sequence": P00044, "mutations": "F88F"},
                   {"sequence": P00044, "mutations": "F999Y"},
                   {"sequence": P00044, "mutations": "A88Y"},
                   {"sequence": P00044, "mutations": "_".join(f"A{i}C" for i in range(1, 12))},
                   {"sequence": P00044, "pos": 88, "wt": "F"},
                   {"sequence": P00044, "pos": True, "wt": "F", "mt": "Y"}]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json())
        self.assertEqual(self.service.calls, [])

    def test_malformed_json(self):
        response = self.client.post("/predict", content="{bad", headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_health_and_documentation(self):
        self.assertEqual(self.client.get("/healthz").json(),
                         {"status": "ok", "model_loaded": True, "device": "test"})
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/predict", schema["paths"])
        self.assertIn("PredictionRequest", schema["components"]["schemas"])
        self.assertIn("PredictionResponse", schema["components"]["schemas"])

    def test_internal_error_is_hidden_and_lock_released(self):
        self.service.fail = True
        with self.assertLogs("ddg", level="ERROR"):
            response = self.client.post("/predict", json=VALID)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal-details", response.text)
        self.service.fail = False
        self.assertEqual(self.client.post("/predict", json=VALID).status_code, 200)

    def test_nonfinite_model_result_is_rejected(self):
        self.service.nonfinite = True
        with self.assertLogs("ddg", level="ERROR"):
            response = self.client.post("/predict", json=VALID)
        self.assertEqual(response.status_code, 500)

    def test_concurrent_request_returns_busy_while_health_works(self):
        entered, release = Event(), Event()
        def slow_predict(sequence, mutations):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
            return -2.5934, mutations
        self.service.predict = slow_predict
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.client.post, "/predict", json=VALID)
            try:
                self.assertTrue(entered.wait(5))
                response = self.client.post("/predict", json=VALID)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers["Retry-After"], "2")
                self.assertEqual(self.client.get("/healthz").status_code, 200)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5).status_code, 200)


class LifecycleTests(unittest.TestCase):
    def test_load_once_and_close_on_shutdown(self):
        service = FakeService()
        created = []
        def factory():
            created.append(True)
            return service
        app = create_app(factory)
        self.assertEqual(created, [])
        with TestClient(app) as client:
            self.assertEqual(len(created), 1)
            client.post("/predict", json=VALID)
            client.post("/predict", json=VALID)
            self.assertEqual(len(created), 1)
            self.assertFalse(service.closed)
        self.assertTrue(service.closed)

    def test_failed_model_load_prevents_startup(self):
        def broken_factory():
            raise RuntimeError("missing model")
        with self.assertRaisesRegex(RuntimeError, "missing model"):
            with TestClient(create_app(broken_factory)):
                self.fail("startup must not complete")


if __name__ == "__main__":
    unittest.main()
