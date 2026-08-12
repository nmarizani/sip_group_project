"""
Tests the API in-process using FastAPI's TestClient — no need to start uvicorn separately.
Run after `pip install -r requirements.txt`:

    python test_client.py
"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    print("GET /health ->", r.status_code, r.json())
    assert r.status_code == 200


def test_predict_valid():
    payload = {"crop": "Rice", "season": "Rainy Season", "area": 50, "annual_rainfall": 1200}
    r = client.post("/predict", json=payload)
    print("POST /predict (valid) ->", r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_yield_per_hectare"] > 0


def test_predict_unknown_crop():
    payload = {"crop": "Dragonfruit", "season": "Rainy Season", "area": 50, "annual_rainfall": 1200}
    r = client.post("/predict", json=payload)
    print("POST /predict (unknown crop) ->", r.status_code, r.json())
    assert r.status_code == 400


def test_predict_bad_season():
    # invalid literal -> FastAPI/pydantic should reject with 422 before hitting our logic
    payload = {"crop": "Rice", "season": "Monsoon", "area": 50, "annual_rainfall": 1200}
    r = client.post("/predict", json=payload)
    print("POST /predict (invalid season) ->", r.status_code)
    assert r.status_code == 422


if __name__ == "__main__":
    test_health()
    test_predict_valid()
    test_predict_unknown_crop()
    test_predict_bad_season()
    print("\nAll tests passed.")