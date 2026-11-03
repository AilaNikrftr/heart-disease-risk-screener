def test_health_reports_model_and_metrics(client):
    body = client.get("/api/health").json()
    assert body["model_loaded"] is True
    assert "logistic_regression" in body["metrics"]


def test_health_without_model(client_without_model):
    assert client_without_model.get("/api/health").json()["model_loaded"] is False


def test_screening_without_model_is_503(client_without_model, healthy_answers):
    response = client_without_model.post("/api/screenings", json=healthy_answers)
    assert response.status_code == 503
    assert "ml.train" in response.json()["detail"]


def test_risky_profile_scores_higher_than_healthy(client, healthy_answers, risky_answers):
    low = client.post("/api/screenings", json=healthy_answers)
    high = client.post("/api/screenings", json=risky_answers)
    assert low.status_code == high.status_code == 201
    low, high = low.json(), high.json()
    assert high["score"] > low["score"]
    assert high["band"] == "high" and high["flagged"] is True
    assert low["band"] == "low" and low["flagged"] is False
    assert "not a diagnosis" in high["disclaimer"]


def test_top_factors_explain_a_risky_score(client, risky_answers):
    factors = client.post("/api/screenings", json=risky_answers).json()["top_factors"]
    assert 1 <= len(factors) <= 3
    assert "age" in {f["feature"] for f in factors}
    assert all(f["contribution"] > 0 for f in factors)


def test_screenings_are_stored_newest_first(client, healthy_answers, risky_answers):
    first = client.post("/api/screenings", json=healthy_answers).json()
    second = client.post("/api/screenings", json=risky_answers).json()
    listed = client.get("/api/screenings").json()
    assert [s["id"] for s in listed] == [second["id"], first["id"]]
    assert listed[1]["answers"] == healthy_answers


def test_list_limit(client, healthy_answers):
    for _ in range(3):
        client.post("/api/screenings", json=healthy_answers)
    assert len(client.get("/api/screenings?limit=2").json()) == 2
    assert client.get("/api/screenings?limit=0").status_code == 422


def test_delete(client, healthy_answers):
    created = client.post("/api/screenings", json=healthy_answers).json()
    assert client.delete(f"/api/screenings/{created['id']}").status_code == 204
    assert client.get("/api/screenings").json() == []
    assert client.delete(f"/api/screenings/{created['id']}").status_code == 404


def test_out_of_range_values_are_rejected(client, healthy_answers):
    for field, value in [("age", 12), ("bmi", 250), ("systolic_bp", 20), ("sex", "other")]:
        response = client.post("/api/screenings", json={**healthy_answers, field: value})
        assert response.status_code == 422, field


def test_inconsistent_values_are_rejected(client, healthy_answers):
    bad_bp = {**healthy_answers, "systolic_bp": 90, "diastolic_bp": 95}
    bad_chol = {**healthy_answers, "total_cholesterol": 150, "hdl_cholesterol": 160}
    assert client.post("/api/screenings", json=bad_bp).status_code == 422
    assert client.post("/api/screenings", json=bad_chol).status_code == 422


def test_unknown_fields_are_rejected(client, healthy_answers):
    response = client.post("/api/screenings", json={**healthy_answers, "weight": 70})
    assert response.status_code == 422
