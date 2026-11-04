from fhir.resources.R4B.bundle import Bundle


def test_fhir_export_is_a_valid_r4_bundle(client, risky_answers):
    created = client.post("/api/screenings", json=risky_answers).json()
    response = client.get(f"/api/screenings/{created['id']}/fhir")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/fhir+json")

    bundle = Bundle.model_validate(response.json())  # raises if it isn't valid FHIR R4B
    assert bundle.type == "collection"
    types = [e.resource.get_resource_type() for e in bundle.entry]
    assert types.count("RiskAssessment") == 1
    assert types.count("Observation") == 8
    assert types.count("Patient") == 1


def test_risk_assessment_carries_the_score_and_band(client, risky_answers):
    created = client.post("/api/screenings", json=risky_answers).json()
    bundle = client.get(f"/api/screenings/{created['id']}/fhir").json()
    risk = bundle["entry"][0]["resource"]
    prediction = risk["prediction"][0]
    assert prediction["probabilityDecimal"] == created["score"]
    assert prediction["qualitativeRisk"]["coding"][0]["code"] == created["band"]
    # the subject and every basis reference point at entries in the same bundle
    urls = {e["fullUrl"] for e in bundle["entry"]}
    assert {b["reference"] for b in risk["basis"]} <= urls
    assert risk["subject"]["reference"] in urls
    patient = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient")
    assert patient == {"resourceType": "Patient", "id": patient["id"], "gender": "male"}


def test_measurements_use_loinc_and_ucum(client, healthy_answers):
    created = client.post("/api/screenings", json=healthy_answers).json()
    bundle = client.get(f"/api/screenings/{created['id']}/fhir").json()
    by_code = {
        e["resource"]["code"]["coding"][0]["code"]: e["resource"]
        for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Observation" and "coding" in e["resource"]["code"]
    }
    systolic = by_code["8480-6"]["valueQuantity"]
    assert systolic == {"value": 112, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
    assert by_code["2085-9"]["valueQuantity"]["value"] == 60
    assert by_code["72166-2"]["valueCodeableConcept"]["text"] == "Not a current smoker"


def test_fhir_export_of_missing_screening_is_404(client):
    assert client.get("/api/screenings/999/fhir").status_code == 404
