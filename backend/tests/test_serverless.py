"""Lambda handlers against mocked AWS (moto): DynamoDB table, SQS queue and the
SQS -> worker hand-off, with the same events API Gateway and SQS send."""

import json

import boto3
import pytest
from moto import mock_aws

from serverless import handlers


@pytest.fixture
def aws(monkeypatch, trained_model):
    model_path, _ = trained_model
    for key, value in {
        "AWS_DEFAULT_REGION": "ca-central-1",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "TABLE_NAME": "screenings",
        "MODEL_PATH": str(model_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(handlers, "_predictor", None)

    with mock_aws():
        boto3.client("dynamodb").create_table(
            TableName="screenings",
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "batch_id", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "batch_id-index",
                    "KeySchema": [{"AttributeName": "batch_id", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        )
        queue_url = boto3.client("sqs").create_queue(QueueName="batch")["QueueUrl"]
        monkeypatch.setenv("QUEUE_URL", queue_url)
        yield queue_url


def http(body=None, path_id=None):
    event = {"body": json.dumps(body) if body is not None else None, "isBase64Encoded": False}
    if path_id is not None:
        event["pathParameters"] = {"id": path_id}
    return event


def body(response):
    return json.loads(response["body"])


def drain(queue_url):
    """Receive everything on the queue and shape it like the event SQS sends Lambda."""
    records = []
    sqs = boto3.client("sqs")
    while True:
        msgs = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10).get("Messages", [])
        if not msgs:
            return {"Records": records}
        records += [{"messageId": m["MessageId"], "body": m["Body"]} for m in msgs]
        for m in msgs:
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=m["ReceiptHandle"])


def test_create_then_get_round_trips_through_dynamodb(aws, risky_answers):
    created = handlers.create_screening(http(risky_answers), None)
    assert created["statusCode"] == 201
    item = body(created)
    assert item["band"] == "high" and item["flagged"] is True

    fetched = handlers.get_screening(http(path_id=item["id"]), None)
    assert fetched["statusCode"] == 200
    stored = body(fetched)
    assert stored["score"] == item["score"]
    assert stored["answers"] == risky_answers  # Decimals converted back to plain numbers


def test_create_rejects_invalid_input(aws, healthy_answers):
    response = handlers.create_screening(http({**healthy_answers, "bmi": 250}), None)
    assert response["statusCode"] == 422
    assert body(response)["detail"][0]["loc"] == ["bmi"]
    assert handlers.create_screening({"body": "not json"}, None)["statusCode"] == 400


def test_get_missing_screening_is_404(aws):
    assert handlers.get_screening(http(path_id="nope"), None)["statusCode"] == 404


def test_batch_is_queued_scored_by_the_worker_and_queryable(aws, healthy_answers, risky_answers):
    profiles = [healthy_answers, risky_answers] * 6  # 12 profiles -> two SQS send batches
    submitted = handlers.submit_batch(http({"screenings": profiles}), None)
    assert submitted["statusCode"] == 202
    batch_id = body(submitted)["batch_id"]
    assert body(submitted)["queued"] == 12

    result = handlers.process_batch(drain(aws), None)
    assert result == {"batchItemFailures": []}

    batch = body(handlers.get_batch(http(path_id=batch_id), None))
    assert batch["completed"] == 12
    assert {r["band"] for r in batch["results"]} == {"low", "high"}


def test_batch_rejects_invalid_profiles_without_queueing_any(aws, healthy_answers):
    bad = {**healthy_answers, "systolic_bp": 80, "diastolic_bp": 90}
    response = handlers.submit_batch(http({"screenings": [healthy_answers, bad]}), None)
    assert response["statusCode"] == 422
    assert body(response)["detail"][0]["index"] == 1
    assert drain(aws)["Records"] == []


@pytest.mark.parametrize("payload", [{"screenings": []}, {"screenings": "x"}, {"wrong": 1}])
def test_batch_shape_is_checked(aws, payload):
    assert handlers.submit_batch(http(payload), None)["statusCode"] == 422


def test_worker_reports_only_the_failed_message(aws, healthy_answers):
    event = {
        "Records": [
            {"messageId": "ok", "body": json.dumps({"batch_id": "b1", "answers": healthy_answers})},
            {"messageId": "broken", "body": "{not json"},
            {"messageId": "invalid", "body": json.dumps({"batch_id": "b1", "answers": {"age": 5}})},
        ]
    }
    result = handlers.process_batch(event, None)
    assert result == {"batchItemFailures": [{"itemIdentifier": "broken"}, {"itemIdentifier": "invalid"}]}
    assert body(handlers.get_batch(http(path_id="b1"), None))["completed"] == 1
