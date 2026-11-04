"""AWS Lambda handlers: the screener as a serverless API.

    POST /screenings          score one profile, store it in DynamoDB      (201)
    GET  /screenings/{id}     fetch a stored screening                     (200/404)
    POST /batches             queue up to 100 profiles on SQS              (202)
    GET  /batches/{id}        results scored so far for a batch            (200)
    SQS  -> process_batch     worker that scores queued profiles

The model and input validation are the same code the FastAPI app uses
(app.predictor, app.schemas), so both deployments score identically.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from pydantic import ValidationError

from app.predictor import Predictor
from app.schemas import ScreeningInput

MAX_BATCH = 100
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "model.joblib"

# Created lazily and reused across invocations of a warm Lambda container.
_predictor: Predictor | None = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL)))
    return _predictor


def table():
    return boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])


def sqs():
    return boto3.client("sqs")


# ---------- helpers ----------

def _response(status: int, body: Any) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def _json_body(event: dict) -> Any:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    return json.loads(raw)


def _to_dynamo(obj: Any) -> Any:
    # DynamoDB rejects Python floats; it stores numbers as Decimal.
    return json.loads(json.dumps(obj), parse_float=Decimal)


def _from_dynamo(obj: Any) -> Any:
    def number(d: Decimal) -> int | float:
        return int(d) if d == d.to_integral_value() else float(d)

    return json.loads(json.dumps(obj, default=number))


def _errors(exc: ValidationError) -> list[dict]:
    return [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]


def _score_and_store(answers: ScreeningInput, batch_id: str | None = None) -> dict:
    result = predictor().predict(answers.to_features())
    item = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "answers": answers.model_dump(),
        "score": result.score,
        "band": result.band,
        "flagged": result.flagged,
        "top_factors": result.top_factors,
    }
    if batch_id:
        item["batch_id"] = batch_id
    table().put_item(Item=_to_dynamo(item))
    return item


# ---------- HTTP API handlers ----------

def create_screening(event: dict, context: Any) -> dict:
    try:
        answers = ScreeningInput.model_validate(_json_body(event))
    except json.JSONDecodeError:
        return _response(400, {"detail": "Body must be JSON"})
    except ValidationError as exc:
        return _response(422, {"detail": _errors(exc)})
    return _response(201, _score_and_store(answers))


def get_screening(event: dict, context: Any) -> dict:
    screening_id = (event.get("pathParameters") or {}).get("id", "")
    item = table().get_item(Key={"id": screening_id}).get("Item")
    if item is None:
        return _response(404, {"detail": "Screening not found"})
    return _response(200, _from_dynamo(item))


def submit_batch(event: dict, context: Any) -> dict:
    """Validate every profile up front, then queue them. Nothing is queued if any is invalid."""
    try:
        profiles = _json_body(event).get("screenings")
    except (json.JSONDecodeError, AttributeError):
        return _response(400, {"detail": 'Body must be JSON like {"screenings": [...]}'})
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= MAX_BATCH:
        return _response(422, {"detail": f"'screenings' must be a list of 1-{MAX_BATCH} profiles"})

    invalid = []
    for i, profile in enumerate(profiles):
        try:
            ScreeningInput.model_validate(profile)
        except ValidationError as exc:
            invalid.append({"index": i, "errors": _errors(exc)})
    if invalid:
        return _response(422, {"detail": invalid})

    batch_id = str(uuid.uuid4())
    queue_url = os.environ["QUEUE_URL"]
    client = sqs()
    for start in range(0, len(profiles), 10):  # SQS accepts at most 10 messages per call
        chunk = profiles[start : start + 10]
        sent = client.send_message_batch(
            QueueUrl=queue_url,
            Entries=[
                {"Id": str(start + j), "MessageBody": json.dumps({"batch_id": batch_id, "answers": p})}
                for j, p in enumerate(chunk)
            ],
        )
        if sent.get("Failed"):
            return _response(502, {"detail": "Could not queue every profile", "batch_id": batch_id})
    return _response(202, {"batch_id": batch_id, "queued": len(profiles)})


def get_batch(event: dict, context: Any) -> dict:
    batch_id = (event.get("pathParameters") or {}).get("id", "")
    items = table().query(
        IndexName="batch_id-index", KeyConditionExpression=Key("batch_id").eq(batch_id)
    )["Items"]
    return _response(200, {"batch_id": batch_id, "completed": len(items), "results": _from_dynamo(items)})


# ---------- SQS worker ----------

def process_batch(event: dict, context: Any) -> dict:
    """Score each queued profile. Report failures individually so SQS only retries those
    (and sends them to the dead-letter queue after repeated failures)."""
    failures = []
    for record in event.get("Records", []):
        try:
            message = json.loads(record["body"])
            _score_and_store(ScreeningInput.model_validate(message["answers"]), batch_id=message["batch_id"])
        except Exception:  # noqa: BLE001 - any failure means "retry this message"
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}
