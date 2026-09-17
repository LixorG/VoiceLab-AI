from fastapi import APIRouter
from pydantic import BaseModel

from app.core.errors import AppError, ErrorCode, classify_exception


def _add_routes(app):
    router = APIRouter()

    @router.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    @router.get("/oom")
    def oom():
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    @router.get("/app-error")
    def app_error():
        raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=404, details={"modelo": "x"})

    class Body(BaseModel):
        n: int

    @router.post("/validate")
    def validate(body: Body):
        return body

    app.include_router(router)


def _assert_error_shape(body):
    assert set(body) == {"error_code", "message", "details"}


def test_not_found_uses_error_format(client):
    res = client.get("/api/does-not-exist")
    assert res.status_code == 404
    _assert_error_shape(res.json())
    assert res.json()["error_code"] == "NOT_FOUND"


def test_unhandled_exception_hides_traceback(app, client):
    _add_routes(app)
    res = client.get("/boom")
    assert res.status_code == 500
    body = res.json()
    _assert_error_shape(body)
    assert body["error_code"] == "INTERNAL_ERROR"
    assert "secret" not in res.text and "Traceback" not in res.text


def test_oom_maps_to_gpu_memory_error(app, client):
    _add_routes(app)
    body = client.get("/oom").json()
    assert body["error_code"] == "GPU_MEMORY_ERROR"
    assert "memoria de GPU" in body["message"]


def test_app_error_keeps_status_and_details(app, client):
    _add_routes(app)
    res = client.get("/app-error")
    assert res.status_code == 404
    assert res.json()["details"] == {"modelo": "x"}


def test_validation_error_format(app, client):
    _add_routes(app)
    res = client.post("/validate", json={"n": "no-es-numero"})
    assert res.status_code == 422
    body = res.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["details"][0]["campo"].endswith("n")


def test_classify_exception():
    assert classify_exception(MemoryError("CUDA out of memory")) is ErrorCode.GPU_MEMORY_ERROR
    assert classify_exception(ValueError("x")) is ErrorCode.INTERNAL_ERROR
