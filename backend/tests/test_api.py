"""End-to-end API tests: upload -> select -> start -> poll -> download."""
import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_image


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "done_with_errors", "canceled"):
            return j
        time.sleep(0.3)
    raise TimeoutError(j)


def files(*items):
    return [("files", (name, data, "application/octet-stream")) for name, data in items]


def test_health(client):
    h = client.get("/api/health").json()
    assert set(h["tiers"]) == {"standard", "strong"}
    for t in h["tiers"].values():  # the models behind both tiers must be present
        assert h["models"][t["model"]]["available"]


def test_size_endpoint(client):
    r = client.get("/api/size", params={"width": 1001, "ratio_w": 10, "ratio_h": 3}).json()
    assert r["height"] == 300 and not r["exact_ratio"] and r["suggested_widths"] == [1000, 1010]
    assert client.get("/api/size", params={"width": 5, "ratio_w": 1, "ratio_h": 1}).status_code == 422


def test_presets_crud(client):
    presets = client.get("/api/presets").json()
    assert any(p["builtin"] for p in presets)
    r = client.post("/api/presets", json={"name": "测试模板", "width": 1080, "ratio_w": 10, "ratio_h": 13})
    assert r.status_code == 201
    p = r.json()
    assert (p["out_width"], p["out_height"]) == (1080, 1404)
    assert client.post("/api/presets", json={"name": "测试模板", "width": 100, "ratio_w": 1, "ratio_h": 1}).status_code == 409
    assert client.put(f"/api/presets/{p['id']}", json={"name": "测试模板2", "width": 900, "ratio_w": 1, "ratio_h": 1}).json()["out_height"] == 900
    builtin = next(x for x in presets if x["builtin"])
    assert client.delete(f"/api/presets/{builtin['id']}").status_code == 403
    assert client.delete(f"/api/presets/{p['id']}").status_code == 204


def test_full_flow_with_selection_invalid_files_and_zip(client):
    r = client.post("/api/jobs", files=files(
        ("产品图A.jpg", make_image(120, 90, seed=1)),
        ("产品图A.png", make_image(80, 80, fmt="PNG", seed=2)),
        ("skip_me.webp", make_image(60, 40, fmt="WEBP", seed=3)),
        ("broken.jpg", make_image()[:200]),
        ("readme.txt", b"hello"),
    ), data={"name": "测试批次"})
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "draft"
    by_name = {it["original_name"]: it for it in job["items"]}
    assert by_name["broken.jpg"]["status"] == "invalid" and by_name["broken.jpg"]["error"]
    assert by_name["readme.txt"]["status"] == "invalid"
    assert client.get(by_name["产品图A.jpg"]["urls"]["thumb"]).status_code == 200

    keep = [by_name["产品图A.jpg"]["id"], by_name["产品图A.png"]["id"]]
    settings = {"tier": "standard", "width": 400, "ratio_w": 10, "ratio_h": 13, "mode": "pad", "format": "jpg",
                "quality": 90, "name_suffix": "_hd"}
    r = client.post(f"/api/jobs/{job['id']}/start", json={"settings": settings, "item_ids": keep})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/jobs/{job['id']}/start", json={"settings": settings}).status_code == 409

    job = wait_done(client, job["id"])
    assert job["status"] == "done"
    assert job["counts"]["done"] == 2 and job["counts"]["skipped"] == 1 and job["counts"]["invalid"] == 2
    names = sorted(it["output_name"] for it in job["items"] if it["status"] == "done")
    assert names == ["产品图A_hd (2).jpg", "产品图A_hd.jpg"]  # same stem -> de-duplicated

    done = [it for it in job["items"] if it["status"] == "done"]
    for it in done:
        img = Image.open(io.BytesIO(client.get(it["urls"]["result"]).content))
        assert img.size == (400, 520)
        assert it["result"]["qa"]["delta_e_mean"] < 3
        assert client.get(it["urls"]["baseline"]).status_code == 200
        dl = client.get(it["urls"]["download"])
        assert "attachment" in dl.headers["content-disposition"]

    z = client.get(f"/api/jobs/{job['id']}/download")
    assert z.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(z.content))
    names_in_zip = zf.namelist()
    assert "产品图A_hd.jpg" in names_in_zip and "处理报告.csv" in names_in_zip
    assert all(zi.flag_bits & 0x800 for zi in zf.infolist())  # UTF-8 names: no mojibake on Windows/macOS
    report = zf.read("处理报告.csv").decode("utf-8-sig")
    assert "broken.jpg" in report and "无法读取" in report and "成功" in report

    hist = client.get("/api/jobs").json()
    assert any(h["id"] == job["id"] for h in hist)


def test_one_shot_api_call_and_retry(client, monkeypatch):
    import app.main as main

    calls = {"n": 0}
    real = main.worker.engine.upscale

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated GPU failure")
        return real(*a, **kw)

    monkeypatch.setattr(main.worker.engine, "upscale", flaky)
    r = client.post("/api/jobs", files=files(("a.jpg", make_image(seed=5))),
                    data={"start": "true", "settings": json.dumps({"width": 300, "ratio_w": 1, "ratio_h": 1, "format": "webp"})})
    job = wait_done(client, r.json()["id"])
    assert job["status"] == "done_with_errors"
    item = job["items"][0]
    assert item["status"] == "failed" and "simulated" in item["error"]
    assert client.post(f"/api/items/{item['id']}/retry").status_code == 200
    job = wait_done(client, job["id"])
    assert job["status"] == "done" and job["items"][0]["attempts"] == 2


def test_bad_settings_rejected(client):
    r = client.post("/api/jobs", files=files(("a.jpg", make_image())),
                    data={"start": "true", "settings": json.dumps({"width": 300, "ratio_w": 1, "ratio_h": 100})})
    assert r.status_code == 422 and "超出范围" in r.json()["detail"]


def test_delete_job(client):
    job = client.post("/api/jobs", files=files(("a.jpg", make_image()))).json()
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
