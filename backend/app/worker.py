"""Background worker: processes queued items one at a time on the GPU.

State lives in SQLite so a crash or restart never loses a batch: items that were
'processing' when the process died are put back in the queue at startup.
Live per-item progress (stage + fraction) is kept in memory for polling.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import config, db
from .imaging import pipeline
from .imaging.io import ImageError
from .imaging.sizing import SizeError
from .imaging.sr import SREngine, get_engine

log = logging.getLogger("aiclear.worker")

OVERHEAD_S = 0.4


def job_dir(job_id: str) -> Path:
    return config.DATA_DIR / "jobs" / job_id


class Worker:
    def __init__(self, engine: SREngine | None = None):
        self.engine = engine or get_engine()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.live: dict[str, dict] = {}  # item_id -> {"stage", "progress"}
        # seconds per SR-output megapixel, per tier; starts from the backend's default and is
        # refined online (EMA). Stored per backend so a CPU fallback doesn't skew GPU estimates.
        self._rates_key = f"rates:{self.engine.backend}:{self.engine.device_info()['device']}"
        default = self.engine.default_rate
        self.rates: dict[str, float] = {"standard": default, "strong": default, **db.kv_get(self._rates_key, {})}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        n = db.ex("UPDATE items SET status='queued', started_at=NULL WHERE status='processing'")
        if n:
            log.info("resumed %d interrupted item(s)", n)
        self._thread = threading.Thread(target=self._loop, name="sr-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def notify(self) -> None:
        self._wake.set()

    # -- estimates ---------------------------------------------------------
    def estimate(self, width: int, height: int, settings: pipeline.ProcessSettings) -> float:
        mp = pipeline.sr_workload(width, height, settings, self.engine)
        return round(OVERHEAD_S + mp * self.rates.get(settings.tier, 0.5), 2)

    def _learn(self, tier: str, mp: float, seconds: float) -> None:
        if mp < 0.05:
            return
        rate = max(0.0, seconds - OVERHEAD_S) / mp
        old = self.rates.get(tier, rate)
        self.rates[tier] = round(0.7 * old + 0.3 * rate, 4)
        db.kv_set(self._rates_key, self.rates)

    # -- main loop ---------------------------------------------------------
    def _claim(self) -> dict | None:
        with db.tx() as c:
            r = c.execute("""SELECT items.* FROM items JOIN jobs ON jobs.id = items.job_id
                             WHERE items.status='queued' ORDER BY items.queued_at, items.idx LIMIT 1""").fetchone()
            if r is None:
                return None
            c.execute("UPDATE items SET status='processing', started_at=?, attempts=attempts+1 WHERE id=?",
                      (time.time(), r["id"]))
            return dict(r)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._claim()
            except Exception:
                log.exception("claim failed")
                item = None
            if item is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self._process(item)

    def _process(self, item: dict) -> None:
        iid = item["id"]
        self.live[iid] = {"stage": "排队完成，开始处理", "progress": 0.0}
        job = db.row_to_dict(db.q1("SELECT * FROM jobs WHERE id=?", (item["job_id"],)))
        try:
            if job is None:
                return
            settings = pipeline.ProcessSettings(**job["settings"])
            data = (job_dir(job["id"]) / "in" / item["input_file"]).read_bytes()

            def progress(stage: str, frac: float) -> None:
                self.live[iid] = {"stage": stage, "progress": round(frac, 3)}

            t0 = time.perf_counter()
            res = pipeline.process(data, settings, self.engine, progress)
            elapsed = time.perf_counter() - t0
            out_dir = job_dir(job["id"]) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = f"{iid}.{res.ext}"
            (out_dir / out_file).write_bytes(res.data)
            (out_dir / f"{iid}.baseline.webp").write_bytes(res.baseline_preview)
            sr = res.report["model"]["sr_output"]
            self._learn(settings.tier, sr[0] * sr[1] / 1e6, elapsed)
            # job may have been deleted while we were working
            db.ex("""UPDATE items SET status='done', error=NULL, output_file=?, report=?, finished_at=?
                     WHERE id=? AND status='processing'""",
                  (out_file, json.dumps(res.report, ensure_ascii=False), time.time(), iid))
        except (ImageError, SizeError, ValueError) as e:
            self._fail(iid, str(e))
        except Exception as e:  # keep the worker alive whatever happens
            log.error("item %s failed:\n%s", iid, traceback.format_exc())
            msg = "显存不足，请缩小输出尺寸后重试" if "out of memory" in str(e).lower() else f"处理出错：{type(e).__name__}: {e}"
            self._fail(iid, msg)
        finally:
            self.live.pop(iid, None)
            # job deleted while we were working: drop anything we just wrote
            if db.q1("SELECT 1 FROM jobs WHERE id=?", (item["job_id"],)) is None:
                shutil.rmtree(job_dir(item["job_id"]), ignore_errors=True)

    @staticmethod
    def _fail(iid: str, msg: str) -> None:
        db.ex("UPDATE items SET status='failed', error=?, finished_at=? WHERE id=? AND status='processing'",
              (msg[:500], time.time(), iid))
