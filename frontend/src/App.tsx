import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, apiKey, type Health, type Job, type Preset, type Settings } from "./api";
import { uploadFiles } from "./upload";
import DropZone from "./components/DropZone";
import ItemGrid from "./components/ItemGrid";
import SpecPanel from "./components/SpecPanel";
import RunPanel from "./components/RunPanel";
import CompareViewer from "./components/CompareViewer";
import HistoryDrawer from "./components/HistoryDrawer";

const SETTINGS_KEY = "aiclear.settings";

function loadSettings(defaults: Settings): Settings {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null");
    return saved ? { ...defaults, ...saved } : defaults;
  } catch {
    return defaults;
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [upload, setUpload] = useState<number | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [toast, setToast] = useState<{ text: string; bad?: boolean } | null>(null);
  const [needKey, setNeedKey] = useState(false);
  const [starting, setStarting] = useState(false);
  const toastTimer = useRef<number>(undefined);

  const notify = useCallback((text: string, bad = false) => {
    setToast({ text, bad });
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), bad ? 6000 : 3000);
  }, []);

  const fail = useCallback(
    (e: unknown) => {
      if (e instanceof ApiError && e.status === 401) setNeedKey(true);
      notify(e instanceof Error ? e.message : String(e), true);
    },
    [notify],
  );

  const boot = useCallback(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setSettings((s) => s ?? loadSettings(h.defaults));
        return api.presets().then(setPresets);
      })
      .catch(fail);
  }, [fail]);
  useEffect(boot, [boot]);

  useEffect(() => {
    if (settings) localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  // Poll while the job is being processed
  const active = job && job.state === "started" && (job.status === "queued" || job.status === "running");
  useEffect(() => {
    if (!job || !active) return;
    const t = window.setInterval(() => {
      api.job(job.id).then(setJob).catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(t);
  }, [job?.id, active]);

  const onFiles = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      const draftId = job && job.state === "draft" ? job.id : null;
      const before = new Set((draftId && job?.items?.map((i) => i.id)) || []);
      setUpload(0);
      try {
        const j = await uploadFiles(draftId, files, setUpload);
        const fresh = await api.job(j.id);
        setJob(fresh);
        const added = fresh.items!.filter((i) => !before.has(i.id));
        setSelected((prev) => {
          const next = new Set(draftId ? prev : []);
          added.filter((i) => i.status === "uploaded").forEach((i) => next.add(i.id));
          return next;
        });
        const bad = added.filter((i) => i.status === "invalid").length;
        if (bad) notify(`${bad} 个文件无法处理，已标红并自动跳过`, true);
      } catch (e) {
        fail(e);
      } finally {
        setUpload(null);
      }
    },
    [job, notify, fail],
  );

  const onStart = async () => {
    if (!job || !settings) return;
    setStarting(true);
    try {
      setJob(await api.start(job.id, settings, [...selected]));
    } catch (e) {
      fail(e);
    } finally {
      setStarting(false);
    }
  };

  const refresh = useCallback(() => {
    if (job) api.job(job.id).then(setJob).catch(fail);
  }, [job, fail]);

  const openJob = async (id: string) => {
    try {
      const j = await api.job(id);
      setJob(j);
      setSelected(new Set(j.items!.filter((i) => i.status === "uploaded").map((i) => i.id)));
      if (j.settings) setSettings((s) => ({ ...s!, ...j.settings! }));
      setHistoryOpen(false);
    } catch (e) {
      fail(e);
    }
  };

  const newJob = () => {
    setJob(null);
    setSelected(new Set());
  };

  const items = job?.items ?? [];
  const firstSelected = useMemo(() => items.find((i) => selected.has(i.id)) ?? items.find((i) => i.status !== "invalid"), [items, selected]);
  const doneItems = items.filter((i) => i.status === "done");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" className="brand-mark" aria-hidden>
            <path d="M4 11V4h7M21 4h7v7M28 21v7h-7M11 28H4v-7" />
          </svg>
          <span>图片高清化与尺寸标准化</span>
        </div>
        <div className="topbar-right">
          {health?.device && (
            <span className={`device ${health.device.device === "cuda" ? "" : "device-cpu"}`} title="处理所用的设备">
              {health.device.device === "cuda" ? `GPU ${health.device.name.replace("NVIDIA GeForce ", "")}` : health.device.name}
            </span>
          )}
          <button className="btn ghost" onClick={newJob} disabled={!job}>
            新建任务
          </button>
          <button className="btn ghost" onClick={() => setHistoryOpen(true)}>
            历史任务
          </button>
        </div>
      </header>

      <main className="workbench">
        <section className="work-area">
          {items.length === 0 ? (
            <DropZone onFiles={onFiles} uploading={upload} limits={health?.limits} big />
          ) : (
            <ItemGrid
              job={job!}
              selected={selected}
              setSelected={setSelected}
              onFiles={onFiles}
              uploading={upload}
              onOpen={setCompareId}
              onRetry={(id) => api.retryItem(id).then(refresh).catch(fail)}
            />
          )}
        </section>

        <aside className="side">
          {settings && health && (!job || job.state === "draft") && (
            <SpecPanel
              health={health}
              presets={presets}
              setPresets={setPresets}
              settings={settings}
              setSettings={setSettings}
              preview={firstSelected}
              count={selected.size}
              canStart={!!job && selected.size > 0 && upload === null}
              starting={starting}
              onStart={onStart}
              notify={notify}
              fail={fail}
            />
          )}
          {job && job.state === "started" && health && (
            <RunPanel
              job={job}
              health={health}
              onCancel={() => api.cancel(job.id).then(setJob).catch(fail)}
              onRetry={() => api.retryJob(job.id).then(setJob).catch(fail)}
              onNew={newJob}
              onReview={() => {
                const first = doneItems.find((i) => i.result?.qa.needs_review) ?? doneItems[0];
                if (first) setCompareId(first.id);
              }}
            />
          )}
        </aside>
      </main>

      {compareId && (
        <CompareViewer items={doneItems} currentId={compareId} setCurrentId={setCompareId} onClose={() => setCompareId(null)} />
      )}
      {historyOpen && (
        <HistoryDrawer
          currentId={job?.id}
          onOpen={openJob}
          onClose={() => setHistoryOpen(false)}
          onDeleted={(id) => {
            if (job?.id === id) newJob();
          }}
          fail={fail}
        />
      )}
      {needKey && (
        <div className="modal-backdrop">
          <form
            className="key-dialog"
            onSubmit={(e) => {
              e.preventDefault();
              const v = new FormData(e.currentTarget).get("key") as string;
              apiKey.set(v.trim());
              setNeedKey(false);
              boot();
            }}
          >
            <h2>需要访问密钥</h2>
            <p>服务已开启访问控制，请输入管理员提供的 API Key。</p>
            <input name="key" type="password" autoFocus defaultValue={apiKey.get()} />
            <button className="btn primary">保存并重试</button>
          </form>
        </div>
      )}
      {toast && (
        <div className={`toast ${toast.bad ? "bad" : ""}`} role="status">
          {toast.text}
        </div>
      )}
    </div>
  );
}
