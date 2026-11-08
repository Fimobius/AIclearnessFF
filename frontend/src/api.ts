// Typed client for the FastAPI backend (see backend/app/main.py).

export type Tier = "standard" | "strong";
export type Mode = "pad" | "crop" | "stretch";
export type Format = "jpg" | "png" | "webp";

export interface Settings {
  tier: Tier;
  width: number;
  ratio_w: number | null;
  ratio_h: number | null;
  height: number | null;
  mode: Mode;
  background: string;
  crop_anchor: "center" | "subject";
  trim_border: boolean;
  text_protect: boolean;
  format: Format;
  quality: number;
  max_kb: number | null;
  name_suffix: string;
}

export type ItemStatus = "uploaded" | "invalid" | "skipped" | "queued" | "processing" | "done" | "failed" | "canceled";

export interface QA {
  ssim: number;
  delta_e_mean: number;
  delta_e_p95: number;
  sharpness_gain: number;
  needs_review: boolean;
  thresholds: { delta_e_mean_max: number; ssim_min: number };
}

export interface TextInfo {
  regions: number;
  protected: number;
  threshold_src_px: number;
  protected_area: number;
  protected_boxes: [number, number][][];
}

export interface Item {
  id: string;
  idx: number;
  original_name: string;
  size_bytes: number;
  width: number | null;
  height: number | null;
  format: string | null;
  status: ItemStatus;
  error: string | null;
  output_name: string | null;
  est_seconds: number | null;
  attempts: number;
  stage: string | null;
  progress: number;
  result?: {
    width: number;
    height: number;
    bytes: number;
    format: string;
    qa: QA;
    flags: string[];
    warnings: string[];
    seconds: number;
    model: string;
    magnification: number;
    text: TextInfo | null;
  };
  urls: { thumb: string; original: string; result?: string; baseline?: string; download?: string };
}

export type JobStatus = "draft" | "queued" | "running" | "done" | "done_with_errors" | "canceled";

export interface Job {
  id: string;
  name: string;
  created_at: number;
  started_at: number | null;
  state: "draft" | "started";
  settings: Settings | null;
  status: JobStatus;
  counts: Record<ItemStatus, number>;
  total: number;
  progress: number;
  eta_seconds: number;
  needs_review: number;
  items?: Item[];
}

export interface Preset {
  id: number;
  name: string;
  width: number;
  ratio_w: number | null;
  ratio_h: number | null;
  height: number | null;
  builtin: boolean;
  out_width: number;
  out_height: number;
  exact_ratio: boolean;
}

export interface SizeInfo {
  width: number;
  height: number;
  exact_ratio: boolean;
  ratio: string;
  suggested_widths: number[];
}

export interface Health {
  version: string;
  device: { device: string; name: string; vram_gb?: number } | null;
  tiers: Record<Tier, { label: string; desc: string; model: string }>;
  allow_stretch: boolean;
  text_protect_available: boolean;
  auth_required: boolean;
  limits: { max_file_mb: number; max_files_per_job: number; max_job_mb: number; min_side: number; max_side: number };
  defaults: Settings;
}

const KEY_STORAGE = "aiclear.apiKey";
export const apiKey = {
  get: () => localStorage.getItem(KEY_STORAGE) || "",
  set: (k: string) => localStorage.setItem(KEY_STORAGE, k),
};

/** Append the API key to URLs used directly by <img> / <a download>. */
export function withKey(url: string): string {
  const k = apiKey.get();
  return k ? `${url}${url.includes("?") ? "&" : "?"}key=${encodeURIComponent(k)}` : url;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const k = apiKey.get();
  if (k) headers.set("X-API-Key", k);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const res = await fetch(path, { ...init, headers });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  health: () => call<Health>("/api/health"),
  size: (s: { width: number; ratio_w?: number | null; ratio_h?: number | null; height?: number | null }) => {
    const q = new URLSearchParams({ width: String(s.width) });
    if (s.height) q.set("height", String(s.height));
    else {
      if (s.ratio_w != null) q.set("ratio_w", String(s.ratio_w));
      if (s.ratio_h != null) q.set("ratio_h", String(s.ratio_h));
    }
    return call<SizeInfo>(`/api/size?${q}`);
  },
  presets: () => call<Preset[]>("/api/presets"),
  createPreset: (p: Omit<Preset, "id" | "builtin" | "out_width" | "out_height" | "exact_ratio">) =>
    call<Preset>("/api/presets", { method: "POST", body: JSON.stringify(p) }),
  deletePreset: (id: number) => call<void>(`/api/presets/${id}`, { method: "DELETE" }),
  createJob: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return call<Job>("/api/jobs", { method: "POST", body: fd });
  },
  addFiles: (jobId: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return call<Job>(`/api/jobs/${jobId}/files`, { method: "POST", body: fd });
  },
  start: (jobId: string, settings: Settings, itemIds: string[]) =>
    call<Job>(`/api/jobs/${jobId}/start`, { method: "POST", body: JSON.stringify({ settings, item_ids: itemIds }) }),
  job: (id: string) => call<Job>(`/api/jobs/${id}`),
  jobs: () => call<Job[]>("/api/jobs?limit=100"),
  cancel: (id: string) => call<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) => call<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),
  retryItem: (id: string) => call<Item>(`/api/items/${id}/retry`, { method: "POST" }),
  deleteJob: (id: string) => call<void>(`/api/jobs/${id}`, { method: "DELETE" }),
};

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDuration(s: number): string {
  if (s < 1) return "不到 1 秒";
  if (s < 60) return `${Math.round(s)} 秒`;
  const m = Math.floor(s / 60);
  return `${m} 分 ${Math.round(s % 60)} 秒`;
}
