import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";
import { api, withKey, type Format, type Health, type Item, type Mode, type Preset, type Settings, type SizeInfo } from "../api";

interface Props {
  health: Health;
  presets: Preset[];
  setPresets: Dispatch<SetStateAction<Preset[]>>;
  settings: Settings;
  setSettings: Dispatch<SetStateAction<Settings | null>>;
  preview?: Item;
  count: number;
  canStart: boolean;
  starting: boolean;
  onStart: () => void;
  notify: (t: string, bad?: boolean) => void;
  fail: (e: unknown) => void;
}

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "pad", label: "等比缩放补边", hint: "完整保留画面，空白处填底色，推荐商品图" },
  { id: "crop", label: "裁剪填满", hint: "铺满画布，超出部分裁掉" },
  { id: "stretch", label: "拉伸", hint: "强制拉到目标尺寸，画面会变形" },
];

const FORMATS: { id: Format; label: string }[] = [
  { id: "jpg", label: "JPG" },
  { id: "png", label: "PNG" },
  { id: "webp", label: "WEBP" },
];

const BG_SWATCHES = [
  { id: "white", label: "白底" },
  { id: "black", label: "黑底" },
  { id: "transparent", label: "透明" },
];

const sameNum = (a: number | null, b: number | null) => (a == null && b == null) || (a != null && b != null && Math.abs(a - b) < 1e-9);

export default function SpecPanel(p: Props) {
  const { settings: s, setSettings, health } = p;
  const set = (patch: Partial<Settings>) => setSettings((prev) => ({ ...prev!, ...patch }));
  const byHeight = s.height != null;

  // Server-side size resolution (authoritative rounding + suggestions), debounced
  const [size, setSize] = useState<SizeInfo | null>(null);
  const [sizeErr, setSizeErr] = useState<string | null>(null);
  useEffect(() => {
    const t = window.setTimeout(() => {
      api
        .size({ width: s.width, ratio_w: s.ratio_w, ratio_h: s.ratio_h, height: s.height })
        .then((r) => {
          setSize(r);
          setSizeErr(null);
        })
        .catch((e) => {
          setSize(null);
          setSizeErr(e.message);
        });
    }, 200);
    return () => window.clearTimeout(t);
  }, [s.width, s.ratio_w, s.ratio_h, s.height]);

  const presetId = useMemo(() => {
    const m = p.presets.find(
      (x) => x.width === s.width && sameNum(x.height, s.height) && (x.height != null || (sameNum(x.ratio_w, s.ratio_w) && sameNum(x.ratio_h, s.ratio_h))),
    );
    return m ? String(m.id) : "custom";
  }, [p.presets, s.width, s.height, s.ratio_w, s.ratio_h]);
  const current = p.presets.find((x) => String(x.id) === presetId);

  const [saving, setSaving] = useState<string | null>(null);
  const savePreset = async () => {
    if (!saving?.trim()) return;
    try {
      const np = await api.createPreset({ name: saving.trim(), width: s.width, ratio_w: byHeight ? null : s.ratio_w, ratio_h: byHeight ? null : s.ratio_h, height: s.height });
      p.setPresets((prev) => [...prev, np]);
      setSaving(null);
      p.notify(`已保存模板「${np.name}」`);
    } catch (e) {
      p.fail(e);
    }
  };
  const deletePreset = async () => {
    if (!current || current.builtin) return;
    try {
      await api.deletePreset(current.id);
      p.setPresets((prev) => prev.filter((x) => x.id !== current.id));
      p.notify(`已删除模板「${current.name}」`);
    } catch (e) {
      p.fail(e);
    }
  };

  const modes = MODES.filter((m) => m.id !== "stretch" || health.allow_stretch);
  const valid = !!size && !sizeErr;

  return (
    <div className="spec">
      <FramePreview settings={s} size={size} item={p.preview} />

      <div className="readout" aria-live="polite">
        {size ? (
          <>
            <div className="dims num">
              {size.width}
              <span className="times">×</span>
              {size.height}
            </div>
            <div className="dims-sub">
              像素，比例 {size.ratio}
              {!size.exact_ratio && <span className="warn-text">（除不尽，高度已四舍五入）</span>}
            </div>
            {!size.exact_ratio && size.suggested_widths.length > 0 && (
              <div className="suggest">
                比例精确的宽度：
                {size.suggested_widths.map((w) => (
                  <button key={w} className="chip" onClick={() => set({ width: w })}>
                    {w}
                  </button>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="err">{sizeErr ?? "计算中…"}</div>
        )}
      </div>

      <fieldset className="group">
        <legend>增强档位</legend>
        <div className="seg two">
          {(Object.keys(health.tiers) as (keyof Health["tiers"])[]).map((k) => (
            <button key={k} className={`seg-btn tall ${s.tier === k ? "on" : ""}`} onClick={() => set({ tier: k })} aria-pressed={s.tier === k}>
              <strong>{health.tiers[k].label}</strong>
              <span>{health.tiers[k].desc}</span>
            </button>
          ))}
        </div>
        {health.text_protect_available && (
          <label className="check top">
            <input type="checkbox" checked={s.text_protect} onChange={(e) => set({ text_protect: e.target.checked })} />
            <span>
              保护小文字（推荐）
              <span className="hint block">原图里太小或太模糊的文字、数字保留原图笔画，不让 AI 补画，避免生成错字</span>
            </span>
          </label>
        )}
      </fieldset>

      <fieldset className="group">
        <legend>输出尺寸</legend>
        <div className="row">
          <select
            value={presetId}
            onChange={(e) => {
              const x = p.presets.find((q) => String(q.id) === e.target.value);
              if (x) set({ width: x.width, ratio_w: x.ratio_w, ratio_h: x.ratio_h, height: x.height });
            }}
            aria-label="尺寸模板"
          >
            {p.presets.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name}
              </option>
            ))}
            <option value="custom" disabled>
              自定义尺寸
            </option>
          </select>
          {current && !current.builtin && (
            <button className="btn ghost small" onClick={deletePreset} title="删除这个模板">
              删除
            </button>
          )}
        </div>
        <div className="seg two small-seg">
          <button className={`seg-btn ${!byHeight ? "on" : ""}`} onClick={() => set({ height: null, ratio_w: s.ratio_w ?? 10, ratio_h: s.ratio_h ?? 10 })}>
            宽度 + 宽高比
          </button>
          <button className={`seg-btn ${byHeight ? "on" : ""}`} onClick={() => set({ height: size?.height ?? s.width })}>
            宽 × 高
          </button>
        </div>
        <div className="size-inputs">
          <label>
            宽
            <NumInput value={s.width} onChange={(v) => set({ width: Math.round(v) })} min={health.limits.min_side} max={health.limits.max_side} />
            <span className="unit">px</span>
          </label>
          {byHeight ? (
            <label>
              高
              <NumInput value={s.height!} onChange={(v) => set({ height: Math.round(v) })} min={health.limits.min_side} max={health.limits.max_side} />
              <span className="unit">px</span>
            </label>
          ) : (
            <label className="ratio">
              比例
              <NumInput value={s.ratio_w ?? 10} onChange={(v) => set({ ratio_w: v })} step={0.1} narrow />
              <span className="colon">:</span>
              <NumInput value={s.ratio_h ?? 10} onChange={(v) => set({ ratio_h: v })} step={0.1} narrow />
            </label>
          )}
        </div>
        {saving === null ? (
          <button className="link" onClick={() => setSaving("")} disabled={!valid}>
            保存为模板
          </button>
        ) : (
          <div className="row">
            <input autoFocus placeholder="模板名称，例如 详情页 750" value={saving} onChange={(e) => setSaving(e.target.value)} onKeyDown={(e) => e.key === "Enter" && savePreset()} maxLength={60} />
            <button className="btn small" onClick={savePreset} disabled={!saving.trim()}>
              保存
            </button>
            <button className="btn ghost small" onClick={() => setSaving(null)}>
              取消
            </button>
          </div>
        )}
      </fieldset>

      <fieldset className="group">
        <legend>缩放方式</legend>
        <div className={`seg ${modes.length === 3 ? "three" : "two"}`}>
          {modes.map((m) => (
            <button key={m.id} className={`seg-btn mode ${s.mode === m.id ? "on" : ""}`} onClick={() => set({ mode: m.id })} aria-pressed={s.mode === m.id} title={m.hint}>
              <ModeIcon mode={m.id} />
              {m.label}
            </button>
          ))}
        </div>
        <p className="hint">{MODES.find((m) => m.id === s.mode)?.hint}</p>
        {s.mode === "pad" && (
          <div className="row wrap">
            {BG_SWATCHES.map((b) => (
              <button key={b.id} className={`swatch ${s.background === b.id ? "on" : ""}`} onClick={() => set({ background: b.id })} aria-pressed={s.background === b.id}>
                <i className={`sw sw-${b.id}`} />
                {b.label}
              </button>
            ))}
            <label className={`swatch ${s.background.startsWith("#") ? "on" : ""}`}>
              <input type="color" value={s.background.startsWith("#") ? s.background : "#f2f2f2"} onChange={(e) => set({ background: e.target.value })} />
              自定义
            </label>
          </div>
        )}
        {s.mode === "pad" && s.background === "transparent" && s.format === "jpg" && <p className="warn-text">JPG 不支持透明，透明区域会填成白色。需要透明请选 PNG 或 WEBP。</p>}
        {s.mode === "crop" && (
          <div className="seg two small-seg">
            <button className={`seg-btn ${s.crop_anchor === "subject" ? "on" : ""}`} onClick={() => set({ crop_anchor: "subject" })}>
              以商品主体为中心
            </button>
            <button className={`seg-btn ${s.crop_anchor === "center" ? "on" : ""}`} onClick={() => set({ crop_anchor: "center" })}>
              以画面中心为准
            </button>
          </div>
        )}
        <label className="check">
          <input type="checkbox" checked={s.trim_border} onChange={(e) => set({ trim_border: e.target.checked })} />
          先去掉原图四周的纯色留边
        </label>
      </fieldset>

      <fieldset className="group">
        <legend>导出</legend>
        <div className="seg three">
          {FORMATS.map((f) => (
            <button key={f.id} className={`seg-btn ${s.format === f.id ? "on" : ""}`} onClick={() => set({ format: f.id })} aria-pressed={s.format === f.id}>
              {f.label}
            </button>
          ))}
        </div>
        {s.format !== "png" && (
          <label className="slider">
            画质
            <input type="range" min={0} max={100} value={s.quality} onChange={(e) => set({ quality: Number(e.target.value) })} />
            <span className="num val">{s.quality}</span>
          </label>
        )}
        <label className="inline">
          文件大小上限
          <input
            type="number"
            min={10}
            placeholder="不限"
            value={s.max_kb ?? ""}
            onChange={(e) => set({ max_kb: e.target.value ? Math.max(10, Number(e.target.value)) : null })}
          />
          <span className="unit">KB</span>
        </label>
        <label className="inline">
          文件名后缀
          <input value={s.name_suffix} maxLength={40} onChange={(e) => set({ name_suffix: e.target.value.replace(/[<>:"/\\|?*]/g, "") })} />
        </label>
        <p className="hint">
          例：商品主图.jpg 导出为 商品主图{s.name_suffix}.{s.format}
        </p>
      </fieldset>

      <div className="start-bar">
        <button className="btn primary block" disabled={!p.canStart || !valid || p.starting} onClick={p.onStart}>
          {p.starting ? "正在提交…" : p.count ? `开始处理 ${p.count} 张图片` : "先添加并勾选图片"}
        </button>
      </div>
    </div>
  );
}

function NumInput({ value, onChange, min, max, step = 1, narrow }: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; narrow?: boolean }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return (
    <input
      className={`num ${narrow ? "narrow" : ""}`}
      type="number"
      inputMode="decimal"
      value={text}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        setText(e.target.value);
        const v = Number(e.target.value);
        if (e.target.value !== "" && Number.isFinite(v) && v > 0) onChange(v);
      }}
      onBlur={() => setText(String(value))}
    />
  );
}

function ModeIcon({ mode }: { mode: Mode }) {
  // Outer rect = output canvas, inner shape = the photo
  return (
    <svg className="mode-icon" viewBox="0 0 40 28" aria-hidden>
      <rect x="1" y="1" width="38" height="26" className="canvas" />
      {mode === "pad" && <rect x="9" y="5" width="22" height="18" className="photo" />}
      {mode === "crop" && (
        <>
          <rect x="-6" y="-4" width="52" height="36" className="photo ghost" />
          <rect x="1" y="1" width="38" height="26" className="photo" />
        </>
      )}
      {mode === "stretch" && <ellipse cx="20" cy="14" rx="18" ry="12" className="photo" />}
    </svg>
  );
}

function FramePreview({ settings: s, size, item }: { settings: Settings; size: SizeInfo | null; item?: Item }) {
  const W = 300, H = 210, M = 18; // drawing box and crop-mark margin
  const ar = size ? size.width / size.height : 1;
  const fw = Math.min(W - 2 * M, (H - 2 * M) * ar);
  const fh = fw / ar;
  const x = (W - fw) / 2, y = (H - fh) / 2;
  const fit = s.mode === "pad" ? "contain" : s.mode === "crop" ? "cover" : "fill";
  const bg = s.mode !== "pad" ? "transparent" : s.background;
  const L = 12, G = 5; // crop-mark length and gap
  const marks = [
    [x, y, -1, -1], [x + fw, y, 1, -1], [x, y + fh, -1, 1], [x + fw, y + fh, 1, 1],
  ] as const;

  return (
    <div className="frame-preview" aria-label="输出构图预览">
      <svg viewBox={`0 0 ${W} ${H}`} className="frame-svg">
        {marks.map(([cx, cy, dx, dy], i) => (
          <g key={i} className="crop-mark">
            <line x1={cx + dx * G} y1={cy} x2={cx + dx * (G + L)} y2={cy} />
            <line x1={cx} y1={cy + dy * G} x2={cx} y2={cy + dy * (G + L)} />
          </g>
        ))}
      </svg>
      <div
        className={`frame-box ${bg === "transparent" && s.mode === "pad" ? "checker" : ""}`}
        style={{ left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%`, width: `${(fw / W) * 100}%`, height: `${(fh / H) * 100}%`, background: bg === "transparent" ? undefined : bg }}
      >
        {item && item.status !== "invalid" ? (
          <img src={withKey(item.urls.thumb)} alt="" style={{ objectFit: fit }} />
        ) : (
          <span className="frame-empty">上传后在这里预览构图</span>
        )}
      </div>
    </div>
  );
}
