import { useCallback, useEffect, useRef, useState } from "react";
import { formatBytes, withKey, type Item } from "../api";

interface Props {
  items: Item[];
  currentId: string;
  setCurrentId: (id: string) => void;
  onClose: () => void;
}

type View = "slider" | "side";

export default function CompareViewer({ items, currentId, setCurrentId, onClose }: Props) {
  const idx = Math.max(0, items.findIndex((i) => i.id === currentId));
  const item = items[idx];
  const [view, setView] = useState<View>("slider");
  const [pos, setPos] = useState(50);
  const [zoom, setZoom] = useState(false);
  const [showText, setShowText] = useState(true);
  const stage = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const go = useCallback(
    (d: number) => {
      const n = items[idx + d];
      if (n) setCurrentId(n.id);
    },
    [idx, items, setCurrentId],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "z" || e.key === "Z") setZoom((z) => !z);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, onClose]);

  const move = (clientX: number) => {
    const el = stage.current?.querySelector(".cmp-frame") as HTMLElement | null;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };

  if (!item || !item.result) return null;
  const r = item.result;
  const qa = r.qa;
  const deOk = qa.delta_e_mean <= qa.thresholds.delta_e_mean_max;
  const ssimOk = qa.ssim >= qa.thresholds.ssim_min;

  return (
    <div className="viewer" role="dialog" aria-modal aria-label="效果对比">
      <header className="viewer-bar">
        <div className="viewer-title">
          <strong title={item.original_name}>{item.output_name}</strong>
          <span className="muted num">
            {idx + 1} / {items.length}
          </span>
        </div>
        <div className="seg two small-seg dark">
          <button className={`seg-btn ${view === "slider" ? "on" : ""}`} onClick={() => setView("slider")}>
            滑动对比
          </button>
          <button className={`seg-btn ${view === "side" ? "on" : ""}`} onClick={() => setView("side")}>
            左右并排
          </button>
        </div>
        <div className="viewer-tools">
          {r.text && r.text.protected > 0 && (
            <button className="btn dark small" onClick={() => setShowText((v) => !v)} aria-pressed={showText}>
              {showText ? "隐藏文字保护框" : "显示文字保护框"}
            </button>
          )}
          <button className="btn dark small" onClick={() => setZoom((z) => !z)} title="快捷键 Z">
            {zoom ? "适应窗口" : "原始像素 100%"}
          </button>
          <a className="btn dark small" href={withKey(item.urls.download!)}>
            下载这张
          </a>
          <button className="btn dark small" onClick={onClose} aria-label="关闭（Esc）">
            关闭
          </button>
        </div>
      </header>

      <div
        ref={stage}
        className={`viewer-stage ${zoom ? "zoom" : ""}`}
        onPointerMove={(e) => dragging.current && move(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => (dragging.current = false)}
      >
        {view === "slider" ? (
          <div
            className="cmp-frame"
            style={{ aspectRatio: `${r.width} / ${r.height}`, width: zoom ? r.width : undefined }}
            onPointerDown={(e) => {
              dragging.current = true;
              move(e.clientX);
            }}
          >
            <img className="checker" src={withKey(item.urls.result!)} alt="高清结果" draggable={false} />
            <img className="cmp-top checker" src={withKey(item.urls.baseline!)} alt="普通缩放" draggable={false} style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
            {showText && r.text && r.text.protected_boxes.length > 0 && (
              <svg className="text-overlay" viewBox={`0 0 ${r.width} ${r.height}`} preserveAspectRatio="none" aria-hidden>
                {r.text.protected_boxes.map((b, i) => (
                  <polygon key={i} points={b.map((p) => p.join(",")).join(" ")} />
                ))}
              </svg>
            )}
            <div className="cmp-handle" style={{ left: `${pos}%` }}>
              <span className="cmp-label left">普通缩放</span>
              <span className="cmp-label right">AI 高清</span>
            </div>
            <input
              className="cmp-range"
              type="range"
              min={0}
              max={100}
              value={pos}
              onChange={(e) => setPos(Number(e.target.value))}
              aria-label="对比分割线位置"
            />
          </div>
        ) : (
          <div className="cmp-side">
            <figure>
              <img className="checker" src={withKey(item.urls.baseline!)} alt="普通缩放" style={{ width: zoom ? r.width : undefined }} />
              <figcaption>普通缩放（未经 AI）</figcaption>
            </figure>
            <figure>
              <img className="checker" src={withKey(item.urls.result!)} alt="AI 高清" style={{ width: zoom ? r.width : undefined }} />
              <figcaption>AI 高清结果</figcaption>
            </figure>
          </div>
        )}
        {idx > 0 && (
          <button className="nav prev" onClick={() => go(-1)} aria-label="上一张">
            ‹
          </button>
        )}
        {idx < items.length - 1 && (
          <button className="nav next" onClick={() => go(1)} aria-label="下一张">
            ›
          </button>
        )}
      </div>

      <footer className="viewer-info">
        <div>
          <span className="k">输出</span>
          <span className="num">
            {r.width} × {r.height}
          </span>{" "}
          {r.format.toUpperCase()} {formatBytes(r.bytes)}
        </div>
        <div>
          <span className="k">放大</span>
          <span className="num">{r.magnification.toFixed(2)} 倍</span>
        </div>
        <div title="处理后缩回原尺寸，与原图的结构相似度；越接近 1 越说明没有改动画面内容">
          <span className="k">结构保真</span>
          <span className={`num ${ssimOk ? "" : "bad"}`}>{qa.ssim.toFixed(3)}</span>
        </div>
        <div title="与原图的平均色差 ΔE2000；小于 2.3 人眼基本察觉不到">
          <span className="k">色差</span>
          <span className={`num ${deOk ? "" : "bad"}`}>ΔE {qa.delta_e_mean.toFixed(2)}</span>
        </div>
        <div title="相对普通缩放的细节锐度提升">
          <span className="k">清晰度</span>
          <span className="num">{qa.sharpness_gain.toFixed(1)} 倍</span>
        </div>
        <div>
          <span className="k">耗时</span>
          <span className="num">{r.seconds.toFixed(1)} 秒</span>
        </div>
        {r.text && (
          <div title="原图中太小或太模糊、AI 无法可靠还原的文字区域保留原图笔画">
            <span className="k">文字保护</span>
            <span className="num">{r.text.protected} 处</span>
            {r.text.regions > r.text.protected && <span className="muted-dark">（另有 {r.text.regions - r.text.protected} 处文字较大，已用 AI 增强，请核对）</span>}
          </div>
        )}
        {qa.needs_review && <div className="tag warn">需复核：与原图差异超过阈值</div>}
        {r.warnings.length > 0 && <div className="viewer-warn">{r.warnings.join("；")}</div>}
      </footer>
    </div>
  );
}
