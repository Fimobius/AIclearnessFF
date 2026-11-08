import type { Dispatch, SetStateAction } from "react";
import { formatBytes, formatDuration, withKey, type Item, type Job } from "../api";
import DropZone from "./DropZone";

interface Props {
  job: Job;
  selected: Set<string>;
  setSelected: Dispatch<SetStateAction<Set<string>>>;
  onFiles: (files: File[]) => void;
  uploading: number | null;
  onOpen: (id: string) => void;
  onRetry: (id: string) => void;
}

export default function ItemGrid({ job, selected, setSelected, onFiles, uploading, onOpen, onRetry }: Props) {
  const items = job.items ?? [];
  const draft = job.state === "draft";
  const selectable = items.filter((i) => i.status === "uploaded");
  const allOn = selectable.length > 0 && selectable.every((i) => selected.has(i.id));

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <div className="grid-wrap">
      <div className="grid-head">
        <div>
          <h2>{job.name}</h2>
          <p className="muted">
            {draft
              ? `已选 ${selected.size} 张，共 ${items.length} 个文件${job.counts.invalid ? `，${job.counts.invalid} 个无法处理` : ""}`
              : `共处理 ${items.length - job.counts.invalid - job.counts.skipped} 张`}
          </p>
        </div>
        {draft && (
          <div className="grid-actions">
            <label className="check">
              <input
                type="checkbox"
                checked={allOn}
                onChange={() => setSelected(allOn ? new Set() : new Set(selectable.map((i) => i.id)))}
              />
              全选
            </label>
            <DropZone onFiles={onFiles} uploading={uploading} />
          </div>
        )}
      </div>
      <ul className="grid">
        {items.map((it) => (
          <ItemCard
            key={it.id}
            item={it}
            draft={draft}
            checked={selected.has(it.id)}
            onToggle={() => toggle(it.id)}
            onOpen={() => onOpen(it.id)}
            onRetry={() => onRetry(it.id)}
          />
        ))}
      </ul>
    </div>
  );
}

function ItemCard({ item, draft, checked, onToggle, onOpen, onRetry }: {
  item: Item;
  draft: boolean;
  checked: boolean;
  onToggle: () => void;
  onOpen: () => void;
  onRetry: () => void;
}) {
  const s = item.status;
  const review = item.result?.qa.needs_review;
  const dim = item.width ? `${item.width}×${item.height}` : "";
  const clickable = s === "done";
  const off = (draft && !checked && s === "uploaded") || s === "skipped" || s === "canceled";

  return (
    <li className={`card s-${s} ${off ? "off" : ""} ${review ? "review" : ""}`}>
      <div
        className="thumb"
        role={clickable || (draft && s === "uploaded") ? "button" : undefined}
        tabIndex={clickable || (draft && s === "uploaded") ? 0 : -1}
        onClick={clickable ? onOpen : draft && s === "uploaded" ? onToggle : undefined}
        onKeyDown={(e) => {
          if (e.key !== "Enter" && e.key !== " ") return;
          e.preventDefault();
          if (clickable) onOpen();
          else if (draft && s === "uploaded") onToggle();
        }}
        title={clickable ? "查看对比" : undefined}
      >
        {s !== "invalid" && <img src={withKey(item.urls.thumb)} alt="" loading="lazy" />}
        {s === "invalid" && <span className="thumb-x">无法读取</span>}
        {s === "processing" && (
          <div className="thumb-progress">
            <i style={{ width: `${item.progress * 100}%` }} />
          </div>
        )}
      </div>
      {draft && s === "uploaded" && (
        <input className="card-check" type="checkbox" checked={checked} onChange={onToggle} aria-label={`处理 ${item.original_name}`} />
      )}
      <div className="card-body">
        <div className="card-name" title={item.original_name}>
          {item.original_name}
        </div>
        <div className="card-meta">
          <StatusLine item={item} dim={dim} />
        </div>
        {s === "failed" && (
          <button className="link" onClick={onRetry}>
            重新处理
          </button>
        )}
        {s === "uploaded" && (
          <a className="link" href={withKey(item.urls.original)} target="_blank" rel="noreferrer">
            查看原图
          </a>
        )}
      </div>
    </li>
  );
}

function StatusLine({ item, dim }: { item: Item; dim: string }) {
  switch (item.status) {
    case "uploaded":
      return (
        <span className="num">
          {dim} <span className="muted">{formatBytes(item.size_bytes)}</span>
        </span>
      );
    case "invalid":
    case "failed":
      return <span className="err">{item.error}</span>;
    case "skipped":
      return <span className="muted">未选择，已跳过</span>;
    case "canceled":
      return <span className="muted">已取消</span>;
    case "queued":
      return <span className="muted">排队中{item.est_seconds ? `，预计 ${formatDuration(item.est_seconds)}` : ""}</span>;
    case "processing":
      return (
        <span className="accent">
          {item.stage ?? "处理中"} {Math.round(item.progress * 100)}%
        </span>
      );
    case "done": {
      const r = item.result!;
      return (
        <span>
          <span className="num">
            {r.width}×{r.height}
          </span>{" "}
          <span className="muted">{formatBytes(r.bytes)}</span>
          {r.qa.needs_review && <span className="tag warn">需复核</span>}
          {r.text?.protected ? <span className="tag info">文字已保护</span> : null}
        </span>
      );
    }
  }
}
