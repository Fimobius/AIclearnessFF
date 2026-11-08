import { formatDuration, withKey, type Health, type Job } from "../api";

interface Props {
  job: Job;
  health: Health;
  onCancel: () => void;
  onRetry: () => void;
  onNew: () => void;
  onReview: () => void;
}

const MODE_LABEL = { pad: "等比缩放补边", crop: "裁剪填满", stretch: "拉伸" } as const;
const BG_LABEL: Record<string, string> = { white: "白底", black: "黑底", transparent: "透明底" };

export default function RunPanel({ job, health, onCancel, onRetry, onNew, onReview }: Props) {
  const c = job.counts;
  const work = c.queued + c.processing + c.done + c.failed + c.canceled;
  const finished = c.done + c.failed;
  const running = job.status === "queued" || job.status === "running";
  const s = job.settings!;
  const sample = job.items?.find((i) => i.result)?.result;
  const headline = {
    draft: "",
    queued: "排队中",
    running: "正在处理",
    done: "全部完成",
    done_with_errors: "处理结束，有图片失败",
    canceled: "已取消",
  }[job.status];

  return (
    <div className="run">
      <div className={`run-head st-${job.status}`}>
        <h2>{headline}</h2>
        <div className="run-count num">
          {finished}
          <span className="of">/ {work}</span>
        </div>
        <div className="bar big">
          <i style={{ width: `${job.progress * 100}%` }} />
        </div>
        <p className="muted">{running ? (job.eta_seconds > 0 ? `预计还需 ${formatDuration(job.eta_seconds)}` : "即将完成") : `成功 ${c.done} 张${c.failed ? `，失败 ${c.failed} 张` : ""}${c.canceled ? `，取消 ${c.canceled} 张` : ""}`}</p>
      </div>

      {job.needs_review > 0 && (
        <div className="notice warn">
          <strong>{job.needs_review} 张需要人工复核</strong>
          <p>这些图片处理后与原图的结构或色彩差异超过阈值，可能有 AI 失真，请对比确认后再使用。</p>
          <button className="link" onClick={onReview}>
            逐张查看
          </button>
        </div>
      )}

      {(() => {
        const prot = job.items?.filter((i) => i.result?.text?.protected).length ?? 0;
        return prot > 0 ? (
          <div className="notice info">
            <strong>{prot} 张图片的小文字已保护</strong>
            <p>这些文字在原图中太小或太模糊，AI 无法可靠还原，已保留原图笔画以免生成错字。对比时可查看保护区域。</p>
          </div>
        ) : null;
      })()}

      <div className="run-actions">
        <a className={`btn primary block ${c.done ? "" : "disabled"}`} href={c.done ? withKey(`/api/jobs/${job.id}/download`) : undefined} aria-disabled={!c.done}>
          下载全部（ZIP，{c.done} 张）
        </a>
        {c.done > 0 && (
          <button className="btn block" onClick={onReview}>
            对比查看效果
          </button>
        )}
        {c.failed + c.canceled > 0 && !running && (
          <button className="btn block" onClick={onRetry}>
            重新处理失败的 {c.failed + c.canceled} 张
          </button>
        )}
        {running && c.queued > 0 && (
          <button className="btn ghost block" onClick={onCancel}>
            取消排队中的 {c.queued} 张
          </button>
        )}
        <a className="link" href={withKey(`/api/jobs/${job.id}/report.csv`)}>
          下载处理报告（CSV）
        </a>
      </div>

      <dl className="spec-summary">
        <dt>增强档位</dt>
        <dd>{health.tiers[s.tier]?.label}</dd>
        <dt>输出尺寸</dt>
        <dd className="num">{sample ? `${sample.width} × ${sample.height}` : s.height ? `${s.width} × ${s.height}` : `宽 ${s.width}，比例 ${s.ratio_w}:${s.ratio_h}`}</dd>
        <dt>缩放方式</dt>
        <dd>
          {MODE_LABEL[s.mode]}
          {s.mode === "pad" && `，${BG_LABEL[s.background] ?? s.background}`}
        </dd>
        <dt>导出</dt>
        <dd>
          {s.format.toUpperCase()}
          {s.format !== "png" && `，画质 ${s.quality}`}
          {s.max_kb ? `，不超过 ${s.max_kb} KB` : ""}
        </dd>
      </dl>

      <button className="btn ghost block" onClick={onNew}>
        用相同设置处理新的图片
      </button>
    </div>
  );
}
