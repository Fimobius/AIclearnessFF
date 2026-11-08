import { useEffect, useState } from "react";
import { api, withKey, type Job } from "../api";

interface Props {
  currentId?: string;
  onOpen: (id: string) => void;
  onClose: () => void;
  onDeleted: (id: string) => void;
  fail: (e: unknown) => void;
}

const STATUS: Record<Job["status"], string> = {
  draft: "未开始",
  queued: "排队中",
  running: "处理中",
  done: "已完成",
  done_with_errors: "部分失败",
  canceled: "已取消",
};

export default function HistoryDrawer({ currentId, onOpen, onClose, onDeleted, fail }: Props) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);

  useEffect(() => {
    api.jobs().then(setJobs).catch(fail);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fail, onClose]);

  const remove = async (id: string) => {
    try {
      await api.deleteJob(id);
      setJobs((j) => j?.filter((x) => x.id !== id) ?? null);
      onDeleted(id);
    } catch (e) {
      fail(e);
    } finally {
      setConfirm(null);
    }
  };

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="历史任务">
        <header>
          <h2>历史任务</h2>
          <button className="btn ghost small" onClick={onClose}>
            关闭
          </button>
        </header>
        {jobs === null ? (
          <p className="muted pad">加载中…</p>
        ) : jobs.length === 0 ? (
          <p className="muted pad">还没有任务。上传图片后会出现在这里，处理结果可以随时重新下载。</p>
        ) : (
          <ul className="history">
            {jobs.map((j) => (
              <li key={j.id} className={j.id === currentId ? "current" : ""}>
                <button className="history-main" onClick={() => onOpen(j.id)}>
                  <strong>{j.name}</strong>
                  <span className="muted">
                    {new Date(j.created_at * 1000).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                    ，{j.total} 张，{STATUS[j.status]}
                    {j.counts.done ? `，成功 ${j.counts.done}` : ""}
                    {j.counts.failed ? `，失败 ${j.counts.failed}` : ""}
                  </span>
                </button>
                <div className="history-actions">
                  {j.counts.done > 0 && (
                    <a className="link" href={withKey(`/api/jobs/${j.id}/download`)}>
                      下载
                    </a>
                  )}
                  {confirm === j.id ? (
                    <>
                      <button className="link bad" onClick={() => remove(j.id)}>
                        确认删除
                      </button>
                      <button className="link" onClick={() => setConfirm(null)}>
                        保留
                      </button>
                    </>
                  ) : (
                    <button className="link" onClick={() => setConfirm(j.id)}>
                      删除
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </aside>
    </div>
  );
}
