import { apiKey, ApiError, type Job } from "./api";

const BATCH = 20; // files per request, so huge batches show progress and never hit request limits

function post(url: string, files: File[], onProgress: (loaded: number) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    xhr.open("POST", url);
    const k = apiKey.get();
    if (k) xhr.setRequestHeader("X-API-Key", k);
    xhr.upload.onprogress = (e) => onProgress(e.loaded);
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(xhr.status, body?.detail ?? `上传失败（${xhr.status}）`));
    };
    xhr.onerror = () => reject(new ApiError(0, "网络错误，无法连接到服务"));
    xhr.send(fd);
  });
}

/** Upload files in batches. Creates a job if jobId is null. Reports overall progress 0..1. */
export async function uploadFiles(jobId: string | null, files: File[], onProgress: (frac: number) => void): Promise<Job> {
  const total = files.reduce((s, f) => s + f.size, 0) || 1;
  let sent = 0;
  let job: Job | null = null;
  for (let i = 0; i < files.length; i += BATCH) {
    const chunk = files.slice(i, i + BATCH);
    const url = jobId ? `/api/jobs/${jobId}/files` : "/api/jobs";
    job = await post(url, chunk, (loaded) => onProgress(Math.min(1, (sent + loaded) / total)));
    jobId = job.id;
    sent += chunk.reduce((s, f) => s + f.size, 0);
    onProgress(sent / total);
  }
  return job!;
}
