import { useRef, useState } from "react";
import type { Health } from "../api";

const ACCEPT = ".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp";

interface Props {
  onFiles: (files: File[]) => void;
  uploading: number | null;
  limits?: Health["limits"];
  big?: boolean;
}

export default function DropZone({ onFiles, uploading, limits, big }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const busy = uploading !== null;

  const pick = () => input.current?.click();

  return (
    <div
      className={`dropzone ${big ? "big" : "slim"} ${over ? "over" : ""} ${busy ? "busy" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!busy) onFiles([...e.dataTransfer.files]);
      }}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          onFiles([...(e.target.files ?? [])]);
          e.target.value = "";
        }}
      />
      {busy ? (
        <div className="upload-progress">
          <span>正在上传 {Math.round(uploading! * 100)}%</span>
          <div className="bar">
            <i style={{ width: `${uploading! * 100}%` }} />
          </div>
        </div>
      ) : big ? (
        <>
          <svg className="drop-frame" viewBox="0 0 120 90" aria-hidden>
            <path d="M8 22V8h14M98 8h14v14M112 68v14H98M22 82H8V68" />
            <rect x="34" y="26" width="52" height="38" />
          </svg>
          <h1>把需要变清晰的图片拖到这里</h1>
          <p>
            支持 JPG、PNG、WEBP，可以一次拖入整个文件夹里的图片
            {limits && `；单张不超过 ${limits.max_file_mb} MB，每批最多 ${limits.max_files_per_job} 张`}
          </p>
          <button className="btn primary" onClick={pick}>
            选择图片
          </button>
        </>
      ) : (
        <button className="btn ghost" onClick={pick}>
          添加图片
        </button>
      )}
    </div>
  );
}
