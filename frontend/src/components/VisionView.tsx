import { useEffect, useState } from 'react';
import { Camera, CheckCircle2, Upload } from 'lucide-react';
import { api, AuthMeResponse, VisionPreset, VisionResult } from '../api/client';

export function VisionView({ solvers }: { solvers?: AuthMeResponse['vision_solvers'] }) {
  const [presets, setPresets] = useState<VisionPreset[]>([]);
  const [result, setResult] = useState<VisionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loggedMsg, setLoggedMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [raHint, setRaHint] = useState('');
  const [decHint, setDecHint] = useState('');
  const [fovHint, setFovHint] = useState('');

  useEffect(() => {
    void api.visionPresets().then((r) => setPresets(r.presets));
  }, []);

  async function runPreset(id: string) {
    setLoading(true);
    setError(null);
    setLoggedMsg(null);
    try {
      setResult(await api.visionIdentify({ preset_id: id }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Plate solve failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    setLoggedMsg(null);
    try {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(new Error('Could not read file'));
        reader.readAsDataURL(file);
      });
      const payload: Record<string, unknown> = { image_base64: dataUrl, filename: file.name };
      if (raHint.trim() && decHint.trim()) {
        payload.ra_hint_deg = Number(raHint);
        payload.dec_hint_deg = Number(decHint);
      }
      if (fovHint.trim()) {
        payload.field_width_deg = Number(fovHint);
      }
      setResult(await api.visionIdentify(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Plate solve failed');
    } finally {
      setLoading(false);
    }
  }

  async function logToSession() {
    if (!result?.result.objects.length) return;
    const res = await api.logVisionTargets({
      title: `Sky Vision (${result.result.objects.map((o) => o.doc_id).slice(0, 3).join(', ')})`,
      obs_date: new Date().toISOString().slice(0, 10),
      targets: result.result.objects.map((o) => ({
        object_id: o.doc_id,
        object_name: o.title,
        ra_deg: o.metadata.ra_deg,
        dec_deg: o.metadata.dec_deg,
      })),
    });
    setLoggedMsg(
      `Logged ${res.session.targets.length} object(s) to Observing Logbook (Messier: ${res.progress.messier.completed}/${res.progress.messier.total}).`
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex items-center gap-2 text-sm font-medium text-white">
          <Camera className="h-4 w-4 text-sky-400" /> Blind Plate-Solving Chain
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className={`rounded-full px-2.5 py-0.5 ${solvers?.astap ? 'bg-emerald-500/15 text-emerald-300' : 'bg-slate-800 text-slate-400'}`}>
            ASTAP local: {solvers?.astap ? 'ready' : 'not installed'}
          </span>
          <span className={`rounded-full px-2.5 py-0.5 ${solvers?.nova_astrometry_net ? 'bg-emerald-500/15 text-emerald-300' : 'bg-slate-800 text-slate-400'}`}>
            nova.astrometry.net: {solvers?.nova_astrometry_net ? 'key configured' : 'add key in Settings'}
          </span>
          <span className="rounded-full bg-sky-500/15 px-2.5 py-0.5 text-sky-300">TAN WCS fallback: ready</span>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-3">
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-950/60 p-8 text-center transition hover:border-sky-500">
            <Upload className="mb-2 h-7 w-7 text-sky-400" />
            <span className="text-sm font-medium text-white">Drop or select a sky photo (PNG, JPEG, FITS)</span>
            <span className="mt-1 text-xs text-slate-400">Solved via ASTAP → nova.astrometry.net → TAN WCS</span>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.fits,.fit"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                e.target.value = '';
                if (f) void handleFile(f);
              }}
            />
          </label>
          <div className="grid grid-cols-3 gap-2 rounded-xl border border-slate-800 bg-slate-900/40 p-3">
            <input
              value={raHint}
              onChange={(e) => setRaHint(e.target.value)}
              placeholder="RA deg (optional)"
              className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-white"
            />
            <input
              value={decHint}
              onChange={(e) => setDecHint(e.target.value)}
              placeholder="Dec deg (optional)"
              className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-white"
            />
            <input
              value={fovHint}
              onChange={(e) => setFovHint(e.target.value)}
              placeholder="FOV deg (optional)"
              className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-white"
            />
          </div>
        </div>

        <div className="space-y-2 rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
          <div className="text-xs font-medium uppercase tracking-wider text-slate-400">1-Click Test Presets</div>
          {presets.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => void runPreset(p.id)}
              className="flex w-full items-center justify-between rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-2.5 text-left text-sm transition hover:border-sky-500/50"
            >
              <div>
                <div className="font-medium text-white">{p.label}</div>
                <div className="text-xs text-slate-400">{p.subtitle}</div>
              </div>
              <span className="rounded-full bg-sky-500/15 px-2.5 py-0.5 text-xs text-sky-300">{p.badge}</span>
            </button>
          ))}
        </div>
      </div>

      {loading && <div className="animate-pulse text-sm text-sky-300">Plate-solving sky field…</div>}
      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/50 p-3 text-sm text-rose-200">{error}</div>
      )}
      {loggedMsg && (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/40 bg-emerald-950/40 p-3 text-sm text-emerald-200">
          <CheckCircle2 className="h-4 w-4" /> {loggedMsg}
        </div>
      )}

      {result && (
        <div className="grid gap-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-5 lg:grid-cols-2">
          {result.annotated_image_data_url && (
            <img
              src={result.annotated_image_data_url}
              alt="Annotated sky capture"
              className="w-full rounded-xl border border-slate-800"
            />
          )}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-white">
                {result.result.object_count} object(s) matched ({result.result.astrometry.method})
              </div>
              {result.result.objects.length > 0 && (
                <button
                  type="button"
                  onClick={() => void logToSession()}
                  className="rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-medium text-slate-950 hover:bg-emerald-400"
                >
                  Log to Observing Logbook
                </button>
              )}
            </div>
            <div className="space-y-1.5">
              {result.result.objects.map((o) => (
                <div
                  key={o.doc_id}
                  className="flex items-center justify-between rounded-lg bg-slate-950 px-3 py-2 text-xs"
                >
                  <span className="font-mono font-semibold text-sky-300">{o.doc_id}</span>
                  <span className="text-slate-200">{o.title}</span>
                  <span className="text-slate-400">{o.metadata.constellation || ''}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
