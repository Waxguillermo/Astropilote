import { useEffect, useState } from 'react';
import { Award } from 'lucide-react';
import { api, CatalogProgressSummary } from '../api/client';

export function CatalogTrackerView() {
  const [summary, setSummary] = useState<CatalogProgressSummary | null>(null);
  const [objId, setObjId] = useState('');
  const [status, setStatus] = useState<'observed' | 'imaged' | 'wishlist'>('observed');
  const [minutes, setMinutes] = useState('45');

  async function refresh() {
    setSummary(await api.getProgress());
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function handleLog(e: React.FormEvent) {
    e.preventDefault();
    if (!objId.trim()) return;
    const res = await api.upsertProgress({
      object_id: objId.trim(),
      status,
      add_exposure_minutes: Number(minutes) || 0,
    });
    setSummary(res.summary);
    setObjId('');
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="text-xs uppercase tracking-wider text-slate-400">Messier Catalog</div>
          <div className="mt-1 text-2xl font-bold text-sky-400">
            {summary?.messier.completed ?? 0} / {summary?.messier.total ?? 110}
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full bg-sky-400 transition-all"
              style={{ width: `${summary?.messier.pct ?? 0}%` }}
            />
          </div>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="text-xs uppercase tracking-wider text-slate-400">Caldwell Catalog</div>
          <div className="mt-1 text-2xl font-bold text-emerald-400">
            {summary?.caldwell.completed ?? 0} / {summary?.caldwell.total ?? 109}
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full bg-emerald-400 transition-all"
              style={{ width: `${summary?.caldwell.pct ?? 0}%` }}
            />
          </div>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="text-xs uppercase tracking-wider text-slate-400">Total Integration</div>
          <div className="mt-1 text-2xl font-bold text-amber-300">
            {((summary?.total_exposure_minutes ?? 0) / 60).toFixed(1)} hrs
          </div>
          <div className="mt-2 text-xs text-slate-400">{summary?.imaged_total ?? 0} imaged objects</div>
        </div>
      </div>

      <form
        onSubmit={(e) => void handleLog(e)}
        className="flex flex-wrap items-center gap-2 rounded-2xl border border-slate-800 bg-slate-900/50 p-4"
      >
        <Award className="h-5 w-5 text-amber-400" />
        <input
          value={objId}
          onChange={(e) => setObjId(e.target.value)}
          placeholder="Object ID (M31, C20, NGC7000…)"
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          required
        />
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value as 'observed' | 'imaged' | 'wishlist')}
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
        >
          <option value="observed">Observed (Visual)</option>
          <option value="imaged">Imaged (Astrophoto)</option>
          <option value="wishlist">Wishlist</option>
        </select>
        <input
          value={minutes}
          onChange={(e) => setMinutes(e.target.value)}
          placeholder="Minutes"
          className="w-24 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
        />
        <button
          type="submit"
          className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-sky-400"
        >
          Log Object
        </button>
      </form>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {(summary?.items || []).map((item) => (
          <div
            key={item.object_id}
            className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-2.5 text-sm"
          >
            <div>
              <span className="font-mono font-semibold text-white">{item.object_id}</span>
              <span className="ml-2 text-xs text-slate-400">{item.catalog}</span>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs text-sky-300">
              {item.status} · {item.total_exposure_minutes}m
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
