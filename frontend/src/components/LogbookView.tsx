import { useEffect, useState } from 'react';
import { Calendar, Download, Plus, Trash2, CheckCircle2 } from 'lucide-react';
import { api, ObservingSession } from '../api/client';

export function LogbookView() {
  const [sessions, setSessions] = useState<ObservingSession[]>([]);
  const [title, setTitle] = useState('');
  const [obsDate, setObsDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [siteName, setSiteName] = useState('');
  const [rigName, setRigName] = useState('');
  const [targetObj, setTargetObj] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const res = await api.listSessions();
      setSessions(res.sessions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;
    setError(null);
    await api.createSession({
      title: title.trim(),
      obs_date: obsDate,
      site_name: siteName.trim(),
      rig_name: rigName.trim(),
    });
    setTitle('');
    await refresh();
  }

  async function handleAddTarget(sessionId: string) {
    const raw = (targetObj[sessionId] || '').trim();
    if (!raw) return;
    await api.addSessionTarget(sessionId, { object_id: raw, status: 'imaged', exposure_minutes: 60 });
    setTargetObj((prev) => ({ ...prev, [sessionId]: '' }));
    await refresh();
  }

  return (
    <div className="space-y-6">
      <form
        onSubmit={(e) => void handleCreate(e)}
        className="grid gap-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-4 md:grid-cols-5"
      >
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Session title (e.g. Autumn Galaxies)"
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white md:col-span-2"
          required
        />
        <input
          type="date"
          value={obsDate}
          onChange={(e) => setObsDate(e.target.value)}
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
        />
        <input
          value={siteName}
          onChange={(e) => setSiteName(e.target.value)}
          placeholder="Site (e.g. Fontainebleau)"
          className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
        />
        <div className="flex gap-2">
          <input
            value={rigName}
            onChange={(e) => setRigName(e.target.value)}
            placeholder="Rig"
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          />
          <button
            type="submit"
            className="inline-flex shrink-0 items-center gap-1 rounded-lg bg-sky-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-sky-400"
          >
            <Plus className="h-4 w-4" /> Plan
          </button>
        </div>
      </form>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/50 p-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="space-y-3">
        {sessions.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-800 p-10 text-center text-sm text-slate-400">
            No observing sessions logged yet. Plan your first night above or log targets from Sky Vision.
          </div>
        ) : (
          sessions.map((s) => (
            <div
              key={s.id}
              className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4 transition hover:border-slate-700"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4 text-sky-400" />
                    <h3 className="font-semibold text-white">{s.title}</h3>
                    <span className="rounded-full bg-slate-800 px-2.5 py-0.5 font-mono text-xs text-slate-300">
                      {s.obs_date}
                    </span>
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        s.status === 'completed'
                          ? 'bg-emerald-500/15 text-emerald-300'
                          : s.status === 'cancelled'
                            ? 'bg-rose-500/15 text-rose-300'
                            : 'bg-amber-500/15 text-amber-300'
                      }`}
                    >
                      {s.status}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-400">
                    {s.site_name || 'No site'} · {s.rig_name || 'No rig'} · {s.targets.length} target(s)
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2 text-xs">
                  {s.status !== 'completed' && (
                    <button
                      type="button"
                      onClick={() => void api.updateSession(s.id, { status: 'completed' }).then(refresh)}
                      className="inline-flex items-center gap-1 rounded-lg border border-emerald-500/40 px-2.5 py-1.5 text-emerald-300 hover:bg-emerald-500/10"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" /> Complete
                    </button>
                  )}
                  <a
                    href={`/api/logbook/sessions/${s.id}/export/ics`}
                    className="inline-flex items-center gap-1 rounded-lg border border-slate-700 px-2.5 py-1.5 text-slate-200 hover:bg-slate-800"
                  >
                    <Download className="h-3.5 w-3.5" /> .ics
                  </a>
                  <a
                    href={`/api/logbook/sessions/${s.id}/export/nina_csv`}
                    className="inline-flex items-center gap-1 rounded-lg border border-slate-700 px-2.5 py-1.5 text-slate-200 hover:bg-slate-800"
                  >
                    <Download className="h-3.5 w-3.5" /> N.I.N.A. CSV
                  </a>
                  <a
                    href={`/api/logbook/sessions/${s.id}/export/stellarium`}
                    className="inline-flex items-center gap-1 rounded-lg border border-slate-700 px-2.5 py-1.5 text-slate-200 hover:bg-slate-800"
                  >
                    <Download className="h-3.5 w-3.5" /> Stellarium
                  </a>
                  <button
                    type="button"
                    onClick={() => {
                      if (window.confirm(`Permanently delete "${s.title}"?`)) {
                        void api.deleteSession(s.id).then(refresh);
                      }
                    }}
                    className="inline-flex items-center gap-1 rounded-lg border border-rose-500/30 px-2 py-1.5 text-rose-300 hover:bg-rose-500/10"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                {s.targets.map((t) => (
                  <span
                    key={t.id}
                    className="rounded-lg bg-slate-950 px-2.5 py-1 font-mono text-xs text-sky-300 ring-1 ring-slate-800"
                  >
                    {t.object_id} {t.object_name ? `· ${t.object_name}` : ''} ({t.exposure_minutes}m)
                  </span>
                ))}
                <div className="flex items-center gap-1">
                  <input
                    value={targetObj[s.id] || ''}
                    onChange={(e) => setTargetObj((prev) => ({ ...prev, [s.id]: e.target.value }))}
                    placeholder="Add target (M31, C20…)"
                    className="w-40 rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1 text-xs text-white"
                  />
                  <button
                    type="button"
                    onClick={() => void handleAddTarget(s.id)}
                    className="rounded-lg bg-slate-800 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-700"
                  >
                    + Log
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
