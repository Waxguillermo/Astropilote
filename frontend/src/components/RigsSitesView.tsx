import { useEffect, useState } from 'react';
import { MapPin, Telescope, Trash2 } from 'lucide-react';
import { api, Rig, Site } from '../api/client';

export function RigsSitesView() {
  const [rigs, setRigs] = useState<Rig[]>([]);
  const [sites, setSites] = useState<Site[]>([]);
  const [rigName, setRigName] = useState('');
  const [focal, setFocal] = useState('250');
  const [aperture, setAperture] = useState('51');
  const [siteName, setSiteName] = useState('');
  const [lat, setLat] = useState('48.4047');
  const [lon, setLon] = useState('2.7016');
  const [bortle, setBortle] = useState('5');

  async function refresh() {
    const [r, s] = await Promise.all([api.listRigs(), api.listSites()]);
    setRigs(r.rigs);
    setSites(s.sites);
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function addRig(e: React.FormEvent) {
    e.preventDefault();
    if (!rigName.trim()) return;
    await api.saveRig({
      name: rigName.trim(),
      focal_length_mm: Number(focal) || 250,
      aperture_mm: Number(aperture) || undefined,
      is_default: rigs.length === 0,
    });
    setRigName('');
    await refresh();
  }

  async function addSite(e: React.FormEvent) {
    e.preventDefault();
    if (!siteName.trim()) return;
    await api.saveSite({
      name: siteName.trim(),
      lat: Number(lat),
      lon: Number(lon),
      bortle_class: Number(bortle) || undefined,
      is_default: sites.length === 0,
    });
    setSiteName('');
    await refresh();
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <div className="flex items-center gap-2 text-white">
          <Telescope className="h-5 w-5 text-sky-400" />
          <h2 className="font-semibold">Optical Rigs</h2>
        </div>
        <form onSubmit={(e) => void addRig(e)} className="grid grid-cols-3 gap-2">
          <input
            value={rigName}
            onChange={(e) => setRigName(e.target.value)}
            placeholder="Rig name (RedCat 51…)"
            className="col-span-3 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
            required
          />
          <input
            value={focal}
            onChange={(e) => setFocal(e.target.value)}
            placeholder="Focal (mm)"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          />
          <input
            value={aperture}
            onChange={(e) => setAperture(e.target.value)}
            placeholder="Aperture (mm)"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          />
          <button
            type="submit"
            className="rounded-lg bg-sky-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-sky-400"
          >
            Save Rig
          </button>
        </form>
        <div className="space-y-2">
          {rigs.map((r) => {
            const fovDeg = ((2 * Math.atan(r.sensor_width_mm / (2 * r.focal_length_mm)) * 180) / Math.PI).toFixed(2);
            return (
              <div
                key={r.id}
                className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-2.5 text-sm"
              >
                <div>
                  <div className="font-medium text-white">{r.name}</div>
                  <div className="font-mono text-xs text-slate-400">
                    {r.focal_length_mm}mm · {r.sensor_format} · FOV ~{fovDeg}°
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void api.deleteRig(r.id).then(refresh)}
                  className="text-slate-500 hover:text-rose-400"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            );
          })}
        </div>
      </section>

      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <div className="flex items-center gap-2 text-white">
          <MapPin className="h-5 w-5 text-emerald-400" />
          <h2 className="font-semibold">Favorite Dark-Sky Sites</h2>
        </div>
        <form onSubmit={(e) => void addSite(e)} className="grid grid-cols-4 gap-2">
          <input
            value={siteName}
            onChange={(e) => setSiteName(e.target.value)}
            placeholder="Site name"
            className="col-span-4 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
            required
          />
          <input
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            placeholder="Lat"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          />
          <input
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            placeholder="Lon"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          />
          <input
            value={bortle}
            onChange={(e) => setBortle(e.target.value)}
            placeholder="Bortle 1-9"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white"
          />
          <button
            type="submit"
            className="rounded-lg bg-emerald-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-emerald-400"
          >
            Save
          </button>
        </form>
        <div className="space-y-2">
          {sites.map((s) => (
            <div
              key={s.id}
              className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-2.5 text-sm"
            >
              <div>
                <div className="font-medium text-white">{s.name}</div>
                <div className="font-mono text-xs text-slate-400">
                  {s.lat.toFixed(4)}°, {s.lon.toFixed(4)}° {s.bortle_class ? `· Bortle ${s.bortle_class}` : ''}
                </div>
              </div>
              <button
                type="button"
                onClick={() => void api.deleteSite(s.id).then(refresh)}
                className="text-slate-500 hover:text-rose-400"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
