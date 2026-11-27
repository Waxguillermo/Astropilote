export interface UserProfile {
  id: string;
  email: string | null;
  display_name: string;
  provider: string;
}

export interface AuthMeResponse {
  auth_mode: 'local' | 'cloud';
  user: UserProfile;
  api_keys: Array<{ provider: string; key_hint: string }>;
  vision_solvers: {
    astap: boolean;
    solve_field: boolean;
    nova_astrometry_net: boolean;
  };
}

export interface Rig {
  id: string;
  name: string;
  focal_length_mm: number;
  aperture_mm: number | null;
  sensor_format: string;
  sensor_width_mm: number;
  sensor_height_mm: number;
  mount_type: string;
  is_default: boolean;
  notes: string;
}

export interface Site {
  id: string;
  name: string;
  lat: number;
  lon: number;
  bortle_class: number | null;
  elevation_m: number | null;
  is_default: boolean;
  notes: string;
}

export interface SessionTarget {
  id: string;
  object_id: string;
  object_name: string;
  ra_deg: number | null;
  dec_deg: number | null;
  exposure_minutes: number;
  status: 'planned' | 'observed' | 'imaged';
  filter_name: string | null;
  notes: string;
}

export interface ObservingSession {
  id: string;
  title: string;
  obs_date: string;
  status: 'planned' | 'completed' | 'cancelled';
  site_id: string | null;
  site_name: string;
  rig_id: string | null;
  rig_name: string;
  cloud_cover_pct: number | null;
  moon_illumination_pct: number | null;
  notes: string;
  targets: SessionTarget[];
}

export interface CatalogProgressSummary {
  messier: { completed: number; total: number; pct: number };
  caldwell: { completed: number; total: number; pct: number };
  imaged_total: number;
  total_exposure_minutes: number;
  items: Array<{
    object_id: string;
    catalog: string;
    status: 'wishlist' | 'observed' | 'imaged';
    first_observed_at: string | null;
    total_exposure_minutes: number;
    notes: string;
  }>;
}

export interface VisionPreset {
  id: string;
  label: string;
  subtitle: string;
  badge: string;
}

export interface VisionResult {
  result: {
    astrometry: {
      method: string;
      centre_icrs_deg: { ra: number; dec: number };
    };
    object_count: number;
    objects: Array<{
      doc_id: string;
      title: string;
      metadata: {
        ra_deg?: number;
        dec_deg?: number;
        magnitude?: number;
        object_type?: string;
        constellation?: string;
      };
    }>;
    warnings: string[];
  };
  annotated_image_data_url?: string;
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err?.detail) detail = String(err.detail);
    } catch {
      /* ignore non-JSON body */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => apiJson<Record<string, unknown>>('/api/health'),
  authMe: () => apiJson<AuthMeResponse>('/api/auth/me'),
  requestMagicLink: (email: string) =>
    apiJson<{ status: string; message: string; dev_token?: string; dev_verify_url?: string }>(
      '/api/auth/magic-link',
      { method: 'POST', body: JSON.stringify({ email }) }
    ),
  verifyMagicLink: (token: string) =>
    apiJson<{ status: string }>(`/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`),
  logout: () => apiJson<{ status: string }>('/api/auth/logout', { method: 'POST' }),
  saveApiKey: (provider: 'anthropic' | 'openrouter' | 'astrometry', api_key: string) =>
    apiJson<{ status: string; provider: string; key_hint: string }>(`/api/auth/keys/${provider}`, {
      method: 'PUT',
      body: JSON.stringify({ api_key }),
    }),
  deleteApiKey: (provider: 'anthropic' | 'openrouter' | 'astrometry') =>
    apiJson<{ status: string }>(`/api/auth/keys/${provider}`, { method: 'DELETE' }),

  listRigs: () => apiJson<{ rigs: Rig[] }>('/api/logbook/rigs'),
  saveRig: (payload: Partial<Rig> & { name: string }) =>
    apiJson<{ rig: Rig }>('/api/logbook/rigs', { method: 'POST', body: JSON.stringify(payload) }),
  deleteRig: (id: string) =>
    apiJson<{ deleted: boolean }>(`/api/logbook/rigs/${id}`, { method: 'DELETE' }),

  listSites: () => apiJson<{ sites: Site[] }>('/api/logbook/sites'),
  saveSite: (payload: Partial<Site> & { name: string; lat: number; lon: number }) =>
    apiJson<{ site: Site }>('/api/logbook/sites', { method: 'POST', body: JSON.stringify(payload) }),
  deleteSite: (id: string) =>
    apiJson<{ deleted: boolean }>(`/api/logbook/sites/${id}`, { method: 'DELETE' }),

  listSessions: () => apiJson<{ sessions: ObservingSession[] }>('/api/logbook/sessions'),
  createSession: (payload: {
    title: string;
    obs_date: string;
    status?: 'planned' | 'completed' | 'cancelled';
    site_name?: string;
    rig_name?: string;
    notes?: string;
  }) =>
    apiJson<{ session: ObservingSession }>('/api/logbook/sessions', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateSession: (id: string, payload: Partial<ObservingSession>) =>
    apiJson<{ session: ObservingSession }>(`/api/logbook/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteSession: (id: string) =>
    apiJson<{ deleted: boolean }>(`/api/logbook/sessions/${id}`, { method: 'DELETE' }),
  addSessionTarget: (
    sessionId: string,
    payload: { object_id: string; object_name?: string; exposure_minutes?: number; status?: 'planned' | 'observed' | 'imaged' }
  ) =>
    apiJson<{ target: SessionTarget }>(`/api/logbook/sessions/${sessionId}/targets`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getProgress: () => apiJson<CatalogProgressSummary>('/api/logbook/progress'),
  upsertProgress: (payload: {
    object_id: string;
    status: 'wishlist' | 'observed' | 'imaged';
    add_exposure_minutes?: number;
  }) =>
    apiJson<{ summary: CatalogProgressSummary }>('/api/logbook/progress', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listThreads: () =>
    apiJson<{ threads: Array<{ id: string; title: string; updated_at: string }> }>('/api/logbook/threads'),
  getThread: (id: string) =>
    apiJson<{
      thread: {
        id: string;
        title: string;
        messages: Array<{ id: string; role: 'user' | 'assistant'; content: string }>;
      };
    }>(`/api/logbook/threads/${id}`),

  chat: (payload: {
    question: string;
    llm_mode?: 'local' | 'api';
    thread_id?: string;
    history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  }) =>
    apiJson<{
      answer: string;
      verified?: boolean;
      citations?: Array<{ doc_id: string; title: string }>;
      tool_calls?: Array<{ name: string; status: string }>;
      thread_id?: string;
    }>('/api/chat', { method: 'POST', body: JSON.stringify(payload) }),

  visionPresets: () => apiJson<{ presets: VisionPreset[] }>('/api/vision/presets'),
  visionIdentify: (payload: Record<string, unknown>) =>
    apiJson<VisionResult>('/api/vision/identify', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  logVisionTargets: (payload: {
    title: string;
    obs_date: string;
    targets: Array<{ object_id: string; object_name?: string; ra_deg?: number; dec_deg?: number }>;
  }) =>
    apiJson<{ session: ObservingSession; progress: CatalogProgressSummary }>(
      '/api/logbook/vision-log',
      { method: 'POST', body: JSON.stringify(payload) }
    ),
};
