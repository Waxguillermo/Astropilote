import { useState } from 'react';
import { KeyRound, ShieldCheck, X } from 'lucide-react';
import { api, AuthMeResponse } from '../api/client';

export function SettingsModal({
  me,
  onClose,
  onUpdated,
}: {
  me: AuthMeResponse | null;
  onClose: () => void;
  onUpdated: () => Promise<void>;
}) {
  const [provider, setProvider] = useState<'anthropic' | 'openrouter' | 'astrometry'>('anthropic');
  const [apiKey, setApiKey] = useState('');
  const [email, setEmail] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function saveKey(e: React.FormEvent) {
    e.preventDefault();
    if (!apiKey.trim()) return;
    setError(null);
    setNotice(null);
    try {
      const res = await api.saveApiKey(provider, apiKey.trim());
      setApiKey('');
      setNotice(`Saved encrypted ${res.provider} key (${res.key_hint}).`);
      await onUpdated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save API key');
    }
  }

  async function sendMagicLink(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim()) return;
    setError(null);
    setNotice(null);
    try {
      const res = await api.requestMagicLink(email.trim());
      if (res.dev_token) {
        await api.verifyMagicLink(res.dev_token);
        setNotice('Signed in via local Magic Link preview.');
        await onUpdated();
      } else {
        setNotice(res.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send Magic Link');
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg space-y-5 rounded-2xl border border-slate-800 bg-slate-950 p-6 shadow-2xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 font-semibold text-white">
            <KeyRound className="h-5 w-5 text-sky-400" /> Account &amp; Encrypted BYOK Vault
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-white">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex items-center justify-between rounded-xl bg-slate-900 px-4 py-3 text-xs text-slate-300">
          <span className="inline-flex items-center gap-1.5">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            Mode: <strong className="text-white">{me?.auth_mode ?? 'cloud'}</strong>
            {me?.user ? ` · ${me.user.display_name}` : ' · Signed out'}
          </span>
          {me?.auth_mode === 'cloud' && me.user && (
            <button
              type="button"
              onClick={() => void api.logout().then(onUpdated)}
              className="text-rose-300 hover:underline"
            >
              Sign out
            </button>
          )}
        </div>

        {me === null && (
          <form onSubmit={(e) => void sendMagicLink(e)} className="flex gap-2">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Sign in via Magic Link (email)"
              className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white"
              required
            />
            <button
              type="submit"
              className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-slate-950"
            >
              Sign in
            </button>
          </form>
        )}

        {me !== null && (
          <>
            <form onSubmit={(e) => void saveKey(e)} className="space-y-3">
              <div className="text-xs text-slate-400">
                Keys are encrypted at rest with Fernet AES+HMAC and scoped strictly to your account.
              </div>
              <div className="flex gap-2">
                <select
                  value={provider}
                  onChange={(e) =>
                    setProvider(e.target.value as 'anthropic' | 'openrouter' | 'astrometry')
                  }
                  className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white"
                >
                  <option value="anthropic">Anthropic</option>
                  <option value="openrouter">OpenRouter</option>
                  <option value="astrometry">nova.astrometry.net</option>
                </select>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Paste API key…"
                  className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white"
                />
                <button
                  type="submit"
                  className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-slate-950"
                >
                  Save
                </button>
              </div>
            </form>

            <div className="space-y-1.5">
              {me.api_keys.map((k) => (
                <div
                  key={k.provider}
                  className="flex items-center justify-between rounded-lg bg-slate-900 px-3 py-2 text-xs"
                >
                  <span className="font-mono text-slate-200">
                    {k.provider}: {k.key_hint}
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      void api
                        .deleteApiKey(k.provider as 'anthropic' | 'openrouter' | 'astrometry')
                        .then(onUpdated)
                    }
                    className="text-rose-300 hover:underline"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

        {error && <div className="text-xs text-rose-300">{error}</div>}
        {notice && <div className="text-xs text-emerald-300">{notice}</div>}
      </div>
    </div>
  );
}
