import { useState } from 'react';
import { Send, Sparkles, CheckCircle2, Wrench } from 'lucide-react';
import { api } from '../api/client';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  verified?: boolean;
  tools?: Array<{ name: string; status: string }>;
  citations?: Array<{ doc_id: string; title: string }>;
}

const STARTERS = [
  'Compare observing conditions tonight between Paris and Fontainebleau',
  'What bright Messier targets are observable tonight above 25° from Lyon?',
  'Best sub-exposure and framing for Andromeda (M31) with a 250mm APO on APS-C',
];

export function ChatView({ llmMode }: { llmMode: 'local' | 'api' }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [threadId, setThreadId] = useState<string>('');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(promptText?: string) {
    const q = (promptText ?? input).trim();
    if (!q || loading) return;
    setError(null);
    setInput('');
    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setLoading(true);
    try {
      const res = await api.chat({
        question: q,
        llm_mode: llmMode,
        thread_id: threadId,
        history,
      });
      if (res.thread_id) setThreadId(res.thread_id);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.answer,
          verified: res.verified,
          tools: res.tool_calls,
          citations: res.citations,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex-1 space-y-4 overflow-y-auto rounded-2xl border border-slate-800/80 bg-slate-900/50 p-5">
        {messages.length === 0 ? (
          <div className="mx-auto max-w-2xl py-10 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-500/15 text-sky-400">
              <Sparkles className="h-6 w-6" />
            </div>
            <h2 className="text-xl font-semibold text-white">Sky Copilot &amp; Session Planner</h2>
            <p className="mt-1 text-sm text-slate-400">
              Real-time Skyfield ephemerides, Open-Meteo cloud cover, ONNX/BM25 catalog search, and optics framing.
            </p>
            <div className="mt-6 grid gap-2 text-left">
              {STARTERS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => void send(s)}
                  className="rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-200 transition hover:border-sky-500/50 hover:bg-slate-900"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, idx) => (
            <div
              key={idx}
              className={`rounded-2xl p-4 ${
                m.role === 'user'
                  ? 'ml-auto max-w-2xl bg-sky-600/20 text-sky-50 ring-1 ring-sky-500/30'
                  : 'max-w-3xl bg-slate-950/80 text-slate-100 ring-1 ring-slate-800'
              }`}
            >
              <div className="whitespace-pre-wrap text-sm leading-relaxed">{m.content}</div>
              {m.role === 'assistant' && (
                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                  {m.verified && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-emerald-300">
                      <CheckCircle2 className="h-3.5 w-3.5" /> Verified
                    </span>
                  )}
                  {m.tools?.map((t, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center gap-1 rounded-full bg-slate-800 px-2.5 py-0.5 font-mono text-slate-300"
                    >
                      <Wrench className="h-3 w-3" /> {t.name}
                    </span>
                  ))}
                  {m.citations?.slice(0, 5).map((c) => (
                    <span
                      key={c.doc_id}
                      className="rounded-full bg-sky-500/10 px-2 py-0.5 font-mono text-sky-300"
                    >
                      [{c.doc_id}]
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
        {loading && (
          <div className="max-w-md animate-pulse rounded-xl bg-slate-900 px-4 py-3 text-sm text-sky-300">
            Computing ephemerides &amp; weather…
          </div>
        )}
        {error && (
          <div className="rounded-xl border border-rose-500/40 bg-rose-950/50 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about tonight's sky, target framing, weather, or calibration…"
          className="flex-1 rounded-xl border border-slate-800 bg-slate-900 px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-sky-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="inline-flex items-center gap-2 rounded-xl bg-sky-500 px-5 py-3 text-sm font-medium text-slate-950 transition hover:bg-sky-400 disabled:opacity-40"
        >
          <Send className="h-4 w-4" /> Send
        </button>
      </form>
    </div>
  );
}
