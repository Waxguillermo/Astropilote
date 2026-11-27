import { useCallback, useEffect, useState } from 'react';
import {
  Award,
  BookOpen,
  Camera,
  MessageSquare,
  Settings,
  Telescope,
} from 'lucide-react';
import { api, AuthMeResponse } from './api/client';
import { CatalogTrackerView } from './components/CatalogTrackerView';
import { ChatView } from './components/ChatView';
import { LogbookView } from './components/LogbookView';
import { RigsSitesView } from './components/RigsSitesView';
import { SettingsModal } from './components/SettingsModal';
import { VisionView } from './components/VisionView';

type Tab = 'chat' | 'logbook' | 'rigs' | 'tracker' | 'vision';

export function App() {
  const [tab, setTab] = useState<Tab>('chat');
  const [llmMode, setLlmMode] = useState<'local' | 'api'>('local');
  const [me, setMe] = useState<AuthMeResponse | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const loadMe = useCallback(async () => {
    try {
      setMe(await api.authMe());
    } catch {
      setMe(null);
    }
  }, []);

  useEffect(() => {
    void loadMe();
  }, [loadMe]);

  const navItems: Array<{ id: Tab; label: string; icon: typeof MessageSquare }> = [
    { id: 'chat', label: 'Sky Chat', icon: MessageSquare },
    { id: 'logbook', label: 'Logbook', icon: BookOpen },
    { id: 'rigs', label: 'Rigs & Sites', icon: Telescope },
    { id: 'tracker', label: 'Messier / Caldwell', icon: Award },
    { id: 'vision', label: 'Sky Vision', icon: Camera },
  ];

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-4 sm:px-6">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800/80 bg-slate-900/70 px-5 py-3 backdrop-blur">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold tracking-tight text-white">
            Astropilote <span className="font-mono text-xs font-normal text-sky-400">v1.2</span>
          </span>
          <span className="hidden rounded-full bg-slate-800 px-2.5 py-0.5 text-xs text-slate-300 sm:inline">
            {me?.auth_mode === 'cloud' ? 'Cloud BYOK' : 'Local Solo'}
          </span>
        </div>

        <nav className="flex flex-wrap items-center gap-1">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-sm transition ${
                tab === id
                  ? 'bg-sky-500/20 font-medium text-sky-300 ring-1 ring-sky-500/40'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100'
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <div className="flex rounded-xl border border-slate-800 bg-slate-950 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setLlmMode('local')}
              className={`rounded-lg px-2.5 py-1 ${llmMode === 'local' ? 'bg-slate-800 text-white' : 'text-slate-400'}`}
            >
              Local ($0)
            </button>
            <button
              type="button"
              onClick={() => setLlmMode('api')}
              className={`rounded-lg px-2.5 py-1 ${llmMode === 'api' ? 'bg-sky-500 text-slate-950 font-medium' : 'text-slate-400'}`}
            >
              Cloud API
            </button>
          </div>
          <button
            type="button"
            onClick={() => setShowSettings(true)}
            className="rounded-xl border border-slate-800 bg-slate-950 p-2 text-slate-300 hover:border-sky-500/50 hover:text-white"
            title="Account & BYOK Keys"
          >
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </header>

      <main className="flex-1">
        {tab === 'chat' && <ChatView llmMode={llmMode} />}
        {tab === 'logbook' && <LogbookView />}
        {tab === 'rigs' && <RigsSitesView />}
        {tab === 'tracker' && <CatalogTrackerView />}
        {tab === 'vision' && <VisionView solvers={me?.vision_solvers} />}
      </main>

      {showSettings && (
        <SettingsModal
          me={me}
          onClose={() => setShowSettings(false)}
          onUpdated={loadMe}
        />
      )}
    </div>
  );
}
