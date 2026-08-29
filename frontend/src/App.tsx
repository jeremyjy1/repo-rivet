import { memo, useCallback, useEffect, useRef, useState } from "react";
import {
  Bot,
  CheckCircle2,
  ChevronRight,
  CircleStop,
  FileCode2,
  Files,
  Folder,
  GitBranch,
  Hammer,
  ListChecks,
  LoaderCircle,
  MessageSquareText,
  Moon,
  Play,
  Plus,
  RefreshCw,
  Send,
  Settings2,
  ShieldAlert,
  Sparkles,
  Sun,
  XCircle,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AgentEvent, Approval, EventPage, SessionDetail, SessionSummary, api, authenticate } from "./api";

type Bootstrap = {
  workspace: string;
  active_session_id: string | null;
  sessions: SessionSummary[];
  settings: { model: string; base_url: string; context_limit: number; approval_mode: string; auto_plan: "off" | "adaptive" | "always" };
};
type FileEntry = { name: string; path: string; kind: "file" | "directory"; size: number | null };
type Skill = { id: string; name: string; description: string; version: string; source: string };
type Theme = "dark" | "light";

function initialTheme(): Theme {
  const stored = localStorage.getItem("reporivet-theme");
  const theme = stored === "dark" || stored === "light"
    ? stored
    : window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  return theme;
}

const eventLabels: Record<string, string> = {
  "tool.requested": "Tool requested",
  "tool.finished": "Tool finished",
  "run.started": "Run started",
  "run.finished": "Run finished",
  "web.run.finished": "Run finished",
};

const sessionRefreshEvents = new Set([
  "approval.requested",
  "approval.awaiting.human",
  "approval.resolved",
  "external.files.changed",
  "plan.approved",
  "plan.cancelled",
  "plan.submitted",
  "plan.updated",
  "auto.plan.started",
  "run.finished",
  "session.start",
  "verification.result",
  "web.run.finished",
]);

const hiddenTimelineEvents = new Set([
  "model.call.finished",
  "model.stream.progress",
  "run.finished",
  "web.run.finished",
]);

function shouldRefreshSession(eventType: string) {
  return sessionRefreshEvents.has(eventType);
}

function mergeEvents(left: AgentEvent[], right: AgentEvent[]) {
  if (left.length === 0) return right;
  if (right.length === 0) return left;
  if (left.at(-1)!.seq < right[0].seq) return [...left, ...right];
  const merged = new Map<number, AgentEvent>();
  for (const event of left) merged.set(event.seq, event);
  for (const event of right) merged.set(event.seq, event);
  return [...merged.values()].sort((first, second) => first.seq - second.seq);
}

function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [currentDir, setCurrentDir] = useState(".");
  const [fileView, setFileView] = useState<{ path: string; content: string; start_line: number; snapshot_tag: string } | null>(null);
  const [diffView, setDiffView] = useState("");
  const [skills, setSkills] = useState<Skill[]>([]);
  const [selectedSkill, setSelectedSkill] = useState("");
  const [mode, setMode] = useState<"execute" | "planning">("execute");
  const [approvalMode, setApprovalMode] = useState("safe-auto");
  const [autoPlan, setAutoPlan] = useState<"off" | "adaptive" | "always">("adaptive");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const refreshSession = useCallback(async (sessionId?: string | null) => {
    if (!sessionId) return;
    const detail = await api<SessionDetail>(`/api/v1/sessions/${sessionId}`);
    setSession(detail);
    if (detail.approval_mode) setApprovalMode(detail.approval_mode);
  }, []);

  const initialize = useCallback(async () => {
    setLoading(true);
    try {
      await authenticate();
      const data = await api<Bootstrap>("/api/v1/bootstrap");
      setBoot(data);
      setApprovalMode(data.settings.approval_mode);
      setAutoPlan(data.settings.auto_plan);
      await refreshSession(data.active_session_id);
      void Promise.allSettled([
        api<FileEntry[]>("/api/v1/workspace/files").then(setFiles),
        api<Skill[]>("/api/v1/skills").then(setSkills),
        api<{ diff: string }>("/api/v1/workspace/diff").then((value) => setDiffView(value.diff)),
      ]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, [refreshSession]);

  useEffect(() => void initialize(), [initialize]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("reporivet-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (session?.workflow_mode === "planning" || session?.workflow_mode === "plan_ready") {
      setMode("planning");
    } else if (session?.workflow_mode === "execute") {
      setMode("execute");
    }
  }, [session?.workflow_mode]);

  const invoke = useCallback(async (work: () => Promise<unknown>) => {
    setError("");
    try { await work(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, []);

  const createSession = () => invoke(async () => {
    const created = await api<SessionSummary>("/api/v1/sessions", { method: "POST", body: JSON.stringify({}) });
    const data = await api<Bootstrap>("/api/v1/bootstrap");
    setBoot(data);
    await refreshSession(created.session_id);
  });

  const selectSession = (id: string) => invoke(async () => {
    await api(`/api/v1/sessions/${id}/use`, { method: "POST", body: "{}" });
    await refreshSession(id);
  });

  const submit = (task: string) => invoke(async () => {
    if (!session) return;
    await api(`/api/v1/sessions/${session.session_id}/runs`, {
      method: "POST",
      body: JSON.stringify({ task, mode, approval_mode: approvalMode, auto_plan: autoPlan, skill: selectedSkill || null }),
    });
    await refreshSession(session.session_id);
  });

  const openDirectory = (path: string) => invoke(async () => {
    setCurrentDir(path);
    setFiles(await api<FileEntry[]>(`/api/v1/workspace/files?path=${encodeURIComponent(path)}`));
  });

  const openFile = (path: string) => invoke(async () => {
    setFileView(await api(`/api/v1/workspace/file?path=${encodeURIComponent(path)}`));
  });

  const approval = session?.run?.pending_approval || null;
  const isRunning = ["queued", "running", "awaiting_approval", "stopping"].includes(session?.run?.status || "");

  if (loading) return <div className="splash"><LoaderCircle className="spin" /> Loading RepoRivet…</div>;
  if (!boot) return <div className="splash error"><XCircle /> {error || "Authentication required. Open the one-use URL printed by the CLI."}</div>;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><GitBranch size={18} /></span><strong>RepoRivet</strong><span className="local-pill">LOCAL</span></div>
        <div className="workspace"><Folder size={15} /> <span>{boot.workspace}</span></div>
        <div className="topbar-actions">
          <div className="model-pill"><Sparkles size={14} /> <span>{boot.settings.model}</span></div>
          <button
            className="theme-toggle"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            aria-pressed={theme === "light"}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
          >{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</button>
        </div>
      </header>

      <aside className="sidebar">
        <section className="section-head"><span>SESSIONS</span><button className="icon-button" onClick={createSession} title="New session"><Plus size={16} /></button></section>
        <div className="session-list">
          {boot.sessions.map((item) => <button key={item.session_id} className={`session-item ${session?.session_id === item.session_id ? "active" : ""}`} onClick={() => selectSession(item.session_id)}>
            <MessageSquareText size={16} /><span><strong>{item.name}</strong><small>{item.task_preview || item.short_id}</small></span><i className={`status-dot ${item.status}`} />
          </button>)}
        </div>
        <section className="section-head file-heading"><span>WORKSPACE</span><button className="icon-button" onClick={() => openDirectory(currentDir)}><RefreshCw size={14} /></button></section>
        <button className="breadcrumb" onClick={() => openDirectory(currentDir.includes("/") ? currentDir.slice(0, currentDir.lastIndexOf("/")) || "." : ".")}>‹ {currentDir}</button>
        <div className="file-list">
          {files.map((item) => <button key={item.path} onClick={() => item.kind === "directory" ? openDirectory(item.path) : openFile(item.path)}>
            {item.kind === "directory" ? <Folder size={15} /> : <FileCode2 size={15} />}<span>{item.name}</span>{item.kind === "directory" && <ChevronRight size={13} />}
          </button>)}
        </div>
      </aside>

      <main className="main-pane">
        <div className="runbar">
          <div><Bot size={18} /><strong>{session?.name || "No session selected"}</strong>{session && <span className="revision">rev {session.workspace_revision}</span>}</div>
          <div className="run-actions">
            <button className={`mode-button ${mode === "planning" ? "plan" : ""}`} disabled={session?.workflow_mode === "plan_ready"} title={session?.workflow_mode === "plan_ready" ? "Review the plan in the Plan panel before execution" : "Switch the workflow for the next request"} onClick={() => setMode(mode === "execute" ? "planning" : "execute")}>{mode === "planning" ? <ListChecks size={15} /> : <Hammer size={15} />}{session?.workflow_mode === "plan_ready" ? "Plan ready" : mode === "planning" ? "Plan" : "Execute"}</button>
          </div>
        </div>
        {error && <div className="error-banner"><ShieldAlert size={16} />{error}<button onClick={() => setError("")}>×</button></div>}
        <Timeline session={session} isRunning={isRunning} pendingApprovalId={approval?.request_id || null} invoke={invoke} onRefresh={refreshSession} />
        <Composer
          disabled={!session}
          isRunning={isRunning}
          mode={mode}
          setMode={setMode}
          planReady={session?.workflow_mode === "plan_ready"}
          autoPlan={autoPlan}
          setAutoPlan={setAutoPlan}
          approvalMode={approvalMode}
          setApprovalMode={(value) => {
            setApprovalMode(value);
            if (session) void invoke(() => api(`/api/v1/sessions/${session.session_id}/approval-mode`, { method: "PUT", body: JSON.stringify({ mode: value }) }));
          }}
          skills={skills}
          selectedSkill={selectedSkill}
          setSelectedSkill={setSelectedSkill}
          settings={boot.settings}
          onStop={() => invoke(() => api(`/api/v1/sessions/${session!.session_id}/stop`, { method: "POST", body: "{}" }))}
          onSubmit={submit}
        />
      </main>

      <aside className="inspector">
        <Inspector session={session} fileView={fileView} diffView={diffView} refreshDiff={() => invoke(async () => setDiffView((await api<{ diff: string }>("/api/v1/workspace/diff")).diff))} mode={mode} invoke={invoke} refresh={() => refreshSession(session?.session_id)} />
      </aside>
      {approval && <ApprovalDialog approval={approval} onDecision={(action, guidance) => invoke(async () => {
        await api(`/api/v1/sessions/${session!.session_id}/approvals/decision`, { method: "POST", body: JSON.stringify({ request_id: approval.request_id, state_version: approval.state_version, action, guidance }) });
        await refreshSession(session!.session_id);
      })} />}
    </div>
  );
}

const Timeline = memo(function Timeline({ session, isRunning, pendingApprovalId, invoke, onRefresh }: {
  session: SessionDetail | null;
  isRunning: boolean;
  pendingApprovalId: string | null;
  invoke: (work: () => Promise<unknown>) => Promise<void>;
  onRefresh: (sessionId?: string | null) => Promise<void>;
}) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const timelineContentRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const programmaticScroll = useRef(false);
  const pointerScrolling = useRef(false);
  const previousRunning = useRef(false);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [historyLimit, setHistoryLimit] = useState(120);
  const [hasOlderEvents, setHasOlderEvents] = useState(false);
  const [loadingOlderEvents, setLoadingOlderEvents] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const messages = session?.messages || [];
  const fallbackMessages = historyLoaded && events.length === 0 ? messages : [];
  const resolvedApprovalIds = new Set(events.filter((event) => event.type === "approval.resolved").map((event) => textValue(event.payload.request_id)));
  const visibleEvents = events.filter((event) => {
    if (hiddenTimelineEvents.has(event.type)) return false;
    return event.type !== "approval.requested" || textValue(event.payload.request_id) === pendingApprovalId || !resolvedApprovalIds.has(textValue(event.payload.request_id));
  });
  const messageStart = Math.max(0, fallbackMessages.length - historyLimit);
  const hasEarlierHistory = messageStart > 0 || hasOlderEvents;
  const latestModelActivity = [...events].reverse().find((event) => [
    "model.call",
    "model.response",
    "model.response.invalid",
    "model.error",
    "model.stream.progress",
    "model.stream.retry",
    "approval.review.started",
    "approval.review.completed",
    "approval.review.failed",
  ].includes(event.type));
  let workingLabel = "Model is working";
  if (latestModelActivity?.type === "model.stream.progress") {
    const chunks = numberValue(latestModelActivity.payload.chunk_count);
    const toolChars = numberValue(latestModelActivity.payload.tool_argument_chars);
    workingLabel = latestModelActivity.payload.completed === true
      ? "Finishing streamed model response"
      : `Receiving model response${chunks !== null ? ` · ${chunks.toLocaleString()} chunks` : ""}${toolChars !== null && toolChars > 0 ? ` · ${toolChars.toLocaleString()} tool characters` : ""}`;
  } else if (latestModelActivity?.type === "model.stream.retry") {
    workingLabel = `Retrying model stream · attempt ${textValue(latestModelActivity.payload.attempt)}/${textValue(latestModelActivity.payload.max_attempts)}`;
  } else if (latestModelActivity?.type === "approval.review.started") {
    workingLabel = `Approval model is reviewing ${humanize(textValue(latestModelActivity.payload.tool) || "the tool request")}`;
  } else if (latestModelActivity?.type === "approval.review.completed") {
    workingLabel = "Approval review completed";
  } else if (latestModelActivity?.type === "approval.review.failed") {
    workingLabel = "Approval review unavailable · preparing human approval";
  }

  useEffect(() => {
    if (!session) return;
    let active = true;
    setEvents([]);
    setHasOlderEvents(false);
    setHistoryLoaded(false);
    const sessionId = session.session_id;
    const snapshotSeq = session.last_event_seq;
    void api<EventPage>(`/api/v1/sessions/${sessionId}/events/history?before=${snapshotSeq + 1}&limit=240`).then((page) => {
      if (!active) return;
      setEvents((current) => mergeEvents(page.items, current));
      setHasOlderEvents(page.has_more);
    }).finally(() => {
      if (active) setHistoryLoaded(true);
    });
    const source = new EventSource(`/api/v1/sessions/${sessionId}/events?after=${snapshotSeq}`);
    let refreshTimer: number | undefined;
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as AgentEvent;
      const timeline = timelineRef.current;
      if (timeline && timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 160) {
        stickToBottom.current = true;
      }
      if (event.type === "session.start") stickToBottom.current = true;
      setEvents((current) => mergeEvents(current, [event]));
      if (shouldRefreshSession(event.type)) {
        window.clearTimeout(refreshTimer);
        refreshTimer = window.setTimeout(() => void onRefresh(sessionId), 80);
      }
    };
    return () => {
      active = false;
      window.clearTimeout(refreshTimer);
      source.close();
    };
  }, [session?.session_id, onRefresh]);

  useEffect(() => {
    setHistoryLimit(120);
    stickToBottom.current = true;
  }, [session?.session_id]);

  useEffect(() => {
    if (isRunning && !previousRunning.current) stickToBottom.current = true;
    previousRunning.current = isRunning;
  }, [isRunning]);

  const scrollToBottom = useCallback(() => {
    const timeline = timelineRef.current;
    if (!timeline || !stickToBottom.current) return;
    programmaticScroll.current = true;
    timeline.scrollTop = timeline.scrollHeight;
    requestAnimationFrame(() => {
      programmaticScroll.current = false;
    });
  }, []);

  useEffect(() => {
    const content = timelineContentRef.current;
    if (!content || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => scrollToBottom());
    observer.observe(content);
    return () => observer.disconnect();
  }, [scrollToBottom, session?.session_id]);

  useEffect(() => {
    if (!stickToBottom.current) return;
    let finalFrame = 0;
    const frame = requestAnimationFrame(() => {
      scrollToBottom();
      finalFrame = requestAnimationFrame(() => {
        scrollToBottom();
      });
    });
    return () => {
      cancelAnimationFrame(frame);
      cancelAnimationFrame(finalFrame);
    };
  }, [events.length, messages.length, isRunning, pendingApprovalId, session?.plan?.status, session?.run?.result, scrollToBottom]);

  return <div className="timeline" ref={timelineRef} onScroll={(event) => {
    const target = event.currentTarget;
    const nearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 80;
    if (nearBottom) stickToBottom.current = true;
    else if (!programmaticScroll.current && pointerScrolling.current) stickToBottom.current = false;
  }} onWheel={(event) => {
    if (event.deltaY < 0) stickToBottom.current = false;
  }} onPointerDown={() => { pointerScrolling.current = true; }} onPointerUp={() => { pointerScrolling.current = false; }} onPointerCancel={() => { pointerScrolling.current = false; }}>
    <div className="timeline-content" ref={timelineContentRef}>
    {hasEarlierHistory && <button className="load-history" disabled={loadingOlderEvents} onClick={() => {
      stickToBottom.current = false;
      setHistoryLimit((current) => current + 120);
      if (hasOlderEvents && session) {
        setLoadingOlderEvents(true);
        const before = events[0]?.seq || session.last_event_seq + 1;
        void api<EventPage>(`/api/v1/sessions/${session.session_id}/events/history?before=${before}&limit=240`).then((page) => {
          setEvents((current) => mergeEvents(page.items, current));
          setHasOlderEvents(page.has_more);
        }).finally(() => setLoadingOlderEvents(false));
      }
    }}>{loadingOlderEvents ? "Loading earlier history…" : "Show earlier history"}</button>}
    {!session && <EmptyState />}
    {fallbackMessages.slice(messageStart).map((message, index) => <MessageBlock message={message} key={`m-${messageStart + index}`} />)}
    {visibleEvents.map((event) => <EventCard event={event} pendingApprovalId={pendingApprovalId} key={event.event_id} />)}
    {isRunning && <div className="generating"><LoaderCircle className="spin" size={17} /><span>{pendingApprovalId ? "Waiting for your approval" : workingLabel}</span><i /><i /><i /></div>}
    {!isRunning && session?.plan && (session.plan.status === "ready" || session.plan.status === "stale") && <PlanResultCard session={session} invoke={invoke} refresh={() => onRefresh(session.session_id)} />}
    {session?.run?.result && !(session.run.result.status === "plan_ready" && session.plan && (session.plan.status === "ready" || session.plan.status === "stale")) && <ResultCard result={session.run.result} />}
    </div>
  </div>;
});

const MessageBlock = memo(function MessageBlock({ message }: { message: SessionDetail["messages"][number] }) {
  return <article className={`message ${message.role}`}>
    <div className="message-label">{message.role === "user" ? "YOU" : message.role.toUpperCase()}</div>
    <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{message.content}</ReactMarkdown>
  </article>;
}, (previous, next) => previous.message.role === next.message.role && previous.message.content === next.message.content && previous.message.step === next.message.step);

const Composer = memo(function Composer({ disabled, isRunning, mode, setMode, planReady, autoPlan, setAutoPlan, approvalMode, setApprovalMode, skills, selectedSkill, setSelectedSkill, settings, onStop, onSubmit }: {
  disabled: boolean;
  isRunning: boolean;
  mode: "execute" | "planning";
  setMode: (mode: "execute" | "planning") => void;
  planReady: boolean;
  autoPlan: "off" | "adaptive" | "always";
  setAutoPlan: (mode: "off" | "adaptive" | "always") => void;
  approvalMode: string;
  setApprovalMode: (mode: string) => void;
  skills: Skill[];
  selectedSkill: string;
  setSelectedSkill: (skill: string) => void;
  settings: Bootstrap["settings"];
  onStop: () => Promise<void> | void;
  onSubmit: (task: string) => Promise<void> | void;
}) {
  const [value, setValue] = useState("");
  const send = async () => {
    const task = value.trim();
    if (!task || disabled || isRunning) return;
    setValue("");
    await onSubmit(task);
  };
  return <div className="composer-wrap">
    <div className="composer-shell">
      <div className="composer-input">
        <textarea value={value} disabled={disabled || isRunning} onChange={(event) => setValue(event.target.value)} onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void send();
          }
        }} placeholder={mode === "planning" ? "Ask RepoRivet to inspect and prepare a plan…" : "Describe what you want to change…"} />
        <button className={`send-button ${isRunning ? "stop" : ""}`} aria-label={isRunning ? "Stop current run" : "Send prompt"} title={isRunning ? "Stop current run" : "Send prompt"} disabled={disabled || (!isRunning && !value.trim())} onClick={() => void (isRunning ? onStop() : send())}>{isRunning ? <CircleStop size={18} /> : <Send size={18} />}</button>
        <div className="composer-meta"><span>Enter to send · Shift+Enter for newline</span><span>{value.length.toLocaleString()} chars</span></div>
      </div>
      <div className="composer-toolbar">
        <div className="composer-controls">
          <span className="composer-settings-icon" title="Request settings"><Settings2 size={13} /></span>
          <label className={`composer-control ${mode === "planning" ? "plan" : ""}`} title="Workflow mode">{mode === "planning" ? <ListChecks size={13} /> : <Hammer size={13} />}<select aria-label="Workflow mode" value={mode} disabled={planReady} onChange={(event) => setMode(event.target.value as "execute" | "planning")}><option value="execute">Execute</option><option value="planning">{planReady ? "Plan ready" : "Plan"}</option></select></label>
          <label className="composer-control" title="Automatic planning"><RefreshCw size={13} /><select aria-label="Automatic planning" value={autoPlan} onChange={(event) => setAutoPlan(event.target.value as "off" | "adaptive" | "always")}><option value="adaptive">Adaptive plan</option><option value="always">Plan first</option><option value="off">No auto plan</option></select></label>
          <label className="composer-control" title="Approval mode"><ShieldAlert size={13} /><select aria-label="Approval mode" value={approvalMode} onChange={(event) => setApprovalMode(event.target.value)}><option value="safe-auto">Safe auto</option><option value="llm-auto">LLM auto</option><option value="always-ask">Always ask</option><option value="allow-all">Allow all</option></select></label>
          <label className="composer-control" title="Global Skill"><Sparkles size={13} /><select aria-label="Global Skill" value={selectedSkill} onChange={(event) => setSelectedSkill(event.target.value)}><option value="">No skill</option>{skills.filter((skill) => skill.source === "global").map((skill) => <option key={skill.id} value={skill.id}>{skill.name}</option>)}</select></label>
        </div>
        <div className="composer-model" title={`${settings.base_url} · ${settings.context_limit.toLocaleString()} token context`}><Bot size={13} /><span>{settings.model}</span><i>·</i><span>{settings.context_limit.toLocaleString()}</span></div>
      </div>
    </div>
  </div>;
});

function EmptyState() { return <div className="empty-state"><div className="empty-icon"><GitBranch size={30} /></div><h2>Build with evidence</h2><p>Create or select a session, then ask RepoRivet to inspect, plan, edit, and verify your workspace.</p></div>; }

type EventPresentation = { title: string; summary: string; detail: string };

function textValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function recordValue(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function clipped(value: string, maximum = 520): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > maximum ? `${compact.slice(0, maximum - 1)}…` : compact;
}

function describeToolRequest(tool: string, argumentsValue: unknown): string {
  const args = recordValue(argumentsValue);
  const path = textValue(args.path) || textValue(args.file_path);
  if (tool === "read_file") {
    const start = numberValue(args.start_line);
    const end = numberValue(args.end_line);
    const lines = start !== null ? ` · lines ${start}${end !== null ? `–${end}` : "–end"}` : "";
    return `${path || "Workspace file"}${lines}`;
  }
  if (tool === "search_text") {
    const query = textValue(args.query) || textValue(args.pattern);
    return `${query ? `“${clipped(query, 160)}”` : "Text search"} in ${path || "."}`;
  }
  if (tool === "list_files") {
    const depth = numberValue(args.max_depth);
    return `${path || "."}${depth !== null ? ` · depth ${depth}` : ""}`;
  }
  if (tool === "write_file") return `Create or replace ${path || "workspace file"}`;
  if (tool === "edit_file") {
    const operations = Array.isArray(args.operations) ? args.operations : [];
    const first = recordValue(operations[0]);
    const start = numberValue(first.start_line) ?? numberValue(first.line);
    const end = numberValue(first.end_line);
    const operation = humanize(textValue(first.op) || "edit");
    const location = start !== null ? ` · line${end !== null && end !== start ? "s" : ""} ${start}${end !== null && end !== start ? `–${end}` : ""}` : "";
    return `${path || "Workspace file"} · ${operation}${location} · ${operations.length || 1} operation${operations.length === 1 ? "" : "s"}`;
  }
  if (tool === "run_command") {
    const command = textValue(args.command) || stringList(args.argv).join(" ");
    const cwd = textValue(args.cwd) || ".";
    return `${clipped(command || "Command", 360)} · cwd ${cwd}`;
  }
  if (tool === "run_verification") return `Check ${textValue(args.check_id) || "registered verification"}`;
  if (tool === "git_diff") return `Inspect changes${path ? ` under ${path}` : ""}`;
  if (tool === "git_status") return "Inspect workspace status";
  if (tool === "record_decision") return clipped(textValue(args.summary) || "Record the next decision");
  if (tool === "register_verification") {
    const checks = Array.isArray(args.checks) ? args.checks : [];
    const checkNames = checks.map((check) => textValue(recordValue(check).check_id)).filter(Boolean);
    return checkNames.length ? `Register checks: ${checkNames.join(", ")}` : "Register deterministic verification checks";
  }
  if (tool === "submit_plan") {
    const plan = recordValue(args.plan);
    return clipped(textValue(plan.goal) || "Submit a plan for review");
  }
  if (tool === "update_plan") return clipped(textValue(args.reason) || "Update the active plan");
  if (tool === "request_plan") return clipped(textValue(args.reason) || "Enter read-only Plan Mode");
  if (path) return path;
  return "Agent operation";
}

function approvalTitle(action: string, source: string): string {
  if (action === "allow") {
    if (source === "web_human" || source === "human") return "Approved by you";
    if (source === "session_grant") return "Allowed by session rule";
    return "Automatically approved";
  }
  if (action === "deny") {
    if (source === "web_human" || source === "human") return "Denied by you";
    return "Denied by policy";
  }
  return "Approval evaluated";
}

function presentEvent(event: AgentEvent, pendingApprovalId: string | null): EventPresentation {
  const payload = event.payload;
  const tool = textValue(payload.name) || textValue(payload.tool) || textValue(payload.tool_name);
  const summary = textValue(payload.summary) || textValue(payload.result_summary);
  switch (event.type) {
    case "tool.requested":
      return { title: `Action · ${humanize(tool || "tool")}`, summary: describeToolRequest(tool, payload.arguments), detail: "" };
    case "observation": {
      const ok = payload.ok !== false;
      const details = [
        numberValue(payload.exit_code) !== null ? `exit ${numberValue(payload.exit_code)}` : "",
        stringList(payload.affected_paths).length ? `files: ${stringList(payload.affected_paths).join(", ")}` : "",
      ].filter(Boolean).join(" · ");
      return { title: `${ok ? "Observed" : "Failed"} · ${humanize(tool || "tool")}`, summary, detail: details };
    }
    case "tool.finished": {
      const ok = payload.ok !== false;
      return {
        title: `${ok ? "Completed" : "Failed"} · ${humanize(tool || "tool")}`,
        summary: textValue(payload.error) || (ok ? "Tool completed successfully" : "Tool operation failed"),
        detail: textValue(payload.error_code),
      };
    }
    case "approval.requested": {
      const requestId = textValue(payload.request_id);
      const pending = requestId !== "" && requestId === pendingApprovalId;
      const operation = textValue(payload.operation_class);
      const paths = stringList(payload.affected_paths);
      return {
        title: pending ? "Approval required" : "Checking approval policy",
        summary: `${humanize(tool || "tool")} · ${textValue(payload.risk) || "unknown"} risk`,
        detail: [operation ? humanize(operation) : "", paths.length ? paths.join(", ") : ""].filter(Boolean).join(" · "),
      };
    }
    case "approval.awaiting.human":
      return {
        title: "Approval required",
        summary: `${humanize(tool || "tool")} · ${textValue(payload.risk) || "unknown"} risk`,
        detail: payload.llm_review_available === true
          ? "Review the request and choose an action"
          : "Automatic review was unavailable; waiting for your decision",
      };
    case "approval.resolved": {
      const action = textValue(payload.action);
      const source = textValue(payload.source);
      return {
        title: approvalTitle(action, source),
        summary: humanize(tool || "tool"),
        detail: clipped(textValue(payload.reason)),
      };
    }
    case "approval.review.started":
      return { title: "Approval model review", summary: `Reviewing ${humanize(tool || "tool request")}`, detail: textValue(payload.risk) ? `${textValue(payload.risk)} risk` : "" };
    case "approval.review.completed":
      return { title: "Approval review complete", summary: `${humanize(textValue(payload.recommendation) || "reviewed")} · ${humanize(tool || "tool request")}`, detail: [textValue(payload.risk) ? `${textValue(payload.risk)} risk` : "", numberValue(payload.duration_seconds) !== null ? `${numberValue(payload.duration_seconds)!.toFixed(1)}s` : ""].filter(Boolean).join(" · ") };
    case "approval.review.failed":
      return { title: "Approval review unavailable", summary: `Falling back to human approval for ${humanize(tool || "tool request")}`, detail: numberValue(payload.duration_seconds) !== null ? `${numberValue(payload.duration_seconds)!.toFixed(1)}s` : "" };
    case "reasoning": {
      const nextAction = recordValue(payload.next_action);
      const nextTool = textValue(nextAction.tool_name) || textValue(nextAction.tool);
      const nextSummary = textValue(nextAction.argument_summary);
      return {
        title: humanize(textValue(payload.phase) || "Decision"),
        summary,
        detail: [textValue(payload.current_goal), nextTool ? `next: ${humanize(nextTool)}${nextSummary ? ` · ${nextSummary}` : ""}` : ""].filter(Boolean).join(" · "),
      };
    }
    case "assessment":
      return { title: "Assessment", summary, detail: stringList(payload.changes).length ? `changes: ${stringList(payload.changes).join(", ")}` : "" };
    case "auto.plan.started":
      return { title: "Automatic planning", summary: "Entered read-only Plan Mode", detail: clipped(textValue(payload.reason)) };
    case "verification.result": {
      const check = textValue(payload.check_id) || "verification";
      const reasons = stringList(payload.reasons);
      return { title: `Verification · ${check}`, summary: humanize(textValue(payload.status) || "completed"), detail: reasons.join(" · ") };
    }
    case "model.call":
      return {
        title: "Model request",
        summary: numberValue(payload.message_count) !== null ? `${numberValue(payload.message_count)} messages` : "Preparing the next model action",
        detail: numberValue(payload.effective_estimated_prompt_tokens) !== null ? `${numberValue(payload.effective_estimated_prompt_tokens)!.toLocaleString()} estimated prompt tokens` : "",
      };
    case "model.response": {
      const tools = stringList(payload.tools);
      return { title: "Model response", summary: tools.length ? `Requested ${tools.map(humanize).join(", ")}` : humanize(textValue(payload.finish_reason) || "Response received"), detail: "" };
    }
    case "model.response.invalid": {
      const recovery = textValue(payload.recovery);
      const argumentChars = numberValue(payload.argument_chars);
      return {
        title: "Invalid model tool request",
        summary: clipped(textValue(payload.error) || "The model returned a malformed tool call"),
        detail: [
          recovery === "bounded_edit" ? "Retrying as smaller snapshot-bound edits" : "Requesting a corrected response",
          textValue(payload.tool_name) ? humanize(textValue(payload.tool_name)) : "",
          argumentChars !== null ? `${argumentChars.toLocaleString()} argument characters` : "",
        ].filter(Boolean).join(" · "),
      };
    }
    case "model.response.continuation":
      return { title: "Recovering truncated response", summary: payload.thinking_disabled === true ? "Restarting with provider thinking disabled" : "Continuing the model response", detail: "" };
    case "model.stream.retry":
      return {
        title: "Retrying model stream",
        summary: `${humanize(textValue(payload.error_type) || "stream interrupted")} · attempt ${textValue(payload.attempt)}/${textValue(payload.max_attempts)}`,
        detail: numberValue(payload.delay_seconds) !== null ? `Retrying in ${numberValue(payload.delay_seconds)} seconds` : "",
      };
    case "model.stream.usage.unavailable":
      return {
        title: "Streaming usage unavailable",
        summary: "The provider supports streaming but not streamed usage metadata",
        detail: "Local token estimation remains active",
      };
    case "model.error": {
      const status = numberValue(payload.status_code);
      const code = textValue(payload.error_code);
      return {
        title: `Model error${status !== null ? ` · HTTP ${status}` : ""}`,
        summary: clipped(textValue(payload.message) || textValue(payload.error_type) || "The provider rejected the request"),
        detail: [code, payload.retryable === true ? `retry ${textValue(payload.attempt)}/${textValue(payload.max_attempts)}` : "not retryable"].filter(Boolean).join(" · "),
      };
    }
    default:
      return { title: eventLabels[event.type] || humanize(event.type), summary: summary || textValue(payload.status) || humanize(tool), detail: "" };
  }
}

const EventCard = memo(function EventCard({ event, pendingApprovalId }: { event: AgentEvent; pendingApprovalId: string | null }) {
  if (event.type === "session.start") {
    const task = textValue(event.payload.task);
    return <article className="message user timeline-user-message"><div className="message-label"><span>YOU</span><time>{new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></div><ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{task}</ReactMarkdown></article>;
  }
  const isTool = event.type.startsWith("tool.");
  const presentation = presentEvent(event, pendingApprovalId);
  const step = numberValue(event.payload.step);
  return <div className={`event-card ${event.type.replaceAll(".", "-")}`}><span className="event-icon">{isTool ? <Hammer size={14} /> : event.type.includes("approval") ? <ShieldAlert size={14} /> : <CheckCircle2 size={14} />}</span><div className="event-copy"><div className="event-heading"><strong>{presentation.title}</strong>{step !== null && <b>STEP {step}</b>}</div>{presentation.summary && <span>{presentation.summary}</span>}{presentation.detail && <small>{presentation.detail}</small>}</div><time>{new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></div>;
});

const ResultCard = memo(function ResultCard({ result }: { result: Record<string, any> }) {
  const successful = result.status === "success" || result.status === "plan_ready";
  const message = successful ? result.summary || result.reason : result.reason || result.summary;
  return <article className={`result-card ${result.status}`}><div className="result-title">{successful ? <CheckCircle2 /> : <ShieldAlert />}<strong>{result.status}</strong></div><ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{String(message || "Run finished")}</ReactMarkdown>{result.modified_files?.length > 0 && <small>Modified: {result.modified_files.join(", ")}</small>}</article>;
});

const PlanResultCard = memo(function PlanResultCard({ session, invoke, refresh }: {
  session: SessionDetail;
  invoke: (work: () => Promise<unknown>) => Promise<void>;
  refresh: () => Promise<void>;
}) {
  const plan = session.plan!;
  const [submitting, setSubmitting] = useState(false);
  const stale = plan.status === "stale";
  const execute = async () => {
    if (stale || submitting) return;
    setSubmitting(true);
    try {
      await invoke(async () => {
        await api(`/api/v1/sessions/${session.session_id}/plan/execute`, { method: "POST", body: "{}" });
        await refresh();
      });
    } finally {
      setSubmitting(false);
    }
  };
  return <article className={`conversation-plan ${stale ? "stale" : ""}`}>
    <div className="conversation-plan-header">
      <span><ListChecks size={18} /></span>
      <div><small>{stale ? "PLAN NEEDS REVIEW" : "PLAN READY"}</small><h2>{String(plan.goal)}</h2></div>
      <b>rev {plan.workspace_revision}</b>
    </div>
    <ol className="conversation-plan-steps">
      {(plan.steps || []).map((step: any, index: number) => <li key={step.step_id}>
        <b>{index + 1}</b>
        <div><strong>{step.title}</strong><small>{humanize(step.operation)} · {step.risk} risk{step.target_files?.length ? ` · ${step.target_files.join(", ")}` : ""}</small></div>
      </li>)}
    </ol>
    {plan.verification?.length > 0 && <div className="conversation-plan-verification"><strong>Verification</strong><span>{plan.verification.map((check: any) => check.title || check.check_id).join(" · ")}</span></div>}
    {plan.affected_files?.length > 0 && <div className="conversation-plan-files"><strong>Affected files</strong><span>{plan.affected_files.join(", ")}</span></div>}
    <div className="conversation-plan-actions">
      <p>{stale ? "The workspace changed after this plan was created. Revise it from the Plan panel before execution." : "Executing the plan still uses the configured approval policy for edits and commands."}</p>
      <button className="primary" disabled={stale || submitting} onClick={() => void execute()}>{submitting ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}{submitting ? "Starting…" : "Execute plan"}</button>
    </div>
  </article>;
});

function Inspector({ session, fileView, diffView, refreshDiff, mode, invoke, refresh }: any) {
  const [tab, setTab] = useState<"plan" | "file" | "diff">("plan");
  const plan = session?.plan;
  return <><div className="tabs"><button className={tab === "plan" ? "active" : ""} onClick={() => setTab("plan")}>Plan</button><button className={tab === "file" ? "active" : ""} onClick={() => setTab("file")}>File</button><button className={tab === "diff" ? "active" : ""} onClick={() => { setTab("diff"); void refreshDiff(); }}>Diff</button></div>
    <div className="inspector-content">
      {tab === "plan" && <>{!plan ? <div className="muted-card"><ListChecks size={20} /><p>{mode === "planning" ? "Plan Mode is active. Submit a task to begin read-only inspection." : "No plan artifact for this session."}</p></div> : <div className="plan-card"><div className="plan-status"><strong>{plan.goal}</strong><span>{plan.status}</span></div>{plan.steps?.map((step: any, index: number) => <div className={`plan-step ${step.status}`} key={step.step_id}><b>{index + 1}</b><div><strong>{step.title}</strong><small>{step.operation} · {step.risk}</small></div></div>)}{plan.status === "ready" || plan.status === "stale" ? <PlanControls sessionId={session.session_id} invoke={invoke} refresh={refresh} /> : null}</div>}
        <h3>Verification</h3>{Object.entries(session?.verification || {}).length === 0 ? <p className="subtle">No verification results yet.</p> : Object.entries(session.verification).map(([id, value]: any) => <div className="verification" key={id}><CheckCircle2 size={15} /><span><strong>{id}</strong><small>{value.status}</small></span></div>)}</>}
      {tab === "file" && (!fileView ? <div className="muted-card"><Files size={20} /><p>Select a workspace file to inspect its current snapshot.</p></div> : <><div className="file-title"><FileCode2 size={15} /><strong>{fileView.path}</strong><span>{fileView.snapshot_tag}</span></div><pre className="code-view">{fileView.content.split("\n").map((line: string, index: number) => <div key={index}><i>{fileView.start_line + index}</i><code>{line || " "}</code></div>)}</pre></>)}
      {tab === "diff" && <>{!diffView.trim() ? <div className="muted-card"><GitBranch size={20} /><p>The workspace has no tracked changes.</p></div> : <pre className="diff-view">{diffView}</pre>}</>}
    </div></>;
}

function PlanControls({ sessionId, invoke, refresh }: { sessionId: string; invoke: (work: () => Promise<unknown>) => Promise<void>; refresh: () => Promise<void> }) {
  const [action, setAction] = useState<"revise" | "inspect" | null>(null);
  const [instruction, setInstruction] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const submitInstruction = async () => {
    const value = instruction.trim();
    if (!action || !value || submitting) return;
    setSubmitting(true);
    await invoke(async () => {
      await api(`/api/v1/sessions/${sessionId}/plan/${action}`, { method: "POST", body: JSON.stringify({ instruction: value }) });
      setAction(null);
      setInstruction("");
      await refresh();
    });
    setSubmitting(false);
  };
  return <div className="plan-controls"><div className="plan-buttons"><button className="primary" disabled={submitting} onClick={() => void invoke(async () => { await api(`/api/v1/sessions/${sessionId}/plan/execute`, { method: "POST", body: "{}" }); await refresh(); })}><Play size={14} /> Execute plan</button><button disabled={submitting} onClick={() => { setAction("revise"); setInstruction(""); }}>Request revision</button><button disabled={submitting} onClick={() => { setAction("inspect"); setInstruction(""); }}>Continue inspection</button><button disabled={submitting} onClick={() => void invoke(async () => { await api(`/api/v1/sessions/${sessionId}/plan/cancel`, { method: "POST", body: "{}" }); await refresh(); })}>Cancel</button></div>{action && <div className="plan-instruction"><label>{action === "revise" ? "Revision direction" : "What should RepoRivet inspect next?"}<textarea autoFocus value={instruction} onChange={(event) => setInstruction(event.target.value)} /></label><div><button onClick={() => setAction(null)}>Back</button><button className="primary" disabled={!instruction.trim() || submitting} onClick={() => void submitInstruction()}>{submitting ? "Submitting…" : action === "revise" ? "Revise plan" : "Continue planning"}</button></div></div>}</div>;
}

function ApprovalDialog({ approval, onDecision }: { approval: Approval; onDecision: (action: string, guidance?: string) => void }) {
  const [guidance, setGuidance] = useState("");
  const targets = [...approval.writes, ...approval.deletes, ...approval.reads];
  return <div className="modal-backdrop"><div className="approval-modal">
    <div className="approval-header"><span><ShieldAlert /></span><div><small>APPROVAL REQUIRED</small><h2>{humanize(approval.tool)}</h2></div><b className={`risk ${approval.risk}`}>{approval.risk} risk</b></div>
    <div className="approval-scroll">
      <div className="approval-summary">
        <div><span>Operation</span><strong>{humanize(approval.operation)}</strong></div>
        <div><span>Target</span><strong title={targets.join(", ")}>{targets.join(", ") || "Workspace"}</strong></div>
        <div><span>Known effects</span><strong title={approval.effects.join(", ")}>{approval.effects.map(humanize).join(", ") || "None declared"}</strong></div>
      </div>

      {approval.details.length > 0 && <section className="approval-section"><h3>Requested action</h3><dl className="approval-details">{approval.details.map((detail) => <div key={`${detail.label}-${detail.value}`}><dt>{detail.label}</dt><dd>{detail.value}</dd></div>)}</dl></section>}
      {approval.operations.length > 0 && <section className="approval-section"><h3>Edit operations</h3><ol className="operation-list">{approval.operations.map((operation, index) => <li key={`${index}-${operation}`}><b>{index + 1}</b><span>{operation}</span></li>)}</ol></section>}
      {approval.preview && <section className="approval-section"><h3>{approval.preview.title}</h3><ApprovalPreview preview={approval.preview} /></section>}
      {approval.reasons.length > 0 && <section className="approval-section risk-reasons"><h3>Why approval is needed</h3>{approval.reasons.map((reason) => <p key={reason}>• {reason}</p>)}</section>}
      {approval.review && <section className="approval-section review-card"><div><strong>LLM review</strong><span>{approval.review.recommendation} · {approval.review.risk} risk · {approval.review.relevance}</span></div><p>{approval.review.reason}</p>{approval.review.question && <p className="review-question">{approval.review.question}</p>}</section>}
    </div>
    <div className="approval-footer">
      <label className="guidance">Optional direction if denied<textarea value={guidance} onChange={(event) => setGuidance(event.target.value)} placeholder="Tell the agent what to do instead…" /></label>
      <div className="approval-actions"><button className="primary" onClick={() => onDecision("allow_once")}><CheckCircle2 size={15} /> Allow once</button><button disabled={!approval.allow_matching_repeats} title={approval.allow_matching_repeats ? "Allow this exact request for the rest of the session" : "Repeating grants are unavailable for high-risk requests"} onClick={() => onDecision("allow_session")}><RefreshCw size={15} /> Allow matching repeats</button><button className="deny" onClick={() => onDecision("deny", guidance || undefined)}><XCircle size={15} /> Deny and continue</button><button className="stop" onClick={() => onDecision("stop")}><CircleStop size={15} /> Stop run</button></div>
    </div>
  </div></div>;
}

function ApprovalPreview({ preview }: { preview: NonNullable<Approval["preview"]> }) {
  if (preview.kind === "content") return <pre className="approval-code"><code>{preview.text || "(empty file)"}</code></pre>;
  return <div className="approval-diff">{preview.text.split("\n").map((line, index) => {
    const tone = line.startsWith("@@") ? "hunk" : line.startsWith("+") && !line.startsWith("+++") ? "added" : line.startsWith("-") && !line.startsWith("---") ? "removed" : "context";
    return <div className={tone} key={index}><i>{line.slice(0, 1) || " "}</i><code>{line || " "}</code></div>;
  })}</div>;
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replaceAll("-", " ");
}

export default App;
