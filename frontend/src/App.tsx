import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  Bot,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleStop,
  CircleOff,
  ClipboardList,
  CornerUpRight,
  Feather,
  FileCode2,
  Files,
  Folder,
  Gauge,
  GitBranch,
  Hammer,
  ListChecks,
  ListPlus,
  ListTodo,
  LoaderCircle,
  MessageSquareText,
  Moon,
  Play,
  Plus,
  PackageOpen,
  Puzzle,
  RefreshCw,
  Route,
  Send,
  ShieldAlert,
  ShieldCheck,
  ShieldOff,
  ShieldQuestion,
  Sparkles,
  Sun,
  Trash2,
  XCircle,
  Zap,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AgentEvent, Approval, EventPage, SessionDetail, SessionSummary, api, authenticate } from "./api";

type ReasoningEffort = "low" | "medium" | "high" | "xhigh" | "max";
type ReasoningPolicy = "adaptive" | "fixed";
type Bootstrap = {
  workspace: string;
  active_session_id: string | null;
  sessions: SessionSummary[];
  settings: { model: string; base_url: string; context_limit: number; approval_mode: string; auto_plan: "off" | "adaptive" | "always"; auto_plan_llm: boolean; reasoning_policy: ReasoningPolicy; reasoning_effort: ReasoningEffort; reasoning_supported_efforts: ReasoningEffort[] };
};
type FileEntry = { name: string; path: string; kind: "file" | "directory"; size: number | null };
type Skill = { id: string; name: string; description: string; version: string; source: string };
type Theme = "dark" | "light";
type RunDelivery = "redirect" | "queue";

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
  "action.recovery.started",
  "external.files.changed",
  "plan.approved",
  "plan.cancelled",
  "plan.submitted",
  "plan.updated",
  "plan.step.finished",
  "auto.plan.started",
  "run.finished",
  "runtime.settings.changed",
  "session.start",
  "user.input",
  "verification.result",
  "web.run.finished",
]);

const hiddenTimelineEvents = new Set([
  "model.call.finished",
  "model.stream.progress",
  "runtime.transition",
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
  const [reasoningPolicy, setReasoningPolicy] = useState<ReasoningPolicy>("adaptive");
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>("max");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<SessionSummary | null>(null);
  const [deletingSession, setDeletingSession] = useState(false);
  const creatingSession = useRef<Promise<string> | null>(null);
  const selectedSessionId = useRef<string | null>(null);

  const refreshSession = useCallback(async (sessionId?: string | null) => {
    if (!sessionId) return;
    const detail = await api<SessionDetail>(`/api/v1/sessions/${sessionId}`);
    selectedSessionId.current = detail.session_id;
    setSession(detail);
    const runtimeSettings = detail.run?.settings;
    if (runtimeSettings?.mode) setMode(runtimeSettings.mode);
    if (runtimeSettings?.approval_mode) setApprovalMode(runtimeSettings.approval_mode);
    else if (detail.approval_mode) setApprovalMode(detail.approval_mode);
    if (runtimeSettings?.auto_plan) setAutoPlan(runtimeSettings.auto_plan);
    setSelectedSkill(runtimeSettings?.skill ?? detail.active_skill?.id ?? "");
    if (runtimeSettings?.reasoning_policy) setReasoningPolicy(runtimeSettings.reasoning_policy);
    if (runtimeSettings?.reasoning_effort) setReasoningEffort(runtimeSettings.reasoning_effort);
  }, []);

  const initialize = useCallback(async () => {
    setLoading(true);
    try {
      await authenticate();
      const data = await api<Bootstrap>("/api/v1/bootstrap");
      setBoot(data);
      setApprovalMode(data.settings.approval_mode);
      setAutoPlan(data.settings.auto_plan);
      setReasoningPolicy(data.settings.reasoning_policy);
      setReasoningEffort(data.settings.reasoning_effort);
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
    if (session?.run?.settings.mode) {
      setMode(session.run.settings.mode);
    } else if (session?.workflow_mode === "planning") {
      setMode("planning");
    } else if (session?.workflow_mode === "execute") {
      setMode("execute");
    }
  }, [session?.workflow_mode, session?.run?.settings.mode]);

  const invoke = useCallback(async (work: () => Promise<unknown>) => {
    setError("");
    try { await work(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }, []);

  const createAndSelectSession = useCallback(async () => {
    const created = await api<SessionSummary>("/api/v1/sessions", { method: "POST", body: JSON.stringify({}) });
    selectedSessionId.current = created.session_id;
    const data = await api<Bootstrap>("/api/v1/bootstrap");
    setBoot(data);
    await refreshSession(created.session_id);
    return created.session_id;
  }, [refreshSession]);

  const createSession = () => invoke(async () => { await createAndSelectSession(); });

  const ensureSession = useCallback(async () => {
    if (selectedSessionId.current) return selectedSessionId.current;
    if (!creatingSession.current) {
      creatingSession.current = createAndSelectSession().finally(() => {
        creatingSession.current = null;
      });
    }
    return creatingSession.current;
  }, [createAndSelectSession]);

  const selectSession = (id: string) => invoke(async () => {
    await api(`/api/v1/sessions/${id}/use`, { method: "POST", body: "{}" });
    await refreshSession(id);
  });

  const deleteSession = async () => {
    if (!deleteTarget || deletingSession) return;
    const targetId = deleteTarget.session_id;
    const deletedCurrent = selectedSessionId.current === targetId;
    setDeletingSession(true);
    setError("");
    try {
      await api(`/api/v1/sessions/${targetId}`, { method: "DELETE" });
      const data = await api<Bootstrap>("/api/v1/bootstrap");
      if (deletedCurrent) {
        selectedSessionId.current = null;
        setSession(null);
        const next = data.sessions[0];
        if (next) {
          await api(`/api/v1/sessions/${next.session_id}/use`, { method: "POST", body: "{}" });
          data.active_session_id = next.session_id;
          await refreshSession(next.session_id);
        }
      }
      setBoot(data);
      setDeleteTarget(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setDeletingSession(false);
    }
  };

  const submit = async (task: string, delivery: RunDelivery) => {
    setError("");
    try {
      const sessionId = await ensureSession();
      await api(`/api/v1/sessions/${sessionId}/runs`, {
        method: "POST",
        body: JSON.stringify({ task, mode, approval_mode: approvalMode, auto_plan: autoPlan, skill: selectedSkill || null, clear_skill: !selectedSkill, reasoning_policy: reasoningPolicy, reasoning_effort: reasoningEffort, delivery }),
      });
      const data = await api<Bootstrap>("/api/v1/bootstrap");
      setBoot(data);
      await refreshSession(sessionId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      throw cause;
    }
  };

  const openDirectory = (path: string) => invoke(async () => {
    setCurrentDir(path);
    setFiles(await api<FileEntry[]>(`/api/v1/workspace/files?path=${encodeURIComponent(path)}`));
  });

  const openFile = (path: string) => invoke(async () => {
    setFileView(await api(`/api/v1/workspace/file?path=${encodeURIComponent(path)}`));
  });

  const approval = session?.run?.pending_approval || null;
  const isRunning = ["queued", "running", "awaiting_approval", "stopping"].includes(session?.run?.status || "");
  const planReady = session?.plan?.status === "ready" || session?.plan?.status === "stale";
  const updateOptions = (updates: Partial<{
    mode: "execute" | "planning";
    approvalMode: string;
    autoPlan: "off" | "adaptive" | "always";
    selectedSkill: string;
    reasoningPolicy: ReasoningPolicy;
    reasoningEffort: ReasoningEffort;
  }>) => {
    const nextApprovalMode = updates.approvalMode ?? approvalMode;
    if (updates.mode) setMode(updates.mode);
    if (updates.approvalMode) setApprovalMode(updates.approvalMode);
    if (updates.autoPlan) setAutoPlan(updates.autoPlan);
    if (updates.selectedSkill !== undefined) setSelectedSkill(updates.selectedSkill);
    if (updates.reasoningPolicy) setReasoningPolicy(updates.reasoningPolicy);
    if (updates.reasoningEffort) setReasoningEffort(updates.reasoningEffort);
    if (!session) return;
    if (isRunning) {
      const payload: Record<string, unknown> = {};
      if (updates.mode) payload.mode = updates.mode;
      if (updates.approvalMode) payload.approval_mode = updates.approvalMode;
      if (updates.autoPlan) payload.auto_plan = updates.autoPlan;
      if (updates.selectedSkill !== undefined) {
        payload.skill = updates.selectedSkill || null;
        payload.clear_skill = !updates.selectedSkill;
      }
      if (updates.reasoningPolicy) payload.reasoning_policy = updates.reasoningPolicy;
      if (updates.reasoningEffort) payload.reasoning_effort = updates.reasoningEffort;
      void invoke(() => api(`/api/v1/sessions/${session.session_id}/runtime-settings`, {
        method: "PUT",
        body: JSON.stringify(payload),
      }));
    } else if (updates.approvalMode) {
      void invoke(() => api(`/api/v1/sessions/${session.session_id}/approval-mode`, { method: "PUT", body: JSON.stringify({ mode: nextApprovalMode }) }));
    } else if (updates.selectedSkill === "") {
      void invoke(() => api(`/api/v1/sessions/${session.session_id}/skill`, { method: "DELETE" }));
    }
  };

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
          {boot.sessions.map((item) => <div key={item.session_id} className={`session-row ${session?.session_id === item.session_id ? "active" : ""}`}>
            <button className="session-item" onClick={() => selectSession(item.session_id)}>
              <MessageSquareText size={16} /><span><strong>{item.name}</strong><small>{item.task_preview || item.short_id}</small></span><i className={`status-dot ${item.status}`} />
            </button>
            <button
              className="session-delete"
              aria-label={`Delete conversation ${item.name}`}
              title={session?.session_id === item.session_id && isRunning ? "Stop the current run before deleting" : "Delete conversation"}
              disabled={session?.session_id === item.session_id && isRunning}
              onClick={() => setDeleteTarget(item)}
            ><Trash2 size={13} /></button>
          </div>)}
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
          <div>
            <Bot size={18} />
            <strong>{session?.name || "No session selected"}</strong>
            {session && <span className="revision">rev {session.workspace_revision}</span>}
            {session?.runtime && <span className={`runtime-state ${session.runtime.status}`} title={`Run ${session.runtime.run_id}`}>
              {humanize(session.runtime.phase)}
              {session.runtime.wait ? ` · waiting for ${humanize(session.runtime.wait.kind)}` : ""}
            </span>}
          </div>
        </div>
        {error && <div className="error-banner"><ShieldAlert size={16} />{error}<button onClick={() => setError("")}>×</button></div>}
        <Timeline session={session} isRunning={isRunning} pendingApprovalId={approval?.request_id || null} invoke={invoke} onRefresh={refreshSession} />
        <Composer
          disabled={false}
          isRunning={isRunning}
          mode={mode}
          setMode={(value) => updateOptions({ mode: value })}
          planReady={planReady}
          autoPlan={autoPlan}
          setAutoPlan={(value) => updateOptions({ autoPlan: value })}
          approvalMode={approvalMode}
          setApprovalMode={(value) => updateOptions({ approvalMode: value })}
          skills={skills}
          selectedSkill={selectedSkill}
          setSelectedSkill={(value) => updateOptions({ selectedSkill: value })}
          reasoningEffort={reasoningEffort}
          reasoningPolicy={reasoningPolicy}
          setReasoningPolicy={(value) => updateOptions({ reasoningPolicy: value })}
          setReasoningEffort={(value) => updateOptions({ reasoningEffort: value })}
          settings={boot.settings}
          onActivate={() => { void invoke(async () => { await ensureSession(); }); }}
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
      {deleteTarget && <div className="modal-backdrop delete-backdrop" role="presentation" onMouseDown={(event) => {
        if (event.target === event.currentTarget && !deletingSession) setDeleteTarget(null);
      }}><div className="delete-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-dialog-title" aria-describedby="delete-dialog-description">
        <span className="delete-dialog-icon"><Trash2 size={20} /></span>
        <div><small>DELETE CONVERSATION</small><h2 id="delete-dialog-title">{deleteTarget.name}</h2></div>
        <p id="delete-dialog-description">This permanently deletes the conversation, its history, and saved run artifacts. This action cannot be undone.</p>
        <div className="delete-dialog-actions"><button disabled={deletingSession} onClick={() => setDeleteTarget(null)}>Cancel</button><button className="danger" disabled={deletingSession} onClick={() => void deleteSession()}>{deletingSession ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}{deletingSession ? "Deleting…" : "Delete conversation"}</button></div>
      </div></div>}
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
    "model.reasoning.effort.downgraded",
    "model.reasoning.protocol.recovery",
    "auto.plan.review.started",
    "auto.plan.review.completed",
    "auto.plan.review.failed",
    "approval.review.started",
    "approval.review.completed",
    "approval.review.failed",
  ].includes(event.type));
  let workingLabel = "Model is working";
  if (latestModelActivity?.type === "model.stream.progress") {
    const elapsed = numberValue(latestModelActivity.payload.elapsed_seconds);
    const contentChars = numberValue(latestModelActivity.payload.content_chars);
    const toolChars = numberValue(latestModelActivity.payload.tool_argument_chars);
    const phase = textValue(latestModelActivity.payload.activity_phase);
    const reasoningEffort = textValue(latestModelActivity.payload.reasoning_effort);
    const effortLabel = reasoningEffort ? ` · ${reasoningEffort} reasoning` : "";
    const elapsedLabel = elapsed !== null ? ` · ${formatElapsed(elapsed)}` : "";
    if (latestModelActivity.payload.completed === true) {
      workingLabel = "Finishing streamed model response";
    } else if (toolChars !== null && toolChars > 0) {
      workingLabel = `Receiving tool request · ${toolChars.toLocaleString()} characters${elapsedLabel}`;
    } else if (contentChars !== null && contentChars > 0) {
      workingLabel = `Receiving answer · ${contentChars.toLocaleString()} characters${elapsedLabel}`;
    } else if (phase && phase !== "waiting") {
      workingLabel = `Model reasoning${effortLabel} · ${reasoningProgressLabel(phase)}${elapsedLabel}`;
    } else {
      workingLabel = `Waiting for model output${elapsedLabel}`;
    }
  } else if (latestModelActivity?.type === "model.reasoning.protocol.recovery") {
    workingLabel = "Provider reasoning state was not replayable · retrying this request directly";
  } else if (latestModelActivity?.type === "model.reasoning.effort.downgraded") {
    workingLabel = `No actionable response yet · retrying with ${textValue(latestModelActivity.payload.reasoning_effort) || "lower"} reasoning`;
  } else if (latestModelActivity?.type === "auto.plan.review.started") {
    workingLabel = "Evaluating whether Plan Mode is needed";
  } else if (latestModelActivity?.type === "auto.plan.review.completed") {
    workingLabel = "Plan Mode evaluation completed";
  } else if (latestModelActivity?.type === "auto.plan.review.failed") {
    workingLabel = "Plan Mode evaluation unavailable · continuing safely";
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
      if (event.type === "session.start" || event.type === "user.input") {
        stickToBottom.current = true;
      }
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

type ComposerIconOption = {
  value: string;
  label: string;
  icon: ReactNode;
};

function ComposerIconSelect({ label, value, options, disabled = false, className = "", onChange }: {
  label: string;
  value: string;
  options: ComposerIconOption[];
  disabled?: boolean;
  className?: string;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const current = options.find((option) => option.value === value) || options[0];

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return <div className="composer-control-wrap" ref={rootRef}>
    <button
      type="button"
      className={`composer-control ${className}`}
      aria-label={`${label}: ${current.label}`}
      aria-expanded={open}
      aria-haspopup="listbox"
      disabled={disabled}
      title={`${label}: ${current.label}`}
      onClick={() => setOpen((currentOpen) => !currentOpen)}
    >
      {current.icon}<ChevronDown className="control-chevron" size={8} />
    </button>
    {open && <div className="composer-menu" role="listbox" aria-label={label}>
      <small>{label}</small>
      {options.map((option) => <button
        type="button"
        role="option"
        aria-selected={option.value === value}
        className={option.value === value ? "selected" : ""}
        key={option.value}
        onClick={() => {
          onChange(option.value);
          setOpen(false);
        }}
      ><span>{option.icon}</span><strong>{option.label}</strong>{option.value === value && <Check size={13} />}</button>)}
    </div>}
  </div>;
}

function ComposerReasoningSlider({ policy, value, onPolicyChange, onChange }: {
  policy: ReasoningPolicy;
  value: ReasoningEffort;
  onPolicyChange: (value: ReasoningPolicy) => void;
  onChange: (value: ReasoningEffort) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const levels: ReasoningEffort[] = ["low", "medium", "high", "xhigh", "max"];
  const labels: Record<ReasoningEffort, string> = { low: "Low", medium: "Medium", high: "High", xhigh: "XHigh", max: "Maximum" };
  const levelIcons = {
    low: <Feather size={15} />,
    medium: <Activity size={15} />,
    high: <Activity size={15} />,
    xhigh: <BrainCircuit size={15} />,
    max: <Zap size={15} />,
  };

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return <div className="composer-control-wrap reasoning-control-wrap" ref={rootRef}>
    <button
      type="button"
      className={`composer-control reasoning-trigger ${value}`}
      aria-label={policy === "adaptive" ? `Adaptive reasoning up to ${labels[value]}` : `Fixed reasoning: ${labels[value]}`}
      aria-expanded={open}
      aria-haspopup="dialog"
      title={policy === "adaptive" ? `Adaptive reasoning · up to ${labels[value]}` : `Fixed reasoning · ${labels[value]}`}
      onClick={() => setOpen((currentOpen) => !currentOpen)}
    ><Gauge size={16} /><ChevronDown className="control-chevron" size={8} /></button>
    {open && <div className="composer-menu reasoning-menu" role="dialog" aria-label="Reasoning policy">
      <small>Reasoning policy</small>
      <div className="reasoning-policy-toggle" role="group" aria-label="Reasoning selection mode">
        <button type="button" className={policy === "adaptive" ? "selected" : ""} onClick={() => onPolicyChange("adaptive")}>Adaptive</button>
        <button type="button" className={policy === "fixed" ? "selected" : ""} onClick={() => onPolicyChange("fixed")}>Fixed</button>
      </div>
      <div className={`reasoning-value ${value}`}>{levelIcons[value]}<span>{policy === "adaptive" ? "Maximum level" : "Fixed level"}</span><strong>{labels[value]}</strong></div>
      <input
        className={`reasoning-range ${value}`}
        type="range"
        min={0}
        max={4}
        step={1}
        value={levels.indexOf(value)}
        aria-label={policy === "adaptive" ? "Maximum reasoning effort" : "Fixed reasoning effort"}
        aria-valuetext={labels[value]}
        onChange={(event) => onChange(levels[Number(event.target.value)])}
      />
      <div className="reasoning-scale"><span>Low</span><span>Med</span><span>High</span><span>XHigh</span><span>Max</span></div>
      <p>{policy === "adaptive" ? "The controller selects a one-call effort at or below this ceiling." : "Advanced: every model call uses this fixed level unless provider recovery lowers it."}</p>
    </div>}
  </div>;
}

const Composer = memo(function Composer({ disabled, isRunning, mode, setMode, planReady, autoPlan, setAutoPlan, approvalMode, setApprovalMode, skills, selectedSkill, setSelectedSkill, reasoningPolicy, setReasoningPolicy, reasoningEffort, setReasoningEffort, settings, onActivate, onStop, onSubmit }: {
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
  reasoningPolicy: ReasoningPolicy;
  setReasoningPolicy: (policy: ReasoningPolicy) => void;
  reasoningEffort: ReasoningEffort;
  setReasoningEffort: (effort: ReasoningEffort) => void;
  settings: Bootstrap["settings"];
  onActivate: () => void;
  onStop: () => Promise<void> | void;
  onSubmit: (task: string, delivery: RunDelivery) => Promise<void> | void;
}) {
  const [value, setValue] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [delivery, setDelivery] = useState<RunDelivery>("redirect");
  const send = async () => {
    const task = value.trim();
    if (!task || disabled || submitting) return;
    setSubmitting(true);
    try {
      await onSubmit(task, delivery);
      setValue((current) => current.trim() === task ? "" : current);
    }
    catch { /* The app-level submit handler preserves and displays the error. */ }
    finally { setSubmitting(false); }
  };
  const stopsRun = isRunning && !value.trim();
  const sendLabel = isRunning
    ? delivery === "redirect" ? "Redirect current run" : "Queue follow-up"
    : "Send prompt";
  const iconSize = 14;
  const deliveryOptions: ComposerIconOption[] = [
    { value: "redirect", label: "Redirect now", icon: <CornerUpRight size={iconSize} /> },
    { value: "queue", label: "Queue next", icon: <ListPlus size={iconSize} /> },
  ];
  const workflowOptions: ComposerIconOption[] = [
    { value: "execute", label: "Execute", icon: <Hammer size={iconSize} /> },
    { value: "planning", label: planReady ? "Plan ready" : "Plan", icon: <ClipboardList size={iconSize} /> },
  ];
  const autoPlanOptions: ComposerIconOption[] = [
    { value: "adaptive", label: "Adaptive plan", icon: <Route size={iconSize} /> },
    { value: "always", label: "Plan first", icon: <ListTodo size={iconSize} /> },
    { value: "off", label: "No auto plan", icon: <CircleOff size={iconSize} /> },
  ];
  const approvalOptions: ComposerIconOption[] = [
    { value: "safe-auto", label: "Safe auto", icon: <ShieldCheck size={iconSize} /> },
    { value: "llm-auto", label: "LLM auto", icon: <BrainCircuit size={iconSize} /> },
    { value: "always-ask", label: "Always ask", icon: <ShieldQuestion size={iconSize} /> },
    { value: "allow-all", label: "Allow all", icon: <ShieldOff size={iconSize} /> },
  ];
  const skillOptions: ComposerIconOption[] = [
    { value: "", label: "No skill", icon: <PackageOpen size={iconSize} /> },
    ...skills.filter((skill) => skill.source === "global").map((skill) => ({ value: skill.id, label: skill.name, icon: <Puzzle size={iconSize} /> })),
  ];
  return <div className="composer-wrap">
    <div className="composer-shell">
      <div className="composer-input">
        <textarea value={value} disabled={disabled} onFocus={onActivate} onChange={(event) => setValue(event.target.value)} onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void send();
          }
        }} placeholder={isRunning ? delivery === "redirect" ? "Redirect the current run now…" : "Queue a follow-up for after the current task…" : mode === "planning" ? "Ask RepoRivet to inspect and prepare a plan…" : "Describe what you want to change…"} />
        <button className={`send-button ${stopsRun ? "stop" : ""}`} aria-label={stopsRun ? "Stop current run" : sendLabel} title={stopsRun ? "Stop current run" : sendLabel} disabled={disabled || submitting || (!isRunning && !value.trim())} onClick={() => void (stopsRun ? onStop() : send())}>{stopsRun ? <CircleStop size={18} /> : submitting ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}</button>
        <div className="composer-meta"><span>{isRunning ? delivery === "redirect" ? "Enter to redirect · clear the field to stop" : "Enter to queue · clear the field to stop" : "Enter to send · Shift+Enter for newline"}</span><span>{value.length.toLocaleString()} chars</span></div>
      </div>
      <div className="composer-toolbar">
        <div className="composer-controls">
          {isRunning && <ComposerIconSelect label="Message delivery" value={delivery} options={deliveryOptions} className="active" onChange={(next) => setDelivery(next as RunDelivery)} />}
          <ComposerIconSelect label="Workflow mode" value={mode} options={workflowOptions} disabled={planReady} className={mode === "planning" ? "plan" : ""} onChange={(next) => setMode(next as "execute" | "planning")} />
          <ComposerIconSelect label={settings.auto_plan_llm ? "Automatic planning" : "Automatic planning · LLM classifier disabled"} value={autoPlan} options={autoPlanOptions} className={autoPlan === "off" ? "muted" : ""} onChange={(next) => setAutoPlan(next as "off" | "adaptive" | "always")} />
          <ComposerIconSelect label="Approval mode" value={approvalMode} options={approvalOptions} className={approvalMode === "always-ask" ? "warning" : ""} onChange={setApprovalMode} />
          <ComposerReasoningSlider policy={reasoningPolicy} value={reasoningEffort} onPolicyChange={setReasoningPolicy} onChange={setReasoningEffort} />
          <ComposerIconSelect label="Global Skill" value={selectedSkill} options={skillOptions} className={selectedSkill ? "active" : "muted"} onChange={setSelectedSkill} />
        </div>
        <div className="composer-model" title={`${settings.base_url} · ${settings.context_limit.toLocaleString()} token context`}><Bot size={13} /><span>{settings.model}</span><i>·</i><span>{settings.context_limit.toLocaleString()}</span></div>
      </div>
    </div>
  </div>;
});

function EmptyState() { return <div className="empty-state"><div className="empty-icon"><GitBranch size={30} /></div><h2>Build with evidence</h2><p>Click the composer to start a new conversation, or select an existing session from the sidebar.</p></div>; }

type EventPresentation = { title: string; summary: string; detail: string };

function textValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}m ${remainder.toString().padStart(2, "0")}s`;
}

function reasoningProgressLabel(phase: string): string {
  const labels: Record<string, string> = {
    understanding_task: "understanding the task",
    analyzing_context: "understanding the task › analyzing context and constraints",
    evaluating_options: "understanding the task › analyzing context › evaluating the next action",
    refining_action: "understanding the task › analyzing context › evaluating options › refining the action",
    composing_answer: "composing the answer",
    preparing_tool: "preparing a structured tool request",
    completed: "finishing the response",
  };
  return labels[phase] || "working through the request";
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
  if (tool === "delete_path") {
    const recursive = args.recursive === true ? " · recursive" : "";
    return `Delete ${path || "workspace path"}${recursive}`;
  }
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
      return {
        title: "Approval review unavailable",
        summary: `Falling back to human approval for ${humanize(tool || "tool request")}`,
        detail: [
          textValue(payload.error_type) ? humanize(textValue(payload.error_type)) : "",
          textValue(payload.stage) ? `during ${humanize(textValue(payload.stage))}` : "",
          numberValue(payload.duration_seconds) !== null ? `${numberValue(payload.duration_seconds)!.toFixed(1)}s` : "",
        ].filter(Boolean).join(" · "),
      };
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
    case "action.result.reused":
      return {
        title: "Previous result reused",
        summary: `${humanize(textValue(payload.tool) || "tool")} was not executed again`,
        detail: humanize(textValue(payload.disposition) || "valid result"),
      };
    case "action.retry.scheduled":
      return {
        title: "Transient action retry",
        summary: `${humanize(textValue(payload.tool) || "tool")} · attempt ${textValue(payload.attempt)}`,
        detail: humanize(textValue(payload.error_code)),
      };
    case "action.recovery.started":
      return {
        title: "Recovery started",
        summary: `${humanize(textValue(payload.tool) || "action")} needs a different next action`,
        detail: humanize(textValue(payload.reason_code)),
      };
    case "auto.plan.started":
      return { title: "Automatic planning", summary: "Entered read-only Plan Mode", detail: clipped(textValue(payload.reason)) };
    case "auto.plan.review.started":
      return {
        title: "Adaptive Plan evaluation",
        summary: "Classifying the task before the coding agent starts",
        detail: payload.workspace_empty === true ? "Empty workspace" : `${numberValue(payload.sampled_files) ?? 0} sampled files`,
      };
    case "auto.plan.review.completed":
      return {
        title: "Adaptive Plan evaluation",
        summary: `${humanize(textValue(payload.decision) || "reviewed")} · ${clipped(textValue(payload.reason), 300)}`,
        detail: [
          numberValue(payload.confidence) !== null ? `${Math.round(numberValue(payload.confidence)! * 100)}% confidence` : "",
          payload.applied === true ? "Plan Mode selected" : "Execute Mode selected",
          numberValue(payload.input_tokens) !== null ? `${numberValue(payload.input_tokens)!.toLocaleString()} input tokens` : "",
          numberValue(payload.output_tokens) !== null ? `${numberValue(payload.output_tokens)!.toLocaleString()} output tokens` : "",
          numberValue(payload.duration_seconds) !== null ? `${numberValue(payload.duration_seconds)!.toFixed(1)}s` : "",
        ].filter(Boolean).join(" · "),
      };
    case "auto.plan.review.failed":
      return {
        title: "Adaptive Plan evaluation unavailable",
        summary: "Continuing in Execute Mode; the main agent may still request planning",
        detail: clipped(textValue(payload.reason)),
      };
    case "verification.result": {
      const check = textValue(payload.check_id) || "verification";
      const reasons = stringList(payload.reasons);
      return { title: `Verification · ${check}`, summary: humanize(textValue(payload.status) || "completed"), detail: reasons.join(" · ") };
    }
    case "model.call": {
      const effort = textValue(payload.reasoning_effort);
      const ceiling = textValue(payload.reasoning_effort_ceiling);
      const policy = textValue(payload.reasoning_policy);
      const phase = textValue(payload.reasoning_phase);
      return {
        title: "Model request",
        summary: numberValue(payload.message_count) !== null ? `${numberValue(payload.message_count)} messages` : "Preparing the next model action",
        detail: [
          numberValue(payload.effective_estimated_prompt_tokens) !== null ? `${numberValue(payload.effective_estimated_prompt_tokens)!.toLocaleString()} estimated prompt tokens` : "",
          effort ? `${effort} reasoning · ${policy === "fixed" ? "fixed" : `adaptive up to ${ceiling || effort}`}` : "",
          phase ? humanize(phase) : "",
          clipped(textValue(payload.reasoning_reason), 240),
        ].filter(Boolean).join(" · "),
      };
    }
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
    case "user.input":
      return { title: "Follow-up queued", summary: clipped(textValue(payload.task)), detail: "Starts after the current task finishes" };
    case "model.redirected":
      return { title: "Direction changed", summary: "Stopped the previous model response and continued with your latest instruction", detail: "" };
    case "model.stream.retry":
      return {
        title: "Retrying model stream",
        summary: `${humanize(textValue(payload.error_type) || "stream interrupted")} · attempt ${textValue(payload.attempt)}/${textValue(payload.max_attempts)}`,
        detail: numberValue(payload.delay_seconds) !== null ? `Retrying in ${numberValue(payload.delay_seconds)} seconds` : "",
      };
    case "model.reasoning.protocol.recovery":
      return {
        title: "Recovering provider reasoning protocol",
        summary: "Retrying this request without thinking; the next turn restarts thinking context",
        detail: humanize(textValue(payload.error_type)),
      };
    case "model.reasoning.effort.downgraded":
      return {
        title: "Reasoning effort reduced",
        summary: `No actionable response arrived in ${formatElapsed(numberValue(payload.elapsed_seconds) || 0)}; retrying at ${textValue(payload.reasoning_effort) || "a lower tier"}`,
        detail: `${textValue(payload.previous_effort) || "higher"} → ${textValue(payload.reasoning_effort) || "lower"}`,
      };
    case "model.reasoning.effort.mapped":
      return {
        title: "Reasoning tier mapped",
        summary: `${textValue(payload.requested_effort)} requested · ${textValue(payload.applied_effort)} applied`,
        detail: clipped(textValue(payload.reason)),
      };
    case "stale.snapshot.recovery.started":
      return {
        title: "Refreshing stale edit target",
        summary: `${textValue(payload.path) || "Workspace file"} · lines ${textValue(payload.start_line)}–${textValue(payload.end_line)}`,
        detail: "The rejected edit did not change the file; reading its current contents automatically",
      };
    case "stale.snapshot.recovery.finished":
      return {
        title: payload.ok === true ? "Fresh snapshot ready" : "Snapshot refresh failed",
        summary: textValue(payload.path) || "Workspace file",
        detail: payload.ok === true
          ? `Current lines ${textValue(payload.start_line)}–${textValue(payload.end_line)} are ready for a regenerated edit`
          : "The agent must read the file before editing again",
      };
    case "plan.scope.revision.required":
      return {
        title: "Plan revision required",
        summary: clipped(textValue(payload.reason) || "The required action exceeds the approved plan"),
        detail: [
          textValue(payload.current_step_id) ? `current step: ${textValue(payload.current_step_id)}` : "",
          textValue(payload.rejected_path),
          "Requesting one updated plan before execution continues",
        ].filter(Boolean).join(" · "),
      };
    case "duplicate.successful.command.suppressed":
      return {
        title: "Duplicate command skipped",
        summary: clipped(textValue(payload.command) || "The command already succeeded"),
        detail: `The previous exit-0 result remains valid at workspace revision ${textValue(payload.workspace_revision)}`,
      };
    case "verification.plan.recovery.started":
      return {
        title: "Verification plan required",
        summary: stringList(payload.requested_check_ids).length
          ? `Registering ${stringList(payload.requested_check_ids).join(", ")}`
          : "Registering deterministic checks before verification continues",
        detail: "The invalid verification request was not executed; only plan registration is available on the next turn",
      };
    case "runtime.settings.changed":
      return {
        title: "Run settings updated",
        summary: "Live-safe options apply at the next controller boundary; task routing applies to the next task",
        detail: [
          textValue(payload.mode),
          textValue(payload.approval_mode),
          textValue(payload.auto_plan),
          textValue(payload.reasoning_effort) ? `${textValue(payload.reasoning_policy) || "adaptive"} reasoning · ${textValue(payload.reasoning_effort)}` : "",
        ].filter(Boolean).join(" · "),
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
  if (event.type === "session.start" || (event.type === "user.input" && textValue(event.payload.delivery) === "redirect")) {
    const task = textValue(event.payload.task);
    return <article className="message user timeline-user-message"><div className="message-label"><span>{event.type === "user.input" ? "YOU · REDIRECT" : "YOU"}</span><time>{new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></div><ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{task}</ReactMarkdown></article>;
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
      {(plan.steps || []).map((step: any, index: number) => <li className={step.status} key={step.step_id}>
        <PlanStepMarker index={index} status={step.status} />
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

function PlanStepMarker({ index, status }: { index: number; status: string }) {
  const satisfied = ["completed", "satisfied", "skipped"].includes(status);
  return <b aria-label={satisfied ? `Step ${index + 1} ${status}` : `Step ${index + 1}: ${status}`}>
    {satisfied ? <Check size={13} strokeWidth={3} /> : index + 1}
  </b>;
}

function Inspector({ session, fileView, diffView, refreshDiff, mode, invoke, refresh }: any) {
  const [tab, setTab] = useState<"plan" | "file" | "diff">("plan");
  const plan = session?.plan;
  return <><div className="tabs"><button className={tab === "plan" ? "active" : ""} onClick={() => setTab("plan")}>Plan</button><button className={tab === "file" ? "active" : ""} onClick={() => setTab("file")}>File</button><button className={tab === "diff" ? "active" : ""} onClick={() => { setTab("diff"); void refreshDiff(); }}>Diff</button></div>
    <div className="inspector-content">
      {tab === "plan" && <>{!plan ? <div className="muted-card"><ListChecks size={20} /><p>{mode === "planning" ? "Plan Mode is active. Submit a task to begin read-only inspection." : "No plan artifact for this session."}</p></div> : <div className="plan-card"><div className="plan-status"><strong>{plan.goal}</strong><span>{plan.status}</span></div>{plan.steps?.map((step: any, index: number) => <div className={`plan-step ${step.status}`} key={step.step_id}><PlanStepMarker index={index} status={step.status} /><div><strong>{step.title}</strong><small>{step.operation} · {step.risk}</small></div></div>)}{plan.status === "ready" || plan.status === "stale" ? <PlanControls sessionId={session.session_id} invoke={invoke} refresh={refresh} /> : null}</div>}
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
  return <div className="plan-controls"><div className="plan-buttons"><button disabled={submitting} onClick={() => { setAction("revise"); setInstruction(""); }}>Request revision</button><button disabled={submitting} onClick={() => { setAction("inspect"); setInstruction(""); }}>Continue inspection</button><button disabled={submitting} onClick={() => void invoke(async () => { await api(`/api/v1/sessions/${sessionId}/plan/cancel`, { method: "POST", body: "{}" }); await refresh(); })}>Cancel</button></div>{action && <div className="plan-instruction"><label>{action === "revise" ? "Revision direction" : "What should RepoRivet inspect next?"}<textarea autoFocus value={instruction} onChange={(event) => setInstruction(event.target.value)} /></label><div><button onClick={() => setAction(null)}>Back</button><button className="primary" disabled={!instruction.trim() || submitting} onClick={() => void submitInstruction()}>{submitting ? "Submitting…" : action === "revise" ? "Revise plan" : "Continue planning"}</button></div></div>}</div>;
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
