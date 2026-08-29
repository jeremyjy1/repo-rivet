export type SessionSummary = {
  session_id: string;
  short_id: string;
  name: string;
  task_preview: string;
  status: string;
  updated_at: string;
  step: number;
};

export type Approval = {
  request_id: string;
  state_version: number;
  tool: string;
  risk: string;
  reasons: string[];
  operation: string;
  reads: string[];
  writes: string[];
  deletes: string[];
  effects: string[];
  details: Array<{ label: string; value: string }>;
  operations: string[];
  preview: { kind: "diff" | "content"; title: string; text: string } | null;
  review: {
    recommendation: string;
    risk: string;
    relevance: string;
    reason: string;
    question: string | null;
    unknowns: string[];
    constraints: string[];
  } | null;
  allow_matching_repeats: boolean;
};

export type SessionDetail = SessionSummary & {
  messages: Array<{ role: string; content: string; name?: string; step: number }>;
  workflow_mode: "execute" | "planning" | "plan_ready";
  approval_mode: string | null;
  active_skill: { id: string; version: string } | null;
  plan: Record<string, any> | null;
  verification: Record<string, any>;
  modified_files: string[];
  workspace_revision: number;
  last_event_seq: number;
  run: {
    status: string;
    result: Record<string, any> | null;
    error: string | null;
    pending_approval: Approval | null;
  } | null;
};

export type AgentEvent = {
  seq: number;
  event_id: string;
  type: string;
  timestamp: string;
  payload: Record<string, unknown>;
};

export type EventPage = {
  items: AgentEvent[];
  has_more: boolean;
};

let csrfToken = sessionStorage.getItem("reporivet-csrf") || "";

export async function authenticate(): Promise<void> {
  const params = new URLSearchParams(location.hash.slice(1));
  const token = params.get("bootstrap");
  if (token) {
    const response = await fetch("/api/v1/auth/bootstrap", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    if (!response.ok) throw new Error(await errorText(response));
    const payload = await response.json();
    csrfToken = payload.csrf_token;
    sessionStorage.setItem("reporivet-csrf", csrfToken);
    history.replaceState(null, "", location.pathname + location.search);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch(path, { ...init, headers, credentials: "include" });
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

async function errorText(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}
