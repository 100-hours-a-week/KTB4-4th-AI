const $ = (selector) => document.querySelector(selector);
const main = $("#main");
const state = {
  overview: { runs: [], connections: [], worker: {} },
  fixtures: null,
  service: null,
  serviceError: null,
  page: "overview",
  mode: "needu",
  connectionId: "lmstudio",
  activeRun: null,
  detail: null,
  compare: new Set(),
  compareUsage: new Map(),
  draft: null,
  personas: [],
  selectedPersona: "p01",
  personaCategory: "",
  personaSearch: "",
  chatSessions: [],
  chatDetail: null,
  chatDraft: {},
  chatBusy: false,
  chatPending: "",
  chatMessage: "",
};
const modes = {
  needu: "니쥬 서비스",
  quality: "초기 추출 평가",
  load: "동시성 테스트",
};
const modeLabel = (mode) => modes[mode] || "이전 단일 모델 실험";
const chatStatus = (status) =>
  ({
    active: "대화 중",
    closed: "완료",
    input_locked: "입력 종료",
    review: "분석 완료",
    expired: "만료",
    starting: "생성 중",
    failed: "실패",
  })[status] || status;
const statuses = {
  queued: "대기",
  running: "실행 중",
  completed: "완료",
  failed: "실패",
  cancelled: "취소",
  timed_out: "시간 상한",
  interrupted: "워커 중단",
};
const e = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
const num = (value, digits = 0) =>
  value == null ? "—" : Number(value).toLocaleString("ko-KR", { maximumFractionDigits: digits });
const ms = (value) => (value == null ? "미측정" : `${num(value)} ms`);
const pct = (value) => (value == null ? "미측정" : `${num(value * 100, 1)}%`);
const date = (value) =>
  value
    ? new Date(value).toLocaleString("ko-KR", {
        timeZone: "Asia/Seoul",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";
const active = (run) => ["queued", "running"].includes(run.status);

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((r) => r.msg).join(" · ")
      : body.detail;
    throw new Error(detail || "요청에 실패했습니다.");
  }
  return body;
}

let noticeTimer;
function notice(text, error = false) {
  const node = $("#notice");
  node.textContent = text;
  node.classList.toggle("error", error);
  node.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(
    () => {
      node.hidden = true;
    },
    error ? 10000 : 4500,
  );
}

function pill(status) {
  const cls =
    status === "completed" ? "ok" : status === "running" || status === "queued" ? "running" : "bad";
  return `<span class="pill ${cls}">${e(statuses[status] || status)}</span>`;
}

function heading(eyebrow, title, subtitle, action = "") {
  return `<div class="page-head"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p class="subtitle">${subtitle}</p></div>${action ? `<div class="actions">${action}</div>` : ""}</div>`;
}

function metric(label, value, hint) {
  return `<div class="card metric"><div class="label">${label}</div><div class="value">${value}</div><div class="hint">${hint}</div></div>`;
}

function connectionRows() {
  return state.overview.connections
    .map(
      (c) =>
        `<div class="connection-row"><span class="provider-icon">${c.provider === "ollama" ? "o" : c.provider === "lmstudio" ? "lm" : "↗"}</span><div class="connection-content"><strong>${e(c.name)}</strong><small>${e(c.model || "모델 ID 선택 필요")}</small></div><span class="pill ${c.probe?.ok ? "ok" : c.probe ? "bad" : ""}">${c.probe?.ok ? "연결 확인" : c.probe ? "미연결" : "미검사"}</span></div>`,
    )
    .join("");
}

function runTable(runs, withCheck = false) {
  if (!runs.length)
    return `<div class="empty"><span class="empty-icon">◎</span><strong>아직 실행한 실험이 없어요</strong>모델을 연결하고 첫 테스트를 시작해보세요.</div>`;
  return `<div class="table-wrap"><table><thead><tr>${withCheck ? "<th>비교</th>" : ""}<th>실험 / 모델</th><th>상태</th><th>완료 / 계획</th><th>지연 p95</th><th>실행 시각</th></tr></thead><tbody>${runs.map((r) => `<tr>${withCheck ? `<td><input type="checkbox" data-compare="${e(r.id)}" aria-label="${e(r.connection.name)} ${date(r.created_at)} ${r.id.slice(0, 8)} 비교 선택" ${state.compare.has(r.id) ? "checked" : ""}></td>` : ""}<td><button class="link" data-action="detail" data-id="${e(r.id)}">${e(modeLabel(r.spec.mode))}</button><small>${e(r.connection.model)} · ${e(r.connection.environment)}</small>${r.manifest.pipeline ? `<small>${e(r.manifest.pipeline.recommendation_stage)} · ${r.id.slice(0, 8)}</small>` : ""}</td><td>${pill(r.status)}</td><td>${r.completed ?? r.samples?.length ?? 0} / ${r.spec.requests}</td><td>${ms(r.summary?.latency_p95_ms)}</td><td>${date(r.created_at)}</td></tr>`).join("")}</tbody></table></div>`;
}

function runStatusChart() {
  const runs = state.overview.runs;
  const groups = [
    ["실행 완료", runs.filter((r) => r.status === "completed").length, "#3b82f6"],
    ["진행·대기", runs.filter(active).length, "#14b8a6"],
    [
      "실패·중단",
      runs.filter((r) => ["failed", "timed_out", "interrupted"].includes(r.status)).length,
      "#f59e0b",
    ],
    ["취소", runs.filter((r) => r.status === "cancelled").length, "#94a3b8"],
  ];
  let offset = 0;
  const circles = groups
    .map(([label, count, color]) => {
      const size = runs.length ? (count / runs.length) * 100 : 0;
      const circle = `<circle cx="80" cy="80" r="60" fill="none" stroke="${color}" stroke-width="18" pathLength="100" stroke-dasharray="${size} ${100 - size}" stroke-dashoffset="${-offset}" transform="rotate(-90 80 80)"/>`;
      offset += size;
      return circle;
    })
    .join("");
  return `<section class="card"><div class="card-head"><h2>실험 상태 분포</h2><small>실행 상태 기준</small></div><svg class="donut" viewBox="0 0 160 160" role="img" aria-label="총 ${runs.length}개 실험의 실행 상태"><circle cx="80" cy="80" r="60" fill="none" stroke="#edf2f8" stroke-width="18"/>${circles}<text x="80" y="74" text-anchor="middle" font-size="11">기록된 실험</text><text x="80" y="97" text-anchor="middle" font-size="21" font-weight="600">${runs.length}건</text></svg><div class="chart-legend">${groups.map(([label, count, color]) => `<div class="legend-line"><span><svg width="8" height="8" aria-hidden="true"><circle cx="4" cy="4" r="3" fill="${color}"/></svg> ${label}</span><strong>${count}건</strong></div>`).join("")}</div></section>`;
}

function avatar(persona, index = 0) {
  return `<span class="avatar tone-${index % 5}">${e(persona?.initials?.slice(0, 1) || "N")}</span>`;
}

function personaGuide(p) {
  if (!p) return "";
  return `<div class="persona-top">${avatar(p, state.personas.indexOf(p))}<div><strong>${e(p.name)} <span class="muted">${p.age}세</span></strong><small>${e(p.occupation)} · ${e(p.tag)}</small></div></div><p>${e(p.context)}</p>${[
    ["페르소나 구분", p.group],
    ["중요하게 여기는 것", p.values],
    ["미래 계획", p.future],
    ["추천 검토 상황", p.situation],
    ["생활·공간·예산", p.lifestyle],
    ["대화 말투", p.speech_style],
  ]
    .filter(([, text]) => text)
    .map(([label, text]) => `<div class="detail-section"><h3>${label}</h3><p>${e(text)}</p></div>`)
    .join("")}${[
    ["기대 추출 · 사람이 검토", p.expected],
    ["추천 쿼리에 반영할 소재", p.query_targets],
    ["추천 후보 예시 · 실제 검색 결과 아님", p.suggested_items],
    ["제외·오추출 주의", p.avoid],
  ]
    .map(
      ([label, values]) =>
        `<div class="detail-section"><h3>${label}</h3><ul>${values.map((v) => `<li>${e(v)}</li>`).join("")}</ul></div>`,
    )
    .join(
      "",
    )}${p.expected_queries?.length ? `<div class="detail-section"><h3>기대 추천 쿼리 · 조건 유사도 기준</h3><ul>${p.expected_queries.map((q) => `<li><strong>${e(q.space)} · ${e(q.mode)}</strong> ${e(q.text)}<br><small>확인할 조건: ${e(q.concepts.map((group) => group.join(" / ")).join(" · "))}</small></li>`).join("")}</ul><p class="muted">실행 결과에서 가장 가까운 실제 쿼리와 반영·누락된 조건을 확인합니다.</p></div>` : ""}<details class="sample"><summary>고정 대화 시나리오 · ${p.turns.length}턴</summary><div class="sample-body"><ol>${p.turns.map((t) => `<li>${e(t)}</li>`).join("")}</ol></div></details>`;
}

function personaCards() {
  const filtered = state.personas.filter(
    (p) =>
      (!state.personaCategory || p.category === state.personaCategory) &&
      `${p.name} ${p.occupation} ${p.context} ${p.lifestyle || ""} ${p.values} ${p.future} ${p.tag} ${p.query_targets.join(" ")}`.includes(state.personaSearch),
  );
  return `<div class="persona-grid">${
    filtered
      .map((p) => {
        const runs = state.overview.runs.filter((r) => r.spec.persona_id === p.id);
        return `<button class="card persona-card ${state.selectedPersona === p.id ? "selected" : ""}" data-action="persona" data-id="${p.id}" aria-label="${e(p.name)} 페르소나 상세" aria-pressed="${state.selectedPersona === p.id}"><div class="persona-top">${avatar(p, state.personas.indexOf(p))}<div><strong>${e(p.name)}</strong><small>${p.age}세 · ${e(p.occupation)}</small></div></div><p>${e(p.context)}</p><div class="persona-tags">${p.group ? `<span>${e(p.group)}</span>` : ""}<span>${e(p.category)}</span><span>${e(p.tag)}</span></div><div class="persona-foot"><span>시나리오 ${p.turns.length}턴</span><span>${runs.length ? `${runs.length}회 실행` : "미실행"}</span></div></button>`;
      })
      .join("") || `<div class="empty">검색에 맞는 페르소나가 없습니다.</div>`
  }</div>`;
}

function overviewPage() {
  const { runs, connections } = state.overview;
  const count = (status) => runs.filter((r) => r.status === status).length;
  const stats = [
    ["페르소나", state.personas.length],
    ["모델 연결 확인", connections.filter((c) => c.probe?.ok).length],
    ["누적 실험 · 최근 100개", runs.length],
    ["진행 중", runs.filter(active).length],
    ["실행 완료", count("completed")],
    [
      "실패·중단",
      runs.filter((r) => ["failed", "interrupted", "timed_out"].includes(r.status)).length,
    ],
  ];
  const p = state.personas.find((p) => p.id === state.selectedPersona);
  return (
    heading(
      "NEED U / MODEL OPERATIONS",
      "모델 검증 대시보드",
      "실제 대화와 페르소나 시나리오로, 니쥬에 맞는 모델을 선택하세요.",
      `<a class="btn" href="#chat">대화 테스트</a><a class="btn primary" href="#experiments">새 실험 시작</a>`,
    ) +
    `<div class="filters"><div class="field"><label for="persona-category">평가 주제</label><select id="persona-category"><option value="">전체 주제</option>${[...new Set(state.personas.map((p) => p.category))].map((category) => `<option ${category === state.personaCategory ? "selected" : ""}>${e(category)}</option>`).join("")}</select></div><div class="field search-field"><label for="persona-search">페르소나 검색</label><input id="persona-search" value="${e(state.personaSearch)}" placeholder="이름, 가치관, 미래 계획으로 검색"></div><a class="btn" href="#history">실험 이력</a></div>` +
    `<div class="stat-strip">${stats.map(([label, value]) => `<div><small>${label}</small><strong>${value}</strong></div>`).join("")}</div>` +
    `<div class="dashboard-layout"><section class="card"><div class="card-head"><h2>페르소나별 검증</h2><span class="pill">합성 페르소나 ${state.personas.length}개</span></div><div id="persona-cards">${personaCards()}</div><p class="persona-help">실행 횟수와 상태는 실제 이력 기준입니다. 기대 쿼리의 조건 유사도와 제외 조건을 실행 결과에서 확인하세요.</p></section><aside>${runStatusChart()}<section class="card"><div class="card-head"><h2>모델 연결</h2><a class="link" href="#connections">관리</a></div>${connectionRows()}<div class="note">${state.overview.worker.online ? "시나리오 워커 실행 중" : "시나리오 실행은 워커 시작 필요"}<br>직접 채팅은 관리 API만으로 실행합니다.</div></section><section class="card"><div class="card-head"><h2>최근 실행</h2><a class="link" href="#history">더 보기</a></div>${
      runs
        .slice(0, 4)
        .map(
          (r) =>
            `<div class="connection-row"><div class="connection-content"><strong>${e(state.personas.find((p) => p.id === r.spec.persona_id)?.name || modeLabel(r.spec.mode))}</strong><small>${date(r.created_at)}</small></div>${pill(r.status)}</div>`,
        )
        .join("") || `<div class="empty">아직 실행한 실험이 없습니다.</div>`
    }</section><section class="card"><h2>운영 자원</h2><p class="persona-help">GPU·VRAM·콜드스타트·비용은 수집기 연결 전까지 미수집입니다.</p></section></aside></div>` +
    `<section class="card persona-detail"><div class="card-head"><h2>선택한 페르소나 · ${e(p?.name || "")}</h2><div class="actions"><button class="btn" data-action="persona-chat" data-id="${e(p?.id)}">이 페르소나로 직접 대화</button><button class="btn primary" data-action="persona-run" data-id="${e(p?.id)}">시나리오로 실험</button></div></div>${personaGuide(p)}</section>`
  );
}

function personasPage() {
  const p = state.personas.find((p) => p.id === state.selectedPersona);
  return (
    heading(
      "SCENARIO LIBRARY",
      "페르소나 시나리오",
      `기존 인물과 20~26세·27~34세 추가 인물을 포함한 ${state.personas.length}명의 생활·취향과 기대 추천 쿼리를 검토합니다.`,
    ) +
    `<div class="dashboard-layout"><div>${personaCards()}</div><section class="card persona-detail">${personaGuide(p)}<div class="actions"><button class="btn" data-action="persona-chat" data-id="${p.id}">직접 대화</button><button class="btn primary" data-action="persona-run" data-id="${p.id}">시나리오 실험</button></div></section></div>`
  );
}

function chatPage() {
  const session = state.chatDetail;
  const draft = state.chatDraft;
  const persona = session?.persona || state.personas.find((p) => p.id === draft.persona_id);
  const disabled = state.chatBusy ? "disabled" : "";
  const setup = `<section class="card"><h2>새 대화의 모델 설정</h2><p class="persona-help">등록한 Ollama·LM Studio 연결을 응답·추출에 각각 선택하세요. 세션 시작 시 설정을 고정합니다.</p><form id="chat-create-form">${field("대화 이름", "title", draft.title || "로컬 모델 대화 테스트", "text", 'maxlength="80" required')}${selectField("응답·요약 모델", "response_connection_id", roleOptions("response"), draft.response_connection_id || "")}${selectField("취향 추출 모델", "extraction_connection_id", roleOptions("extraction"), draft.extraction_connection_id || "")}${selectField(
    "판단 모델",
    "judgment_mode",
    [
      ["environment", "환경변수 기본"],
      ["disabled", "끄기 · 로컬 직접 추출"],
    ],
    draft.judgment_mode || "environment",
  )}${selectField("페르소나 가이드", "persona_id", [["", "자유 대화"], ...state.personas.map((p) => [p.id, `${p.name} · ${p.tag}`])], draft.persona_id || "")}<button class="btn primary" type="submit" ${disabled}>${state.chatBusy ? "모델 연결·첫 인사 생성 중…" : "대화 세션 시작"}</button><p class="persona-help">환경변수 기본값이 원격 모델이면 해당 서버를 호출합니다. 로컬 전용 테스트는 두 모델의 로컬 연결과 판단 끄기를 선택하세요.</p></form></section>`;
  const last = session?.traces?.at(-1);
  const signals = Object.entries(session?.pipeline?.profile || {})
    .filter(([_, value]) => Array.isArray(value))
    .flatMap(([key, values]) =>
      values.map((v) => ({
        field: key,
        value: typeof v === "string" ? v : v.value || JSON.stringify(v),
      })),
    );
  const inspector = session
    ? `<section class="card inspector"><div class="card-head"><h2>실시간 추출·쿼리</h2><span class="pill">${session.turn_count}턴</span></div><div class="signal-list">${signals.map((s) => `<span>${e(s.field)} · ${e(s.value)}</span>`).join("") || `<p class="persona-help">아직 추출된 취향이 없습니다.</p>`}</div>${(session.pipeline?.queries || []).map((q) => `<div class="note"><strong>${e(q.space)}</strong><br>${e(q.text)}</div>`).join("")}${queryReviewHTML(session.pipeline?.query_review)}${session.last_extraction_failed ? `<div class="note error">답변은 생성됐지만 추출이 실패했습니다. 원본 호출을 확인하세요.</div>` : ""}${session.error ? `<div class="note error">${e(session.error)}</div>` : ""}<p class="persona-help">마지막 요청 ${ms(last?.latency_ms)} · 실제 임베딩·DB는 호출하지 않습니다.</p><details class="sample"><summary>프로필·검색·임베딩 입력</summary><div class="sample-body"><pre>${e(JSON.stringify(session.pipeline, null, 2))}</pre></div></details><details class="sample"><summary>원본 모델 호출 · 프롬프트·스키마</summary><div class="sample-body"><pre>${e(JSON.stringify(session.traces, null, 2))}</pre></div></details>${persona ? `<details class="sample"><summary>${e(persona.name)} · 검토 가이드</summary><div class="sample-body persona-detail">${personaGuide(persona)}</div></details>` : ""}</section>`
    : `<section class="card inspector"><h2>직접 대화하면서 검증</h2><p class="persona-help">매 턴 실제 니쥬 응답·취향 추출·프로필 병합·추천 쿼리를 확인합니다. 새로고침 후에도 로컬 DB에 저장된 대화를 다시 열 수 있습니다.</p>${persona ? personaGuide(persona) : `<p class="persona-help">페르소나를 선택하면 상황·시나리오·기대 결과를 참고할 수 있어요.</p>`}</section>`;
  return (
    heading(
      "INTERACTIVE SERVICE CHAT",
      "대화 테스트",
      "실제 니쥬 프롬프트와 서비스 API 흐름으로 로컬 모델과 직접 대화합니다.",
      `<button class="btn" data-action="new-chat" ${disabled}>+ 새 대화</button>`,
    ) +
    `<div class="chat-layout"><aside class="card"><div class="card-head"><h2>대화 세션</h2><span class="pill">${state.chatSessions.length}</span></div>${state.chatSessions.map((s) => `<button class="session-link ${s.id === session?.id ? "selected" : ""}" data-action="open-chat" data-id="${s.id}" ${disabled}>${e(s.title)}<small>${e(s.config.response.model)} · ${s.turn_count}턴 · ${e(chatStatus(s.status))}</small></button>`).join("") || `<div class="empty">아직 대화가 없습니다.</div>`}</aside><div>${session ? `<section class="card"><div class="card-head"><div><h2>${e(session.title)}</h2><small>${e(session.config.response.model)} / ${e(session.config.extraction.model)}</small></div><span class="pill ${session.status === "active" ? "running" : ""}">${e(chatStatus(session.status))}</span></div><div class="chat-feed" role="log" aria-label="모델과의 대화" aria-live="polite">${session.messages.map((m) => `<div class="chat-message ${m.role}"><span class="avatar">${m.role === "user" ? "나" : "N"}</span><div class="chat-bubble">${e(m.content)}</div></div>`).join("")}${state.chatPending ? `<div class="chat-message user"><span class="avatar">나</span><div class="chat-bubble">${e(state.chatPending)}</div></div><div class="chat-pending">응답 생성·취향 추출 중…</div>` : ""}</div>${session.analysis ? `<div class="note success">${e(session.analysis.summary || "프로필 분석 완료 · 요약 미생성")}</div>` : ""}<form id="chat-message-form" class="chat-composer"><label for="chat-message">메시지</label><textarea id="chat-message" name="message" maxlength="500" rows="3" required placeholder="경험, 가치관, 미래 계획… 어떤 이야기든 시작해보세요." ${disabled} ${session.status !== "active" ? "disabled" : ""}>${e(state.chatMessage)}</textarea><div class="actions"><span class="chat-status">${session.turn_count} / 20턴 · 500자까지</span><button type="submit" class="btn primary" ${disabled} ${session.status !== "active" ? "disabled" : ""}>${state.chatBusy ? "처리 중…" : "보내기"}</button></div></form><div class="actions"><button class="btn small" data-action="analyze-chat" data-id="${session.id}" ${disabled} ${!["active", "input_locked"].includes(session.status) ? "disabled" : ""}>분석·대화 완료</button><button class="btn small danger" data-action="delete-chat" data-id="${session.id}" ${disabled}>대화 삭제</button></div><p class="persona-help">${session.status !== "active" ? "서비스 종료·만료 정책에 따라 추가 입력이 잠겼습니다." : "질문과 답변을 직접 이어가세요. 페르소나 시나리오는 참고이며 자동으로 발화를 보내지 않습니다."}</p></section>` : setup}</div>${inspector}</div>`
  );
}

function field(label, name, value = "", type = "text", extra = "", hint = "") {
  return `<div class="field"><label for="${name}">${label}</label><input id="${name}" name="${name}" type="${type}" value="${e(value)}" ${extra}>${hint ? `<small>${hint}</small>` : ""}</div>`;
}

function connectionsPage() {
  const connection = state.overview.connections.find((c) => c.id === state.connectionId);
  const c = connection || {
    name: "",
    provider: "openai",
    environment: "runpod",
    base_url: "",
    model: "",
    api_key_env: "",
  };
  return (
    heading(
      "ENDPOINT REGISTRY",
      "모델 연결",
      "모델을 실행한 서버의 API 주소와 실제 모델 ID를 등록하세요.",
      `<div class="actions"><button class="btn" data-action="hf-connection">Hugging Face API</button><button class="btn" data-action="openrouter-connection">OpenRouter API</button><button class="btn" data-action="new-connection">+ 연결 추가</button></div>`,
    ) +
    `<div class="grid-two"><section class="card"><div class="card-head"><h2>${connection ? e(c.name) : "새 연결"}</h2><span class="pill">${e(c.environment)}</span></div><form id="connection-form">` +
    field("연결 이름", "name", c.name, "text", 'required maxlength="80"') +
    `<div class="form-grid"><div class="field"><label for="provider">런타임</label><select id="provider" name="provider">${[
      ["lmstudio", "LM Studio"],
      ["ollama", "Ollama"],
      ["huggingface", "Hugging Face Inference Providers"],
      ["openai", "OpenAI 호환 서버"],
    ]
      .map(
        ([v, label]) =>
          `<option value="${v}" ${c.provider === v ? "selected" : ""}>${label}</option>`,
      )
      .join(
        "",
      )}</select></div><div class="field"><label for="environment">실행 환경</label><select id="environment" name="environment">${[
      ["local", "로컬"],
      ["runpod", "RunPod"],
      ["other", "기타 원격"],
    ]
      .map(
        ([v, label]) =>
          `<option value="${v}" ${c.environment === v ? "selected" : ""}>${label}</option>`,
      )
      .join("")}</select></div></div>` +
    field(
      "API Base URL",
      "base_url",
      c.base_url,
      "url",
      'required placeholder="http://127.0.0.1:1234/v1"',
      "OpenAI 호환 API의 /v1까지 입력",
    ) +
    field(
      "모델 ID",
      "model",
      c.model,
      "text",
      'list="model-options" maxlength="200" placeholder="연결 검사 후 목록에서 선택"',
    ) +
    `<datalist id="model-options">${(c.probe?.models || []).map((m) => `<option value="${e(m)}"></option>`).join("")}</datalist>` +
    `<div class="form-grid">${field("모델 revision", "revision", c.revision, "text", 'maxlength="200" placeholder="선택 사항"')}${field("정밀도 / 양자화", "precision", c.precision, "text", 'maxlength="100" placeholder="예: GGUF Q4 / BF16"')}</div>` +
    field(
      "하드웨어 / 실행 메모",
      "hardware",
      c.hardware,
      "text",
      'maxlength="200" placeholder="예: Mac Apple Silicon / A40 48GB"',
    ) +
    field(
      "API key 환경변수 이름",
      "api_key_env",
      c.api_key_env,
      "text",
      'placeholder="예: AI_LAB_RUNPOD_KEY"',
      "선택 사항 · 키 값은 관리 서버와 워커의 환경변수로 설정",
    ) +
    `<div class="actions"><button class="btn primary" type="submit">연결 저장</button>${connection ? `<button class="btn" type="button" data-action="probe" data-id="${e(c.id)}">연결 검사 · 모델 목록</button>` : ""}</div><div id="probe-result">${probeResult(c)}</div></form></section>` +
    `<div><section class="card"><div class="card-head"><h2>등록된 연결</h2></div>${state.overview.connections.map((entry) => `<div class="connection-row"><div class="connection-content"><strong>${e(entry.name)}</strong><small>${e(entry.base_url)}</small></div><button class="btn small" data-action="edit-connection" data-id="${e(entry.id)}">선택</button></div>`).join("")}</section>${huggingfaceGuide(c)}<section class="card prose"><h2>모델 실행 → API 연결 → 테스트</h2><div class="steps"><div class="step"><div><h3>모델 API 준비</h3><p>로컬은 LM Studio·Ollama 서버를 실행하고, 호스팅 API는 토큰과 사용 가능한 모델을 준비하세요.</p></div></div><div class="step"><div><h3>주소를 저장하고 연결 검사</h3><p>서버가 반환하는 모델 ID를 선택하고 다시 저장하세요.</p></div></div><div class="step"><div><h3>같은 조건으로 후보 비교</h3><p>제공업체와 실행 조건을 함께 기록하세요. 호스팅 API 속도는 Runpod GPU 성능과 구분합니다.</p></div></div></div></section></div></div>`
  );
}

function huggingfaceGuide(connection) {
  if (connection.provider !== "huggingface") return "";
  const selected = connection.probe?.model_metadata?.find((m) => m.id === connection.model);
  return `<section class="card prose"><h2>Hugging Face 연결 준비</h2><ol><li><a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">Access Token 발급</a>: Fine-grained → Make calls to Inference Providers 권한.</li><li>프로젝트 .env에 HF_TOKEN을 설정하세요. 이 폼에는 키 값 대신 HF_TOKEN이라는 이름만 입력합니다.</li><li><a href="https://huggingface.co/settings/billing" target="_blank" rel="noopener noreferrer">추론 크레딧·결제 설정</a>을 확인하세요.</li><li>연결 검사로 제공 중인 모델·업체 목록을 조회하고 선택한 ID를 다시 저장하세요.</li></ol><p>비교할 때 모델:제공업체 형태로 고정하세요. 기본 예시는 Qwen/Qwen3.5-9B:deepinfra입니다. 공개 모델 목록 조회는 토큰 유효성이나 추론 성공을 확인하지 않습니다.</p>${selected ? `<div class="note">제공업체: ${e(selected.provider)} · 구조화 출력: ${selected.supports_structured_output === true ? "지원" : selected.supports_structured_output === false ? "미지원" : "미확인"} · 컨텍스트: ${num(selected.context_length)}토큰<br>제공업체가 공개한 지원 정보이며 실제 니쥬 스키마는 실행으로 확인합니다.</div>` : ""}<p>니쥬 응답·추출 모델에 이 연결을 선택해 테스트합니다. 페르소나 사용자 역할은 기존 Gemini를 유지합니다. 호스팅 모델의 가중치 revision·양자화·GPU는 별도 확인 전까지 미확인입니다.</p></section>`;
}

function probeResult(connection) {
  if (!connection.probe) return "";
  const p = connection.probe;
  if (connection.provider === "huggingface") {
    return `<div class="note ${p.credential_configured ? "" : "error"}">API 모델 목록 ${p.models.length}개 · HF_TOKEN ${p.credential_configured ? "설정됨" : "미설정"}<br>${p.error ? e(p.error) : "토큰 유효성·과금·JSON Schema 호환성은 실제 호출로 확인하세요."}</div>`;
  }
  return `<div class="note ${p.ok ? "success" : "error"}">${p.ok ? `API 연결 확인 · ${ms(p.latency_ms)} · 모델 ${p.models.length}개` : e(p.error)}<br>${date(connection.checked_at)} 검사 · 모델 적재와 추론 성공은 실제 호출로 확인하세요.</div>`;
}

function experimentsPage() {
  if (state.mode === "needu") return needuPage();
  const draft = state.draft || {};
  const mode = state.mode;
  return (
    heading(
      "EXPERIMENT RUNNER",
      "테스트 실행",
      "모델 출력부터 품질과 동시 처리 성능까지, 실행 조건과 함께 기록합니다.",
    ) +
    `<div class="grid-even"><section class="card"><div class="mode-tabs" role="group" aria-label="테스트 종류">${Object.entries(
      modes,
    )
      .map(
        ([key, label]) =>
          `<button data-action="mode" data-mode="${key}" class="${mode === key ? "selected" : ""}" aria-pressed="${mode === key}">${label}</button>`,
      )
      .join(
        "",
      )}</div><form id="run-form"><div class="field"><label for="connection_id">테스트할 모델 연결</label><select id="connection_id" name="connection_id">${state.overview.connections.map((c) => `<option value="${e(c.id)}" ${c.id === (draft.connection_id || state.connectionId) ? "selected" : ""}>${e(c.name)} · ${e(c.model || "모델 미선택")}</option>`).join("")}</select></div>` +
    (mode === "quality"
      ? `<div class="note">취향 추출 초기 사례 ${state.fixtures?.cases.length || 8}개 · ${e(state.fixtures?.version)}<br>동일한 추출 지시와 JSON Schema를 사용합니다. 니쥬 전체 서비스 평가가 아닙니다.</div>`
      : `<div class="field"><label for="system">시스템 프롬프트</label><textarea id="system" name="system" maxlength="12000" rows="3">${e(draft.system ?? "너는 니쥬의 대화 도우미다. 한국어로 자연스럽고 간결하게 답하라.")}</textarea></div><div class="field"><label for="prompt">사용자 입력</label><textarea id="prompt" name="prompt" maxlength="12000" rows="5" required placeholder="캠핑하면서 커피 내려 마시는 걸 좋아해. 나에게 질문 하나 해줘.">${e(draft.prompt ?? "")}</textarea></div>`) +
    `<div class="form-grid">${field("Temperature", "temperature", draft.temperature ?? 0.2, "number", 'min="0" max="2" step="0.1" required')}${field("최대 출력 토큰", "max_tokens", draft.max_tokens ?? 256, "number", 'min="1" max="512" required')}</div>` +
    (mode === "load"
      ? `<div class="form-grid">${field("총 요청 수", "requests", draft.requests ?? 10, "number", 'min="1" max="100" required')}${field("동시 요청", "concurrency", draft.concurrency ?? 1, "number", 'min="1" max="8" required')}</div>`
      : "") +
    `<div class="form-grid">${field("실험 시간 상한 · 초", "duration_seconds", draft.duration_seconds ?? 120, "number", 'min="5" max="300" required')}${field("요청 시간 상한 · 초", "timeout_seconds", draft.timeout_seconds ?? 60, "number", 'min="1" max="120" required')}</div>` +
    field(
      "실험 메모",
      "note",
      draft.note,
      "text",
      'maxlength="1000" placeholder="예: 웜 상태 / 캐시 조건 / 비교 목적"',
    ) +
    `<div class="note" id="run-budget"></div><div class="actions"><button type="submit" class="btn primary">${mode === "quality" ? "고정 평가 실행" : mode === "load" ? "부하 테스트 시작" : "모델 호출 실행"} ↗</button><span class="muted subtitle">${state.overview.worker.online ? "워커 준비됨" : "워커 시작 필요"}</span></div><small class="muted">합성 입력과 출력이 로컬 실험 이력에 저장됩니다.</small></form></section><div><section class="card"><div class="card-head"><h2>${mode === "quality" ? "평가 사례" : mode === "load" ? "측정할 지표" : "출력과 측정 결과"}</h2><span class="pill">${mode === "quality" ? "SMOKE TEST" : "REAL REQUEST"}</span></div>${mode === "quality" ? `<div class="table-wrap"><table><thead><tr><th>사례</th><th>입력</th></tr></thead><tbody>${(state.fixtures?.cases || []).map((c) => `<tr><td>${e(c.label)}</td><td>${e(c.input)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="steps"><div class="step"><div><h3>완료시간과 첫 출력 청크</h3><p>네트워크를 포함한 클라이언트 측정값입니다. 서버 TTFT와 구분합니다.</p></div></div><div class="step"><div><h3>출력 토큰과 성공률</h3><p>서버가 usage를 보내지 않으면 토큰은 미측정으로 남깁니다.</p></div></div><div class="step"><div><h3>개별 출력과 오류</h3><p>실행 후 이력에서 실패 요청을 열어 원인을 확인하세요.</p></div></div></div>`}</section><section class="card prose"><h2>테스트 범위</h2><p>${mode === "load" ? "고정 동시성으로 다음 요청을 이어 보내는 제한된 부하 테스트입니다. 실제 사용자 유입률, 두 모델의 GPU 경쟁, 니쥬 서비스 전체 경로는 후속 검증이 필요합니다." : mode === "quality" ? "형식 준수와 내용 정답 일치를 각각 측정합니다. 정확한 명사 집합을 비교하므로 유사어도 불일치로 표시될 수 있습니다. 실패 원문을 검토하세요." : "입력한 프롬프트를 선택한 모델에 보냅니다. 대화 자연스러움은 출력 원문으로 판단하고, 후보 비교는 동일 조건으로 반복하세요."}</p><p>로컬 속도는 RunPod GPU의 운영 성능을 보장하지 않습니다.</p></section></div></div>`
  );
}

function selectField(label, name, options, selected, hint = "") {
  return `<div class="field"><label for="${name}">${label}</label><select id="${name}" name="${name}">${options.map(([value, text]) => `<option value="${e(value)}" ${value === selected ? "selected" : ""}>${e(text)}</option>`).join("")}</select>${hint ? `<small>${hint}</small>` : ""}</div>`;
}

function roleOptions(role) {
  const current = state.service?.[role];
  return [
    ["", `환경변수 기본 · ${current?.model || "미설정"}`],
    ...state.overview.connections.map((c) => [c.id, `${c.name} · ${c.model || "모델 미선택"}`]),
  ];
}

function servicePreview(judgment) {
  if (!state.service)
    return `<div class="note error">${e(state.serviceError || "서비스 설정을 읽지 못했습니다.")}</div>`;
  const messages = state.service.opening_messages || [];
  const judged = judgment !== "disabled" && !!state.service.judgment.base_url;
  const extraction = state.service.extraction_prompts[judged ? "judgment" : "direct"];
  return `<div class="note">${e(state.service.source)}<br>취향 추출 경로: ${judged ? "후보 추출 → System One 판단" : "직접 구조화 추출"}</div><details class="sample"><summary>실제 대화 시스템 프롬프트</summary><div class="sample-body"><pre>${e(
    messages
      .filter((m) => m.role === "system")
      .map((m) => m.content)
      .join("\n\n"),
  )}</pre><small class="muted">시작 인사 예시입니다. 소재·시간·턴 지시는 실행 시 서비스가 조립하며 실제 요청도 이력에 기록합니다.</small></div></details><details class="sample"><summary>실제 추출 프롬프트 · JSON Schema</summary><div class="sample-body"><pre>${e(extraction.system)}</pre><pre>${e(JSON.stringify(extraction.schema, null, 2))}</pre></div></details><details class="sample"><summary>환경변수 기본 모델 · 임베딩 설정</summary><div class="sample-body"><pre>${e(JSON.stringify({ response: state.service.response, extraction: state.service.extraction, judgment: state.service.judgment, embedding: state.service.embedding, catalog_configured: state.service.catalog_configured }, null, 2))}</pre></div></details>`;
}

function needuPage() {
  const draft = state.draft || {};
  const judgment = draft.judgment_mode || "environment";
  const simulated = draft.dialogue_mode === "simulated";
  return (
    heading(
      "NEED U / SERVICE PIPELINE",
      "니쥬 서비스 테스트",
      "서비스의 실제 프롬프트·추출·프로필 병합·추천 쿼리 생성 경로를 실행합니다.",
      `<button class="btn" data-action="service-refresh">환경 설정 다시 읽기</button>`,
    ) +
    (draft.persona_id
      ? `<section class="card persona-detail"><div class="card-head"><h2>시나리오 · ${e(state.personas.find((p) => p.id === draft.persona_id)?.name)}</h2><a class="btn small" href="#personas">다른 페르소나 선택</a></div><p class="persona-help">${simulated ? "Gemini가 니쥬의 답변에 따라 발화를 생성합니다. 실제 공개된 정보만 추출 평가에 사용하세요." : "발화는 편집할 수 있습니다. 아래 기대 결과는 원본 시나리오의 수동 검토 기준입니다."}</p><details class="sample"><summary>상황·기대 추출·추천 쿼리 보기</summary><div class="sample-body">${personaGuide(state.personas.find((p) => p.id === draft.persona_id))}</div></details></section>`
      : `<div class="note">페르소나 10개의 고정 시나리오도 선택할 수 있어요. <a href="#personas">페르소나 선택 →</a></div>`) +
    `<div class="grid-even"><section class="card"><div class="mode-tabs" role="group" aria-label="테스트 종류">${Object.entries(
      modes,
    )
      .map(
        ([key, label]) =>
          `<button data-action="mode" data-mode="${key}" class="${key === "needu" ? "selected" : ""}" aria-pressed="${key === "needu"}">${label}</button>`,
      )
      .join("")}</div><form id="run-form">` +
    selectField(
      "사용자 대화 방식",
      "dialogue_mode",
      [
        ["simulated", "Gemini 페르소나 자동 대화"],
        ["fixed", "고정 발화 · 동일 입력 비교"],
      ],
      simulated ? "simulated" : "fixed",
      "자동 대화는 니쥬의 답변에 맞춰 사용자 발화를 생성합니다. 합성 사용자이며 실제 사람 평가와 구분합니다.",
    ) +
    selectField(
      "페르소나",
      "persona_id",
      [["", "페르소나 선택"], ...state.personas.map((p) => [p.id, `${p.name} · ${p.tag}`])],
      draft.persona_id || "",
    ) +
    selectField(
      "응답·요약 모델",
      "response_connection_id",
      roleOptions("response"),
      draft.response_connection_id || "",
      "RESPONSE_MODEL_BASE_URL / RESPONSE_MODEL_NAME 기본값 또는 등록한 로컬 모델",
    ) +
    selectField(
      "취향 추출 모델",
      "extraction_connection_id",
      roleOptions("extraction"),
      draft.extraction_connection_id || "",
      "EXTRACTION_MODEL_BASE_URL / EXTRACTION_MODEL_NAME 기본값 또는 등록한 로컬 모델",
    ) +
    selectField(
      "취향 판단 경로",
      "judgment_mode",
      [
        [
          "environment",
          state.service?.judgment.base_url
            ? "환경변수 기본 · 판단 모델 사용"
            : "환경변수 기본 · 직접 추출",
        ],
        ["disabled", "판단 모델 끄기 · 직접 구조화 추출"],
      ],
      judgment,
      "JUDGMENT_MODEL_BASE_URL이 설정되어 있으면 후보 추출 후 판단 모델을 호출합니다.",
    ) +
    selectField(
      "추천 실행 범위",
      "recommendation_stage",
      [
        ["queries", "실제 추천 쿼리 생성까지"],
        ["embedding", "쿼리 생성 + 실제 임베딩 호출"],
        ["recommendation", "실제 임베딩 + 카탈로그 검색·랭킹"],
      ],
      draft.recommendation_stage || "queries",
      "EMBEDDING_* 설정을 사용합니다. 상품 추천에는 카탈로그 DB 연결이 필요합니다.",
    ) +
    (simulated
      ? field(
          "자동 대화 최대 턴",
          "max_turns",
          draft.max_turns || 10,
          "number",
          'min="1" max="20" required',
          `사용자 역할: ${state.service?.persona?.model || "Gemini 환경 설정 필요"} · 각 발화 500자 이하 · 사용자가 대화를 마치면 조기 종료`,
        )
      : `<div class="field"><label for="prompt">고정 사용자 발화 · 한 줄에 한 턴</label><textarea id="prompt" name="prompt" maxlength="12000" rows="6" required>${e(draft.prompt || "")}</textarea><small>최대 20턴. 같은 입력으로 추출·쿼리를 비교합니다.</small></div>`) +
    field(
      "실험 전체 시간 상한 · 초",
      "duration_seconds",
      draft.duration_seconds || 300,
      "number",
      'min="5" max="1800" required',
      "응답·추출·요약의 개별 시간 상한과 출력 토큰은 실제 서비스 환경변수를 사용",
    ) +
    field(
      "실험 메모",
      "note",
      draft.note,
      "text",
      'maxlength="1000" placeholder="비교 목적·페르소나·기대 결과"',
    ) +
    `<div class="note" id="run-budget"></div><div class="actions"><button type="submit" class="btn primary">니쥬 파이프라인 실행 ↗</button><span class="muted subtitle">${state.overview.worker.online ? "워커 준비됨" : "워커 시작 필요"}</span></div><small class="muted">격리된 테스트 세션을 사용하며 .env와 운영 세션을 수정하지 않습니다.</small></form></section><div><section class="card"><div class="card-head"><h2>프롬프트와 환경 설정</h2><span class="pill">SERVICE SOURCE</span></div><div id="service-preview">${servicePreview(judgment)}</div></section><section class="card prose"><h2>대화와 추출의 역할</h2><p>니쥬 대화는 취미나 퇴근 후 루틴에 한정하지 않고 경험·가치관·관계·선택의 이유·미래 계획을 따라갑니다. 취향이 나오지 않아도 화제를 강제로 바꾸거나 추천 항목을 채우려 묻지 않습니다.</p><p>대화 이력은 넓게 이어가고, 추출기는 현재 스키마에 맞는 근거 있는 취향·관심사만 별도로 읽습니다. 모든 가치관과 계획이 프로필 필드로 저장되는 것은 아닙니다.</p></section><section class="card prose"><h2>실행 후 확인할 결과</h2><p>각 턴의 AI 답변, 실제 모델 요청 메시지·추출 스키마와 응답, 프로필 병합 결과, 검색 쿼리 및 제외·선호 문장을 확인합니다.</p><p>검색 쿼리는 서비스의 규칙 기반 생성 로직을 그대로 사용합니다. 실제 임베딩·상품 추천을 선택하면 해당 외부 모델과 DB도 호출합니다.</p><p>대화·프로필·검색 쿼리의 정답 평가는 이 실험에 포함되지 않습니다. 모델 품질은 출력과 추출 근거를 검토하세요.</p></section>${baselineGuide()}</div></div>`
  );
}

function queryReviewHTML(review) {
  if (!review) return "";
  if (review.reason === "pending")
    return `<div class="note">추출·추천 쿼리 결과가 기록되면 기대 쿼리와 비교합니다.</div>`;
  if (review.score == null)
    return `<div class="note">이 실행에는 기대 쿼리 기준이 저장되지 않았습니다. 업데이트된 페르소나로 새 실험을 실행하면 조건 유사도를 확인할 수 있습니다.</div>`;
  const conditions = `${review.matched_conditions}/${review.total_conditions}개 조건 반영`;
  return `<details class="sample query-review" open><summary>추천 쿼리 비교 · 조건 유사도 ${pct(review.score)} · ${conditions}</summary><div class="sample-body"><p class="muted">키워드·동의어 기준으로 같은 검색 공간·추천 대상의 쿼리 중 조건이 가장 많이 반영된 문장을 비교합니다. 점수는 페르소나 전체 설정과의 조건 일치도입니다. 합성·편집 대화에서 아직 말하지 않은 설정은 누락될 수 있으므로 실제 발화와 추출 결과를 함께 검토하세요.</p><div class="table-wrap"><table><caption>기대 문장과 실제 추천 쿼리의 조건 비교</caption><thead><tr><th scope="col">기대 쿼리</th><th scope="col">가장 가까운 실제 쿼리</th><th scope="col">조건 유사도</th><th scope="col">반영·누락 조건</th></tr></thead><tbody>${review.rows.map((row) => `<tr><td><small>${e(row.space)} · ${e(row.mode)}</small>${e(row.expected)}</td><td>${e(row.actual || "해당 공간·대상의 쿼리 없음")}</td><td>${pct(row.score)}</td><td>반영: ${e(row.matched.join(" · ") || "없음")}<br>누락: ${e(row.missing.join(" · ") || "없음")}</td></tr>`).join("")}</tbody></table></div>${review.exclusions?.length ? `<div class="detail-section"><h3>제외 조건 반영 · ${pct(review.exclusion_score)}</h3><ul>${review.exclusions.map((row) => `<li>${e(row.expected)} → ${e(row.actual || "미반영")}</li>`).join("")}</ul></div>` : ""}${review.forbidden_queries?.length ? `<div class="note error"><strong>주의할 쿼리 ${review.forbidden_queries.length}개</strong><ul>${review.forbidden_queries.map((row) => `<li>${e(row.query)}<br>주의 표현: ${e(row.terms.join(" · "))}</li>`).join("")}</ul></div>` : `<p class="muted">지정한 오추천 주의 표현: 검출 없음</p>`}<small class="muted">기준 버전 ${e(review.persona_version || "미기록")} · 평가 방식 ${e(review.version)}</small></div></details>`;
}

function runQueryReview(run) {
  return (
    run.samples.findLast((sample) => sample.pipeline?.query_review)?.pipeline.query_review ||
    run.summary?.query_review ||
    (run.manifest.persona
      ? {
          score: null,
          reason: run.manifest.persona.expected_queries?.length ? "pending" : "no_reference",
        }
      : null)
  );
}

function pipelineResult(sample) {
  const p = sample.pipeline;
  if (!p) return "";
  let html = "";
  if (p.queries) {
    html += `<div class="muted">실제 서비스 추천 쿼리</div>${p.queries.length ? `<div class="table-wrap"><table><thead><tr><th>검색 공간</th><th>임베딩 검색 문장</th><th>적용</th></tr></thead><tbody>${p.queries.map((q) => `<tr><td>${e(q.space)}</td><td>${e(q.text)}</td><td>${e(q.modes.join(" / "))}</td></tr>`).join("")}</tbody></table></div>` : `<div class="note">현재 프로필에서 생성된 검색 쿼리가 없습니다.</div>`}`;
    html += `<details class="sample"><summary>누적 프로필 · 병합 결과</summary><div class="sample-body"><pre>${e(JSON.stringify({ profile: p.profile, merge_result: p.merge_result, extraction_failed: p.extraction_failed, session_status: p.session_status, next_decision: p.next_decision }, null, 2))}</pre></div></details><details class="sample"><summary>실제 임베딩 입력 · 제외·선호 문장</summary><div class="sample-body"><pre>${e(JSON.stringify({ inputs: p.embedding_inputs, exclusions: p.exclusions, preferences: p.preferences }, null, 2))}</pre></div></details>`;
    html += queryReviewHTML(p.query_review);
  }
  if (p.recommendation_stage)
    html += `<div class="note">실행 범위: ${e(p.recommendation_stage)} · 임베딩: ${e(p.embedding_status || "실패 / 중단")} · 추천: ${e(p.recommendation_status || "실패 / 중단")}${p.stop_reason ? `<br>${e(p.stop_reason)}` : ""}</div>`;
  if (p.embedding_batches)
    html += `<details class="sample"><summary>실제 임베딩 결과 · 차원·벡터 크기</summary><div class="sample-body"><pre>${e(JSON.stringify(p.embedding_batches, null, 2))}</pre></div></details>`;
  if (p.recommendations)
    html += `<div class="muted">실제 카탈로그 추천 결과 · self / gift</div><pre>${e(JSON.stringify(p.recommendations, null, 2))}</pre>`;
  if (sample.model_calls?.length)
    html += `<details class="sample"><summary>실제 모델 호출 ${sample.model_calls.length}회 · 프롬프트·추출 스키마·응답</summary><div class="sample-body">${sample.model_calls.map((c) => `<h3>${e(c.role)} · ${e(c.model)} · ${ms(c.latency_ms)}</h3><pre>${e(JSON.stringify(c, null, 2))}</pre>`).join("")}</div></details>`;
  return html;
}

function serviceSummary(run) {
  if (run.spec.mode !== "needu" || !run.summary) return "";
  const s = run.summary;
  return `<div class="result-strip"><span>대화 <strong>${num(s.executed_turns)} / ${num(s.planned_turns)}턴</strong></span><span>대화 왕복 p95 <strong>${ms(s.turn_latency_p95_ms)}</strong></span><span>니쥬 응답 p95 <strong>${ms(s.role_latency_p95_ms?.response)}</strong></span><span>실제 모델 호출 <strong>${num(s.model_calls)}회</strong></span><span>추출 실패 <strong>${num(s.extraction_failed_turns)}턴</strong></span></div>`;
}

function dialogueView(run) {
  if (run.spec.mode !== "needu" || !run.samples.length) return "";
  const automatic = run.spec.dialogue_mode === "simulated";
  const messages = run.samples.flatMap((s) => {
    if (s.stage === "opening" && s.output) return [{ role: "assistant", content: s.output }];
    if (s.stage !== "turn") return [];
    return [
      ...(s.input ? [{ role: "user", content: s.input }] : []),
      ...(s.output ? [{ role: "assistant", content: s.output }] : []),
    ];
  });
  return `<details class="sample" open><summary>전체 대화 · ${automatic ? `Gemini 합성 사용자 (${e(run.manifest.pipeline.persona.model)})` : "고정 사용자 발화"}</summary><div class="sample-body chat-feed">${messages.map((m) => `<div class="chat-message ${m.role}"><span class="avatar">${m.role === "assistant" ? "N" : automatic ? "P" : "나"}</span><div class="chat-bubble">${e(m.content)}</div></div>`).join("")}</div></details>`;
}

function historyPage() {
  return (
    heading(
      "EXPERIMENT HISTORY",
      "실험 이력",
      "출력과 실패 사례, 실행 당시의 모델과 조건을 함께 확인하세요.",
      `<a class="btn" href="#compare">선택한 결과 비교 (${state.compare.size}/2)</a>`,
    ) +
    `<section class="card">${runTable(state.overview.runs, true)}</section><div id="run-detail">${state.detail ? detailHTML(state.detail) : ""}</div>`
  );
}

function detailHTML(run) {
  const s = run.summary || {};
  const count = run.samples.length;
  return `<section class="card detail-title"><div class="card-head"><div><div class="eyebrow">${e(run.id.slice(0, 8))} / ${e(run.connection.environment)}</div><h2>${e(modeLabel(run.spec.mode))} · ${e(run.connection.model)}</h2></div>${pill(run.status)}</div><div class="actions">${modes[run.spec.mode] ? `<button class="btn small" data-action="copy" data-id="${run.id}">조건 복사</button>` : ""}<a class="btn small" href="/api/runs/${run.id}/export?format=json" download>JSON</a><a class="btn small" href="/api/runs/${run.id}/export?format=csv" download>CSV</a>${active(run) ? `<button class="btn small danger" data-action="cancel" data-id="${run.id}">실행 취소</button>` : `<button class="btn small danger" data-action="delete" data-id="${run.id}">이력 삭제</button>`}</div>${serviceSummary(run)}${dialogueView(run)}${queryReviewHTML(runQueryReview(run))}${run.manifest.persona ? `<details class="sample persona-detail"><summary>페르소나 · ${e(run.manifest.persona.name)} · 기대 결과 검토</summary><div class="sample-body">${personaGuide(run.manifest.persona)}</div></details>` : ""}<progress value="${count}" max="${run.spec.requests}" aria-label="${run.spec.mode === "needu" ? "완료 단계 수" : "완료 요청 수"}"></progress><div class="result-strip"><span><strong>${count} / ${run.spec.requests}</strong> ${run.spec.mode === "needu" ? "단계 기록" : "요청 기록"}</span><span>완료 p95 <strong>${ms(s.latency_p95_ms)}</strong></span><span>첫 청크 p95 <strong>${ms(s.first_chunk_p95_ms)}</strong></span><span>성공률 <strong>${pct(s.success_rate)}</strong></span><span>스키마 <strong>${pct(s.schema_rate)}</strong></span><span>정답 일치 <strong>${pct(s.exact_match_rate)}</strong></span></div>${run.error ? `<div class="note error">${e(run.error)}</div>` : ""}${s.complete === false ? `<div class="note">${s.stop_reason ? e(s.stop_reason) + ". 예정된 나머지 턴은 실행하지 않았습니다." : "전체 계획을 완료하지 못했습니다. 일부 결과를 전체 평가 성능으로 해석하지 마세요."}</div>` : ""}<details class="sample"><summary>실행 조건 · 재현 정보</summary><div class="sample-body"><pre>${e(JSON.stringify({ connection: run.connection, spec: run.spec, manifest: run.manifest, summary: run.summary }, null, 2))}</pre></div></details>${run.samples.map((sample) => `<details class="sample" data-sample="${sample.ordinal}" ${count === 1 ? "open" : ""}><summary><span class="pill ${sample.ok ? "ok" : "bad"}">${sample.ok ? "성공" : "실패"}</span> ${e(sample.label)} · ${ms(sample.latency_ms)} ${sample.evaluation ? `· ${e(sample.evaluation.reason || "평가 실패")}` : ""}${sample.truncated ? " · 출력 상한 도달" : ""}</summary><div class="sample-body"><div class="muted">입력</div><pre>${e(sample.input)}</pre><div class="muted">모델 출력</div><pre>${e(sample.output || "(출력 없음)")}</pre>${pipelineResult(sample)}${sample.expected ? `<div class="muted">기대 정답</div><pre>${e(JSON.stringify(sample.expected, null, 2))}</pre>` : ""}${sample.error ? `<div class="note error">${e(sample.error)}</div>` : ""}<div class="result-strip"><span>첫 청크 ${ms(sample.first_chunk_ms)}</span><span>출력 토큰 ${num(sample.usage?.completion_tokens)}</span><span>종료 사유 ${e(sample.finish_reason || "미제공")}</span></div></div></details>`).join("")}${!count ? `<div class="empty">${active(run) ? "워커가 모델 응답을 기다리고 있습니다." : "기록된 요청이 없습니다."}</div>` : ""}</section>`;
}

function tokenUsage(run) {
  const calls =
    run.spec.mode === "needu"
      ? run.samples.flatMap((sample) => sample.model_calls || [])
      : run.samples;
  const valid = (value) => Number.isSafeInteger(value) && value >= 0;
  return Object.fromEntries(
    ["prompt_tokens", "completion_tokens", "total_tokens"].map((key) => {
      const values = calls
        .map((call) => {
          const usage = call.usage || {};
          if (key === "total_tokens" && !valid(usage.total_tokens)) {
            return valid(usage.prompt_tokens) && valid(usage.completion_tokens)
              ? usage.prompt_tokens + usage.completion_tokens
              : null;
          }
          return usage[key];
        })
        .filter(valid);
      return [
        key,
        {
          tokens: values.length ? values.reduce((sum, value) => sum + value, 0) : null,
          reported: values.length,
          calls: calls.length,
        },
      ];
    }),
  );
}

function comparisonTokens(run, key) {
  const usage = state.compareUsage.get(run.id)?.[key];
  if (!usage) return "조회 중";
  if (usage.tokens == null) return "미제공";
  return `${num(usage.tokens)} 토큰${usage.reported < usage.calls ? ` · 일부 제공 (${usage.reported}/${usage.calls}회)` : ""}`;
}

async function refreshCompareUsage() {
  const ids = [...state.compare].filter((id) => state.overview.runs.some((run) => run.id === id));
  const usages = await Promise.all(
    ids.map(async (id) => {
      const run = await api(`/runs/${id}`);
      return [id, { ...tokenUsage(run), query_review: runQueryReview(run) }];
    }),
  );
  usages.forEach(([id, usage]) => state.compareUsage.set(id, usage));
}

function comparePage() {
  const runs = [...state.compare]
    .map((id) => state.overview.runs.find((r) => r.id === id))
    .filter(Boolean);
  let html = heading(
    "SIDE BY SIDE",
    "결과 비교",
    "실험 이력에서 최대 두 개를 선택하세요. 품질과 속도를 함께 판단합니다.",
    `<a class="btn" href="#history">실험 선택</a>`,
  );
  if (runs.length !== 2)
    return (
      html +
      `<section class="card"><div class="empty"><span class="empty-icon">⇄</span><strong>${runs.length} / 2개 선택됨</strong>이력의 비교 체크박스에서 두 실험을 선택하세요.</div></section>`
    );
  const [a, b] = runs;
  const queryReview = (run) =>
    state.compareUsage.get(run.id)?.query_review || run.summary?.query_review;
  const keys = [
    "mode",
    "prompt",
    "system",
    "temperature",
    "max_tokens",
    "requests",
    "concurrency",
    "timeout_seconds",
    "duration_seconds",
    "response_connection_id",
    "extraction_connection_id",
    "judgment_mode",
    "recommendation_stage",
  ];
  const mismatches = keys.filter((key) => a.spec[key] !== b.spec[key]);
  if (a.manifest.pipeline?.environment_hash !== b.manifest.pipeline?.environment_hash)
    mismatches.push("서비스 환경변수");
  const rows = [
    ["실행 환경", (r) => r.connection.environment],
    ["모델 ID", (r) => r.connection.model],
    [
      "revision / 정밀도",
      (r) => `${r.connection.revision || "미기록"} / ${r.connection.precision || "미기록"}`,
    ],
    ["하드웨어", (r) => r.connection.hardware || "미기록"],
    ["종류 / 동시성", (r) => `${modeLabel(r.spec.mode)} / ${r.spec.concurrency}`],
    [
      "입력·출력 조건",
      (r) =>
        r.spec.mode === "needu"
          ? `서비스 토큰 상한: 응답 ${r.manifest.pipeline.response.max_tokens} / 추출 ${r.manifest.pipeline.extraction.max_tokens} · ${r.spec.requests}단계`
          : `온도 ${r.spec.temperature} · 최대 ${r.spec.max_tokens}토큰 · ${r.spec.requests}요청`,
    ],
    [
      "사용자 대화 방식",
      (r) =>
        r.spec.dialogue_mode === "simulated"
          ? `합성 대화 · ${r.manifest.pipeline?.persona?.model || "미기록"}`
          : "고정 발화",
    ],
    [
      "판단 모델",
      (r) =>
        r.manifest.pipeline
          ? r.manifest.pipeline.judgment_enabled
            ? r.manifest.pipeline.judgment.model
            : "직접 추출"
          : "해당 없음",
    ],
    ["추천 실행 범위", (r) => r.manifest.pipeline?.recommendation_stage || "해당 없음"],
    ["대화 왕복 p95 · 사용자 생성 포함", (r) => ms(r.summary?.turn_latency_p95_ms)],
    ["니쥬 응답 p95", (r) => ms(r.summary?.role_latency_p95_ms?.response)],
    ["사용자 생성 p95", (r) => ms(r.summary?.role_latency_p95_ms?.persona)],
    ["추출 실패 턴", (r) => num(r.summary?.extraction_failed_turns)],
    ["추천 쿼리 조건 유사도", (r) => pct(queryReview(r)?.score)],
    ["제외 조건 반영률", (r) => pct(queryReview(r)?.exclusion_score)],
    ["주의할 추천 쿼리", (r) => num(queryReview(r)?.forbidden_queries?.length)],
    ["상태", (r) => statuses[r.status]],
    ["성공 표본", (r) => `${r.summary?.succeeded ?? "—"} / ${r.summary?.attempted ?? "—"}`],
    ["완료시간 p50", (r) => ms(r.summary?.latency_p50_ms)],
    ["완료시간 p95", (r) => ms(r.summary?.latency_p95_ms)],
    ["첫 청크 p95", (r) => ms(r.summary?.first_chunk_p95_ms)],
    ["성공률", (r) => pct(r.summary?.success_rate)],
    ["스키마 준수", (r) => pct(r.summary?.schema_rate)],
    ["내용 정답 일치", (r) => pct(r.summary?.exact_match_rate)],
    ["입력 토큰 수", (r) => comparisonTokens(r, "prompt_tokens")],
    ["출력 토큰 수", (r) => comparisonTokens(r, "completion_tokens")],
    ["전체 토큰 수", (r) => comparisonTokens(r, "total_tokens")],
    [
      "성공 단계·요청 처리량",
      (r) =>
        r.summary?.requests_per_second == null
          ? "미측정"
          : `${num(r.summary.requests_per_second, 2)} ${r.spec.mode === "needu" ? "stage/s" : "req/s"}`,
    ],
    [
      "전체 출력 처리량",
      (r) =>
        r.summary?.output_tokens_per_second == null
          ? "미측정"
          : `${num(r.summary.output_tokens_per_second, 2)} tok/s`,
    ],
    ["실험 메모", (r) => r.spec.note || "미기록"],
  ];
  html += `<div class="note ${mismatches.length ? "error" : "success"}">${mismatches.length ? `요청 조건 차이: ${e(mismatches.join(", "))}. 속도 우열을 직접 판정하지 마세요.` : "요청 설정이 같습니다. 런타임·하드웨어·캐시 상태·모델 정밀도도 함께 확인하세요."}<br>소규모 테스트 p95는 운영 SLO 근거로 충분하지 않습니다.</div><div class="note">토큰 수는 모델이 반환한 사용량을 합산합니다. 니쥬 서비스는 응답·추출·요약·판단·합성 사용자 호출을 포함합니다. 전체 토큰 수가 없으면 제공된 입력·출력 토큰을 더합니다. 일부 제공은 확인된 호출만 집계한 값이며, 미제공 사용량은 추정하지 않습니다. 실행 중에는 현재까지 기록된 사용량입니다.</div>`;
  return (
    html +
    `<section class="card table-wrap"><table class="compare-metrics"><thead><tr><th>항목</th><th>${e(a.connection.name)} · ${a.id.slice(0, 8)}</th><th>${e(b.connection.name)} · ${b.id.slice(0, 8)}</th></tr></thead><tbody>${rows.map(([label, get]) => `<tr><td>${label}</td><td>${e(get(a))}</td><td>${e(get(b))}</td></tr>`).join("")}</tbody></table></section><div class="note">코드 revision: ${e(a.manifest.code_revision.slice(0, 8))} / ${e(b.manifest.code_revision.slice(0, 8))} · 평가셋 해시 ${a.manifest.fixture_hash && b.manifest.fixture_hash ? (a.manifest.fixture_hash === b.manifest.fixture_hash ? "같음" : "다름") : "미사용 · 서비스 경로 실행"} · 현재 비용·GPU는 미수집</div>`
  );
}

function baselineGuide() {
  const checks = [
    "취미·루틴 밖의 가치관·관계·변화·미래 계획도 자연스럽게 이어가는가",
    "질문을 반복하지 않고, 사용자가 되물으면 먼저 답하는가",
    "짧은 답이나 주제 거절에 질문 공세를 줄이는가",
    "앞에서 말한 취향과 대화 맥락을 이어가는가",
    "단순히 언급한 물건을 좋아한다고 단정하지 않는가",
    "좋아함과 싫어함·부정 표현을 구분하는가",
    "사용자가 정정한 내용을 프로필에 반영하는가",
    "다른 사람이나 선물 대상의 취향을 사용자 취향에 섞지 않는가",
    "근거가 없을 때 정보를 지어내지 않고 추출을 비워 두는가",
    "병합된 프로필·제외 조건이 추천 쿼리에 반영되는가",
  ];
  return `<section class="card prose"><h2>학습 없이 먼저 검증하세요</h2><p>8B·1B·30B는 확정 모델이 아닙니다. 후보 2~3개를 같은 소규모 사례로 비교하고, 니쥬에 필요한 품질·지연으로 선택하세요.</p><ol><li>중요한 상황 10~15개를 고르고, 상황별 발화와 기대 결과를 실험 메모에 적습니다.</li><li>같은 발화·스타일·추천 범위를 유지하고 모델 연결만 바꿔 비교합니다.</li><li>실패하면 프롬프트·추출·프로필 병합·쿼리 중 어느 단계가 원인인지 확인합니다.</li><li>수정 후에도 모델의 실패가 반복되고 더 큰 모델의 비용·지연이 부담될 때 파인튜닝을 검토합니다.</li></ol><details class="sample"><summary>우선 확인할 상황 10개</summary><div class="sample-body"><ol>${checks.map((check) => `<li>${check}</li>`).join("")}</ol></div></details><p>이 목록은 사람이 검토할 기준입니다. 10개의 자동 정답 평가셋이 아닙니다. 니쥬 서비스 테스트에서 출력과 근거를 확인하고, 기존 초기 추출 평가 8개의 점수와 구분하세요.</p><p>상세 페르소나 ${state.personas.length}개를 시나리오와 직접 대화에서 사용하세요. 추가 확장·대규모 학습·GPU 관측은 필요가 확인되면 추가합니다. AI Hub 데이터는 후보로 보관하고, 파인튜닝이 필요해질 때 학습용으로 검토합니다.</p></section>`;
}

function purposePage() {
  return (
    heading(
      "WHY THIS BOARD EXISTS",
      "필요한 검증부터, 모델 선택은 결과로",
      "LM Studio와 Ollama는 모델을 실행하고, AI Lab은 같은 조건으로 테스트하고 비교합니다.",
    ) +
    `<section class="card hero"><div><div class="eyebrow">OUR PURPOSE</div><h2>“기존 모델로 니쥬 요구사항을 만족할까?”부터 확인하기.</h2><p>실험 실행·출력 확인·조건 비교를 중심으로 사용합니다. 실제 프롬프트와 서비스 경로로 실패 원인을 확인하고, 필요한 변경만 적용하세요.</p></div><div class="hero-symbol">◎</div></section>` +
    `<div class="grid-even"><div>${baselineGuide()}</div><section class="card prose"><h2>로컬 품질 → 실제 GPU 검증</h2><p><strong>로컬 또는 원격 모델 비교</strong><br>한국어 답변이 자연스러운가? 취향을 정확히 읽는가? 프롬프트와 구조화 출력이 작동하는가? 큰 모델은 Runpod에서 바로 실행하고 로컬 보드로 호출해도 됩니다.</p><p><strong>운영 환경 검증</strong><br>실제 GPU에서 얼마나 처리하는가? 응답·추출 모델을 함께 실행해도 빠른가? 지연과 비용을 감당하는가?</p><p>모델 가중치·양자화·런타임이 바뀌면 품질도 다시 검증합니다. 현재 보드는 GPU·콜드스타트·비용을 자동 수집하지 않습니다.</p></section></div><section class="card prose"><h2>현재 실행 범위</h2><p>니쥬 서비스의 실제 멀티턴 대화·추출·프로필 병합·추천 쿼리 생성 경로를 실행합니다. 실제 임베딩과 카탈로그 추천은 선택 사항입니다. 별도 초기 추출 평가 8개와 제한된 단일 모델 동시성 테스트도 제공합니다.</p><p>서비스 실행 성공률은 정답 정확도가 아닙니다. 추출 실패는 대화가 계속돼도 실패 단계로 표시하며, 자동 골드셋 채점은 아직 없습니다. 첫 청크 시간은 클라이언트 측정이고 usage가 없는 토큰 수는 추정하지 않습니다.</p></section>`
  );
}

function render() {
  document.querySelectorAll("nav [data-page]").forEach((node) => {
    node.classList.toggle("active", node.dataset.page === state.page);
    if (node.dataset.page === state.page) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  const pages = {
    overview: overviewPage,
    connections: connectionsPage,
    experiments: experimentsPage,
    history: historyPage,
    compare: comparePage,
    purpose: purposePage,
    personas: personasPage,
    chat: chatPage,
  };
  main.innerHTML = pages[state.page]();
  if (state.page === "experiments") updateBudget();
  if (state.page === "chat" || state.page === "history") {
    const feed = $(".chat-feed");
    if (feed) feed.scrollTop = feed.scrollHeight;
  }
}

function updateRunDetail(resetScroll = false) {
  const previousFeed = $("#run-detail .chat-feed");
  const previousScrollTop = previousFeed?.scrollTop ?? 0;
  const previousMessageCount = previousFeed?.childElementCount;
  $("#run-detail").innerHTML = detailHTML(state.detail);
  const feed = $("#run-detail .chat-feed");
  if (feed) {
    feed.scrollTop =
      resetScroll || feed.childElementCount !== previousMessageCount
        ? feed.scrollHeight
        : previousScrollTop;
  }
}

function updateBudget() {
  const form = $("#run-form");
  if (!form) return;
  const data = new FormData(form);
  if (state.mode === "needu") {
    const turns = String(data.get("prompt") || "")
      .split("\n")
      .filter((line) => line.trim()).length;
    const roleModel = (role) => {
      const selected = data.get(`${role}_connection_id`);
      return selected
        ? state.overview.connections.find((c) => c.id === selected)?.model || "모델 미선택"
        : state.service?.[role]?.model || "환경 미설정";
    };
    $("#run-budget").textContent =
      `${data.get("dialogue_mode") === "simulated" ? `최대 ${data.get("max_turns")}턴 Gemini 자동 대화 (${state.service?.persona?.model || "미설정"})` : `${turns}개 고정 사용자 턴`} + 시작 인사·최종 분석. 응답: ${roleModel("response")} / 추출: ${roleModel("extraction")}. 외부 모델 호출은 사용량에 따라 과금될 수 있습니다.`;
    const preview = $("#service-preview");
    const key = data.get("judgment_mode");
    if (preview.dataset.selection !== key) {
      preview.innerHTML = servicePreview(data.get("judgment_mode"));
      preview.dataset.selection = key;
    }
    return;
  }
  const count =
    state.mode === "quality" ? state.fixtures.cases.length : Number(data.get("requests"));
  const tokens = Number(data.get("max_tokens"));
  const connection = state.overview.connections.find((c) => c.id === data.get("connection_id"));
  $("#run-budget").textContent =
    `${connection?.name || "모델"} · ${connection?.environment || "환경 미선택"} · ${count}요청 · 최대 출력 ${num(count * tokens)}토큰. ${connection?.environment === "local" ? "로컬 서버 호출 기준입니다. 클라우드 모델 선택 여부는 런타임에서 확인하세요." : "원격 추론은 사용량에 따라 과금될 수 있습니다. 청구 비용은 아직 수집하지 않습니다."}`;
}

async function refresh(renderPage = false) {
  state.overview = await api("/overview");
  $("#worker-dot").classList.toggle("online", state.overview.worker.online);
  $("#worker-label").textContent = state.overview.worker.online
    ? "테스트 워커 준비됨"
    : "테스트 워커 꺼짐";
  if (state.activeRun) state.detail = await api(`/runs/${state.activeRun}`);
  if (state.page === "compare") await refreshCompareUsage();
  if (renderPage) render();
}

async function showDetail(id, scroll = true) {
  state.activeRun = id;
  state.detail = await api(`/runs/${id}`);
  if (state.page !== "history") {
    location.hash = "history";
  } else {
    updateRunDetail(true);
  }
  if (scroll)
    setTimeout(
      () => $("#run-detail")?.scrollIntoView({ behavior: "instant", block: "start" }),
      100,
    );
}

main.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const button = form.querySelector('[type="submit"]');
  button.disabled = true;
  try {
    const data = Object.fromEntries(new FormData(form));
    if (form.id === "connection-form") {
      const c = await api(
        state.connectionId ? `/connections/${state.connectionId}` : "/connections",
        { method: state.connectionId ? "PUT" : "POST", body: JSON.stringify(data) },
      );
      state.connectionId = c.id;
      await refresh(true);
      notice(
        c.provider === "huggingface"
          ? `API 제공 여부 확인 · ${c.model || "모델 미선택"} 저장. 토큰 유효성은 실제 추론으로 확인하세요.`
          : "연결 설정을 저장했습니다.",
      );
    } else if (form.id === "chat-create-form") {
      state.chatDraft = data;
      data.response_connection_id ||= null;
      data.extraction_connection_id ||= null;
      data.persona_id ||= null;
      state.chatBusy = true;
      render();
      try {
        state.chatDetail = await api("/chats", { method: "POST", body: JSON.stringify(data) });
        state.chatSessions = await api("/chats");
        sessionStorage.setItem("needu-active-chat", state.chatDetail.id);
        state.chatMessage = "";
      } finally {
        state.chatBusy = false;
        render();
      }
    } else if (form.id === "chat-message-form") {
      const id = state.chatDetail.id;
      state.chatMessage = String(data.message || "");
      state.chatPending = state.chatMessage;
      state.chatBusy = true;
      render();
      try {
        state.chatDetail = await api(`/chats/${id}/messages`, {
          method: "POST",
          body: JSON.stringify({ userId: 1, message: state.chatMessage }),
        });
        state.chatSessions = await api("/chats");
        sessionStorage.setItem("needu-active-chat", state.chatDetail.id);
        state.chatMessage = "";
      } catch (error) {
        state.chatDetail = await api(`/chats/${id}`).catch(() => state.chatDetail);
        throw error;
      } finally {
        state.chatPending = "";
        state.chatBusy = false;
        render();
      }
    } else if (form.id === "run-form") {
      for (const key of [
        "temperature",
        "max_tokens",
        "requests",
        "concurrency",
        "duration_seconds",
        "timeout_seconds",
        "max_turns",
      ]) {
        if (data[key] != null) data[key] = Number(data[key]);
      }
      data.mode = state.mode;
      if (data.mode === "needu") {
        data.response_connection_id ||= null;
        data.extraction_connection_id ||= null;
        data.persona_id ||= null;
        data.prompt ||= "";
      }
      const run = await api("/runs", { method: "POST", body: JSON.stringify(data) });
      state.draft = run.spec;
      await refresh();
      await showDetail(run.id);
      notice("테스트를 시작했습니다. 이력에서 진행 상황을 확인하세요.");
    }
  } catch (error) {
    notice(error.message, true);
  } finally {
    button.disabled = false;
  }
});

main.addEventListener("input", (event) => {
  if (event.target.id === "persona-search") {
    state.personaSearch = event.target.value;
    $("#persona-cards").innerHTML = personaCards();
  }
  if (event.target.id === "chat-message") state.chatMessage = event.target.value;
  updateBudget();
});
main.addEventListener("change", (event) => {
  if (
    ["dialogue_mode", "persona_id"].includes(event.target.id) &&
    event.target.closest("#run-form")
  ) {
    state.draft = { ...state.draft, ...Object.fromEntries(new FormData($("#run-form"))) };
    if (event.target.id === "persona_id" && state.draft.dialogue_mode !== "simulated") {
      const p = state.personas.find((p) => p.id === state.draft.persona_id);
      if (p) state.draft.prompt = p.turns.join("\n");
    }
    render();
    return;
  }
  if (event.target.id === "persona-category") {
    state.personaCategory = event.target.value;
    $("#persona-cards").innerHTML = personaCards();
    return;
  }
  if (event.target.closest("#chat-create-form")) {
    state.chatDraft = Object.fromEntries(new FormData($("#chat-create-form")));
    render();
    return;
  }
  const id = event.target.dataset.compare;
  if (!id) return;
  if (event.target.checked && state.compare.size >= 2) {
    event.target.checked = false;
    notice("두 개까지 비교할 수 있어요. 기존 선택을 해제하세요.", true);
    return;
  }
  if (event.target.checked) state.compare.add(id);
  else state.compare.delete(id);
  const link = main.querySelector('a[href="#compare"]');
  if (link) link.textContent = `선택한 결과 비교 (${state.compare.size}/2)`;
});

main.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const { action, id, mode } = button.dataset;
  button.disabled = true;
  try {
    if (action === "persona") {
      state.selectedPersona = id;
      render();
    } else if (action === "persona-run") {
      const p = state.personas.find((p) => p.id === id);
      state.draft = {
        mode: "needu",
        persona_id: p.id,
        dialogue_mode: "simulated",
        max_turns: 10,
        prompt: p.turns.join("\n"),
        note: `${p.name} · ${p.tag} · 기대 결과는 수동 검토`,
      };
      state.mode = "needu";
      location.hash = "experiments";
      if (state.page === "experiments") render();
    } else if (action === "persona-chat" || action === "new-chat") {
      if (state.chatBusy) return;
      const p = action === "persona-chat" ? state.personas.find((p) => p.id === id) : null;
      state.chatDetail = null;
      sessionStorage.removeItem("needu-active-chat");
      state.chatMessage = "";
      state.chatDraft = {
        ...state.chatDraft,
        persona_id: p?.id || "",
        title: p ? `${p.name} · 직접 대화` : "로컬 모델 대화 테스트",
      };
      location.hash = "chat";
      if (state.page === "chat") render();
    } else if (action === "open-chat") {
      if (state.chatBusy) return;
      state.chatDetail = await api(`/chats/${id}`);
      sessionStorage.setItem("needu-active-chat", id);
      state.chatMessage = "";
      render();
    } else if (action === "analyze-chat") {
      state.chatBusy = true;
      render();
      try {
        state.chatDetail = await api(`/chats/${id}/analysis`, { method: "POST" });
        state.chatSessions = await api("/chats");
      } catch (error) {
        state.chatDetail = await api(`/chats/${id}`).catch(() => state.chatDetail);
        throw error;
      } finally {
        state.chatBusy = false;
        render();
      }
    } else if (action === "delete-chat") {
      if (!window.confirm("이 대화와 로컬 추출·호출 기록을 삭제할까요?")) return;
      await api(`/chats/${id}`, { method: "DELETE" });
      state.chatDetail = null;
      sessionStorage.removeItem("needu-active-chat");
      state.chatSessions = await api("/chats");
      render();
    } else if (action === "service-refresh") {
      state.draft = { ...state.draft, ...Object.fromEntries(new FormData($("#run-form"))) };
      state.service = await api("/service-config");
      state.serviceError = null;
      render();
      notice("서비스 환경 설정을 다시 읽었습니다. .env는 수정하지 않았습니다.");
    } else if (action === "hf-connection") {
      const connection = state.overview.connections.find((c) => c.provider === "huggingface");
      if (!connection) throw new Error("Hugging Face 연결을 불러오려면 관리 API를 재시작하세요.");
      state.connectionId = connection.id;
      render();
    } else if (action === "openrouter-connection") {
      const connection = state.overview.connections.find((c) => c.id === "openrouter");
      if (!connection) throw new Error("OpenRouter 연결을 불러오려면 관리 API를 재시작하세요.");
      state.connectionId = connection.id;
      render();
    } else if (action === "new-connection" || action === "edit-connection") {
      state.connectionId = action === "new-connection" ? null : id;
      render();
    } else if (action === "probe") {
      const result = await api(`/connections/${id}/probe`, { method: "POST" });
      await refresh();
      // Preserve unsaved fields while loading provider model choices.
      const c = state.overview.connections.find((entry) => entry.id === id);
      $("#probe-result").innerHTML = probeResult(c);
      $("#model-options").innerHTML = result.models
        .map((m) => `<option value="${e(m)}"></option>`)
        .join("");
      notice(
        result.ok
          ? `${result.models.length}개 모델을 확인했습니다. 모델 ID를 선택하고 저장하세요.`
          : result.error,
        !result.ok,
      );
    } else if (action === "mode") {
      const form = $("#run-form");
      state.draft = { ...state.draft, ...Object.fromEntries(new FormData(form)) };
      state.mode = mode;
      render();
    } else if (action === "detail") {
      await showDetail(id);
    } else if (action === "cancel") {
      await api(`/runs/${id}/cancel`, { method: "POST" });
      notice(
        "취소를 요청했습니다. 클라이언트 요청은 중단되며 원격 연산 종료 시점은 서버에 따라 달라요.",
      );
      await refresh(true);
    } else if (action === "copy") {
      const run = await api(`/runs/${id}`);
      if (!modes[run.spec.mode]) {
        notice("이전 단일 모델 모드는 종료됐습니다. 니쥬 서비스 테스트에서 새 실험을 만드세요.");
        return;
      }
      state.draft = run.spec;
      state.mode = run.spec.mode;
      state.connectionId = run.spec.connection_id;
      location.hash = "experiments";
      notice("실행 조건을 복사했습니다. 비교할 모델 연결을 바꾸고 실행하세요.");
    } else if (action === "delete") {
      if (!window.confirm("이 실험의 입력·출력·측정 기록을 로컬 DB에서 삭제할까요?")) return;
      await api(`/runs/${id}`, { method: "DELETE" });
      state.compare.delete(id);
      if (state.activeRun === id) {
        state.activeRun = null;
        state.detail = null;
      }
      await refresh(true);
      notice("실험 이력을 삭제했습니다.");
    }
  } catch (error) {
    notice(error.message, true);
  } finally {
    button.disabled = false;
  }
});

async function navigate() {
  const page = location.hash.slice(1) || "overview";
  state.page = [
    "overview",
    "connections",
    "experiments",
    "history",
    "compare",
    "purpose",
    "personas",
    "chat",
  ].includes(page)
    ? page
    : "overview";
  render();
  if (state.page === "compare") {
    try {
      await refreshCompareUsage();
      if (state.page === "compare") render();
    } catch (error) {
      notice(error.message, true);
    }
  }
}

window.addEventListener("hashchange", navigate);
$("#clock").textContent = new Date().toLocaleDateString("ko-KR", {
  timeZone: "Asia/Seoul",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

try {
  let personaData;
  [state.overview, state.fixtures, state.service, personaData, state.chatSessions] =
    await Promise.all([
      api("/overview"),
      api("/fixtures"),
      api("/service-config").catch((error) => {
        state.serviceError = error.message;
        return null;
      }),
      api("/personas"),
      api("/chats"),
    ]);
  state.personas = personaData.personas;
  const savedChat = sessionStorage.getItem("needu-active-chat");
  if (savedChat && state.chatSessions.some((s) => s.id === savedChat))
    state.chatDetail = await api(`/chats/${savedChat}`);
  await refresh();
  navigate();
} catch (error) {
  main.innerHTML = `<div class="card note error">관리 API 연결 실패: ${e(error.message)}. 페이지를 다시 열어주세요.</div>`;
}

let refreshing = false;
setInterval(async () => {
  if (document.hidden || refreshing) return;
  refreshing = true;
  try {
    const oldDetail = state.detail;
    await refresh(false);
    if (state.page === "overview") render();
    else if (state.page === "compare") render();
    else if (
      state.page === "history" &&
      state.detail &&
      (active(state.detail) || active(oldDetail || {}))
    ) {
      const expanded = [...document.querySelectorAll("details[data-sample][open]")].map(
        (d) => d.dataset.sample,
      );
      updateRunDetail();
      document.querySelectorAll("details[data-sample]").forEach((d) => {
        if (expanded.includes(d.dataset.sample)) d.open = true;
      });
      const table = main.querySelector("section.card");
      table.innerHTML = runTable(state.overview.runs, true);
    }
  } catch (error) {
    notice(error.message, true);
  } finally {
    refreshing = false;
  }
}, 2000);
