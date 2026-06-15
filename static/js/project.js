/* 统一项目文件前端模块。依赖 app.js 已定义的全局：
   escapeHtml, showToast, setStatus, segmentsState, roleProfilesState,
   voiceLibraryState, getVoiceLabel, getInferenceDevice。
   通过 setWorkspaceTab('project') → renderProjectTab() 进入。 */
(function () {
  "use strict";

  let projectCurrent = null;   // 当前加载的 Project 对象
  let projectListCache = [];
  let projectBusy = false;

  function esc(s) { return (typeof escapeHtml === "function") ? escapeHtml(String(s == null ? "" : s)) : String(s == null ? "" : s); }
  function toast(msg, type) { if (typeof showToast === "function") showToast(msg, type || "info"); else if (typeof setStatus === "function") setStatus(msg); }

  async function projectApi(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const r = await fetch(path, opts);
    let data = null;
    try { data = await r.json(); } catch (e) { /* ignore */ }
    if (!r.ok || (data && data.ok === false)) {
      const detail = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
      throw new Error(detail);
    }
    return data;
  }

  function root() { return document.getElementById("projectRoot"); }

  function setBusy(on, label) {
    projectBusy = on;
    const el = document.getElementById("projectBusy");
    if (el) { el.textContent = on ? (label || "处理中…") : ""; el.style.display = on ? "" : "none"; }
    root().querySelectorAll("button").forEach((b) => { if (!b.dataset.keepEnabled) b.disabled = on; });
  }

  // ── 顶层入口 ───────────────────────────────────────────────────────────────
  async function renderProjectTab() {
    if (!projectCurrent) { await renderProjectList(); }
    else { renderProjectDetail(); }
  }
  window.renderProjectTab = renderProjectTab;

  // ── 列表 / 新建 ────────────────────────────────────────────────────────────
  async function renderProjectList() {
    const r = root();
    r.innerHTML = `
      <div class="card">
        <div class="section-head">
          <h2 style="margin-bottom:0;">项目</h2>
          <div class="section-meta">统一记录分段、角色音色分配与生成状态；支持按条目重生、跨卷复用角色表</div>
        </div>
        <div class="actions">
          <button type="button" onclick="projectSaveCurrentAnalysis()">把当前分析存为项目</button>
          <button type="button" class="secondary" onclick="projectShowCreate()">新建空项目</button>
          <button type="button" class="secondary" onclick="renderProjectTab()">刷新列表</button>
        </div>
        <div id="projectBusy" class="mini-note" style="display:none;color:#8b5e34;"></div>
        <div id="projectCreateForm" hidden></div>
        <div id="projectListBody" class="project-list"></div>
      </div>`;
    await refreshProjectList();
  }

  async function refreshProjectList() {
    const body = document.getElementById("projectListBody");
    try {
      const data = await projectApi("GET", "/api/project/list");
      projectListCache = data.projects || [];
    } catch (e) { body.innerHTML = `<div class="import-hint">加载项目列表失败：${esc(e.message)}</div>`; return; }
    if (!projectListCache.length) { body.innerHTML = `<div class="import-hint">还没有项目。先在「文本分析」完成分段并分配角色，再点「把当前分析存为项目」。</div>`; return; }
    body.innerHTML = projectListCache.map((p) => `
      <div class="project-card" onclick="projectOpen('${esc(p.project_id)}')">
        <div class="project-card-title">${esc(p.title || p.project_id)}</div>
        <div class="project-card-meta">
          <span class="pill">${p.segments} 段</span>
          <span class="pill">已生成 ${p.done}/${p.segments}</span>
          <span class="pill">角色 ${p.cast}</span>
        </div>
        <div class="mini-note">${esc(p.project_id)}${p.updated_at ? " · " + esc(p.updated_at) : ""}</div>
      </div>`).join("");
  }

  function projectShowCreate() {
    const f = document.getElementById("projectCreateForm");
    if (!f) return;
    const opts = projectListCache.map((p) => `<option value="${esc(p.project_id)}">${esc(p.title || p.project_id)}</option>`).join("");
    f.hidden = false;
    f.innerHTML = `
      <div class="editor-shell" style="margin-top:10px;">
        <label>项目 ID（英文/数字，唯一）</label>
        <input id="npProjectId" type="text" placeholder="muli_vol4" />
        <label>标题</label>
        <input id="npTitle" type="text" placeholder="第四卷" />
        <label>从已有项目导入角色表（可选）</label>
        <select id="npFromProject"><option value="">不导入</option>${opts}</select>
        <div class="actions" style="margin-top:8px;">
          <button type="button" onclick="projectDoCreate()">创建</button>
          <button type="button" class="secondary" onclick="document.getElementById('projectCreateForm').hidden=true">取消</button>
        </div>
      </div>`;
  }
  window.projectShowCreate = projectShowCreate;

  async function projectDoCreate() {
    const pid = (document.getElementById("npProjectId").value || "").trim();
    const title = (document.getElementById("npTitle").value || "").trim();
    const fromProject = (document.getElementById("npFromProject").value || "").trim();
    if (!pid) { toast("请填写项目 ID", "error"); return; }
    setBusy(true, "创建中…");
    try {
      const data = await projectApi("POST", "/api/project/create", { project_id: pid, title, from_project: fromProject || undefined });
      toast("项目已创建", "success");
      await projectOpen(data.project.project_id);
    } catch (e) { toast("创建失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectDoCreate = projectDoCreate;

  function collectAliasMap() {
    // 前端若有别名映射可在此补充；当前以空映射交给后端（后端按 speaker 原名建 cast）。
    try { if (typeof characterAliasMap === "object" && characterAliasMap) return characterAliasMap; } catch (e) {}
    return {};
  }

  // 从 narrationProgress 找出与当前分段匹配的已生成进度（每项 null 或 {file, wavFile}）。
  function collectGeneratedClips(count) {
    try {
      const np = (typeof narrationProgress === "object" && narrationProgress) ? narrationProgress : null;
      if (!np) return [];
      const sig = (typeof getNarrationProgressSignature === "function") ? getNarrationProgressSignature() : null;
      let exact = null, anyMatch = null;
      for (const entry of Object.values(np)) {
        if (!entry || entry.total !== count || !Array.isArray(entry.segments)) continue;
        if (sig && entry.signature && entry.signature === sig) { exact = entry; break; }
        if (!anyMatch) anyMatch = entry;
      }
      const chosen = exact || anyMatch;
      return chosen ? (chosen.segments || []) : [];
    } catch (e) { return []; }
  }

  async function projectSaveCurrentAnalysis() {
    const segs = (typeof segmentsState !== "undefined" && Array.isArray(segmentsState)) ? segmentsState : [];
    if (!segs.length) { toast("当前没有分段。请先在「文本分析」完成分析。", "error"); return; }
    const pid = prompt("项目 ID（英文/数字，唯一）：", "vol_" + new Date().toISOString().slice(0, 10).replace(/-/g, ""));
    if (!pid) return;
    const title = prompt("项目标题：", pid) || pid;
    const roleProfiles = (typeof roleProfilesState === "object" && roleProfilesState) ? roleProfilesState : {};
    // 收集当前分段「已生成」的片段路径（来自 narrationProgress），随保存一并导入项目
    const progressSegs = collectGeneratedClips(segs.length);
    setBusy(true, "正在创建项目…");
    try {
      const payload = {
        project_id: pid.trim(), title: title.trim(),
        segments: segs.map((s, i) => ({ speaker: s.speaker, text: s.text, emotion: s.emotion, style: s.style,
          ref_audio: s.ref_audio, ref_text: s.ref_text, voice_engine: s.voice_engine, voice_name: s.voice_name,
          clip: (progressSegs[i] && progressSegs[i].file) || undefined })),
        role_profiles: roleProfiles, alias_map: collectAliasMap(), overwrite: false,
      };
      const data = await projectApi("POST", "/api/project/from-segments", payload);
      const done = (data.project && data.project.status_counts && data.project.status_counts.done) || 0;
      toast(done ? `已存为项目（已导入 ${done} 段已生成音频）` : "已存为项目", "success");
      await projectOpen(data.project.project_id);
    } catch (e) { toast("保存失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectSaveCurrentAnalysis = projectSaveCurrentAnalysis;

  // ── 加载 / 详情 ────────────────────────────────────────────────────────────
  async function projectOpen(pid) {
    setStatus && setStatus("加载项目…");
    try {
      const data = await projectApi("GET", "/api/project/" + encodeURIComponent(pid));
      projectCurrent = data.project;
      renderProjectDetail();
    } catch (e) { toast("加载失败：" + e.message, "error"); }
  }
  window.projectOpen = projectOpen;

  function projectBack() { projectCurrent = null; renderProjectTab(); }
  window.projectBack = projectBack;

  function statusBadge(st) {
    const map = { done: ["已生成", "#1f7a3f", "#e3f3e8"], stale: ["待更新", "#9a6a1a", "#f7eccd"],
      pending: ["未生成", "#6b5b4a", "#efe6d8"], error: ["错误", "#a33", "#f7d9d9"] };
    const [t, c, bg] = map[st] || map.pending;
    return `<span class="proj-badge" style="color:${c};background:${bg};">${t}</span>`;
  }

  function clipUrl(seg) {
    if (!projectCurrent || !seg.gen || seg.gen.status !== "done" || !seg.gen.clip) return "";
    return "/file/projects/" + encodeURIComponent(projectCurrent.project_id) + "/" + seg.gen.clip.split("/").map(encodeURIComponent).join("/");
  }

  function voiceOptionsHtml(selectedRefAudio) {
    const lib = (typeof voiceLibraryState !== "undefined" && Array.isArray(voiceLibraryState)) ? voiceLibraryState : [];
    return `<option value="">未分配</option>` + lib.map((v) => {
      const label = (typeof getVoiceLabel === "function") ? getVoiceLabel(v) : (v.voice_name || v.ref_audio || v.id);
      const sel = (v.ref_audio && v.ref_audio === selectedRefAudio) ? "selected" : "";
      return `<option value="${esc(v.id)}" ${sel}>${esc(label)}</option>`;
    }).join("");
  }

  function renderProjectDetail() {
    const p = projectCurrent;
    const counts = { done: 0, stale: 0, pending: 0, error: 0 };
    p.segments.forEach((s) => { counts[(s.gen && s.gen.status) || "pending"]++; });
    const r = root();
    r.innerHTML = `
      <div class="card">
        <div class="section-head">
          <h2 style="margin-bottom:0;">${esc(p.title || p.project_id)}</h2>
          <div class="section-meta">${esc(p.project_id)} · ${p.segments.length} 段 · 已生成 ${counts.done}/${p.segments.length}</div>
        </div>
        <div class="actions">
          <button type="button" class="secondary" onclick="projectBack()">← 返回列表</button>
          <button type="button" onclick="projectGenerate(false)">生成待处理</button>
          <button type="button" class="secondary" onclick="projectGenerate(true)">全部重生（强制）</button>
          <button type="button" class="secondary" onclick="projectMerge()">合并整章</button>
          <button type="button" class="secondary" onclick="projectImportCastPrompt()">导入角色表</button>
          <button type="button" class="secondary" onclick="projectSave()">保存项目</button>
        </div>
        <div id="projectBusy" class="mini-note" style="display:none;color:#8b5e34;"></div>
        <div id="projectMergeOut" class="mini-note"></div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">角色表（音色分配）</h3>
        <div class="mini-note">改音色后点「保存项目」生效；受影响的段会标记为待更新。</div>
        <div class="project-cast">
          ${p.cast.map((m, i) => `
            <div class="assignment-card">
              <div class="assignment-head"><strong>${esc(m.canonical)}</strong>
                ${m.aliases && m.aliases.length ? `<span class="pill">别名 ${esc(m.aliases.join("、"))}</span>` : ""}</div>
              <label>分配声音</label>
              <select onchange="projectCastVoiceChange(${i}, this.value)">${voiceOptionsHtml(m.voice && m.voice.ref_audio)}</select>
              <div class="mini-note">ref_audio：${esc((m.voice && m.voice.ref_audio) || "未设置")}</div>
            </div>`).join("")}
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">分段（${p.segments.length}）</h3>
        <table class="project-segs">
          <thead><tr><th>#</th><th>角色</th><th>文本</th><th>状态</th><th>音频</th><th>操作</th></tr></thead>
          <tbody>
            ${p.segments.slice().sort((a,b)=>a.order-b.order).map((s) => {
              const url = clipUrl(s);
              return `<tr data-seg="${esc(s.seg_id)}">
                <td>${s.order + 1}</td>
                <td>${esc(s.speaker)}</td>
                <td class="proj-text"><span class="proj-text-view" title="点击编辑" onclick="projectEditText('${esc(s.seg_id)}')">${esc(s.text)}</span></td>
                <td>${statusBadge((s.gen && s.gen.status) || "pending")}</td>
                <td>${url ? `<audio controls preload="none" src="${url}" style="width:200px;"></audio>` : "—"}</td>
                <td><button type="button" class="secondary" onclick="projectRegen('${esc(s.seg_id)}')">重生成</button></td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>`;
  }

  function projectCastVoiceChange(idx, voiceId) {
    const m = projectCurrent.cast[idx];
    if (!m) return;
    const lib = (typeof voiceLibraryState !== "undefined" && Array.isArray(voiceLibraryState)) ? voiceLibraryState : [];
    const v = lib.find((x) => x.id === voiceId);
    m.voice = v ? { voice_engine: v.voice_engine || "index-tts", ref_audio: v.ref_audio || null,
      ref_text: v.ref_text || null, style: v.style || null, voice_name: v.voice_name || null } : {};
  }
  window.projectCastVoiceChange = projectCastVoiceChange;

  async function projectSave() {
    if (!projectCurrent) return;
    setBusy(true, "保存中…");
    try {
      await projectApi("POST", "/api/project/" + encodeURIComponent(projectCurrent.project_id) + "/save", projectCurrent);
      await projectOpen(projectCurrent.project_id);
      toast("已保存", "success");
    } catch (e) { toast("保存失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectSave = projectSave;

  async function projectEditText(segId) {
    const seg = projectCurrent.segments.find((s) => s.seg_id === segId);
    if (!seg) return;
    const next = prompt("编辑文本：", seg.text);
    if (next === null || next === seg.text) return;
    setBusy(true, "保存中…");
    try {
      await projectApi("POST", "/api/project/" + encodeURIComponent(projectCurrent.project_id) + "/segment/" + encodeURIComponent(segId), { text: next });
      await projectOpen(projectCurrent.project_id);
    } catch (e) { toast("编辑失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectEditText = projectEditText;

  async function projectGenerate(force) {
    if (!projectCurrent) return;
    const n = force ? projectCurrent.segments.length : projectCurrent.segments.filter((s) => ["pending","stale","error"].includes((s.gen&&s.gen.status)||"pending")).length;
    if (!n) { toast("没有需要生成的段。", "info"); return; }
    if (!confirm(`将生成 ${n} 段。首段含模型加载（约 1 分钟），其后每段约 1-2 秒。继续？`)) return;
    setBusy(true, `生成中（${n} 段，首段较慢）…`);
    try {
      const data = await projectApi("POST", "/api/project/" + encodeURIComponent(projectCurrent.project_id) + "/generate", { force: !!force, merge: false });
      const res = data.result || {};
      toast(`完成：成功 ${res.done}/${res.requested}${res.errors && res.errors.length ? "，失败 " + res.errors.length : ""}`, res.errors && res.errors.length ? "error" : "success");
      await projectOpen(projectCurrent.project_id);
    } catch (e) { toast("生成失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectGenerate = projectGenerate;

  async function projectRegen(segId) {
    if (!projectCurrent) return;
    setBusy(true, "重生成该段…");
    try {
      const data = await projectApi("POST", "/api/project/" + encodeURIComponent(projectCurrent.project_id) + "/generate", { seg_ids: [segId], merge: false });
      const res = data.result || {};
      toast(res.done ? "已重生成" : ("失败：" + ((res.errors && res.errors[0] && res.errors[0].error) || "")), res.done ? "success" : "error");
      await projectOpen(projectCurrent.project_id);
    } catch (e) { toast("重生成失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectRegen = projectRegen;

  async function projectMerge() {
    if (!projectCurrent) return;
    setBusy(true, "合并中…");
    try {
      const data = await projectApi("POST", "/api/project/" + encodeURIComponent(projectCurrent.project_id) + "/merge", {});
      const out = document.getElementById("projectMergeOut");
      const url = "/file/projects/" + encodeURIComponent(projectCurrent.project_id) + "/" + encodeURIComponent(projectCurrent.project_id) + ".wav";
      if (out) out.innerHTML = `已合并 ${data.merge.segments} 段：<audio controls preload="none" src="${url}" style="width:280px;vertical-align:middle;"></audio>`;
      toast("合并完成", "success");
    } catch (e) { toast("合并失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectMerge = projectMerge;

  async function projectImportCastPrompt() {
    if (!projectCurrent) return;
    let data;
    try { data = await projectApi("GET", "/api/project/list"); } catch (e) { toast("读取项目列表失败", "error"); return; }
    const others = (data.projects || []).filter((p) => p.project_id !== projectCurrent.project_id);
    if (!others.length) { toast("没有其它项目可导入。", "info"); return; }
    const from = prompt("从哪个项目导入角色表？输入项目 ID：\n" + others.map((p) => `· ${p.project_id}（${p.title || ""}，角色 ${p.cast}）`).join("\n"));
    if (!from) return;
    setBusy(true, "导入角色表…");
    try {
      await projectApi("POST", "/api/project/" + encodeURIComponent(projectCurrent.project_id) + "/import-cast", { from_project: from.trim(), replace: true });
      await projectOpen(projectCurrent.project_id);
      toast("已导入角色表", "success");
    } catch (e) { toast("导入失败：" + e.message, "error"); }
    finally { setBusy(false); }
  }
  window.projectImportCastPrompt = projectImportCastPrompt;
})();
