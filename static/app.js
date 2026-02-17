const state = {
  result: null,
};

const el = (id) => document.getElementById(id);

function fmtList(values, max = 4) {
  if (!values || values.length === 0) return "";
  if (values.length <= max) return values.join(", ");
  return `${values.slice(0, max).join(", ")} ... (+${values.length - max})`;
}

function setStatus(text, isError = false) {
  const node = el("status");
  node.textContent = text;
  node.style.color = isError ? "#ff8080" : "";
}

function readPayload() {
  return {
    keys_file: el("keys_file").value.trim(),
    deposit_file: el("deposit_file").value.trim(),
    withdraw_file: el("withdraw_file").value.trim(),
    shelley_extra_gap: Number(el("shelley_extra_gap").value),
    shelley_scan_cap: Number(el("shelley_scan_cap").value),
    byron_scan_cap: Number(el("byron_scan_cap").value),
    byron_gap_limit: Number(el("byron_gap_limit").value),
    max_expansion_rounds: Number(el("max_expansion_rounds").value),
    co_spend_input_cap: Number(el("co_spend_input_cap").value),
    co_spend_output_cap: Number(el("co_spend_output_cap").value),
  };
}

function addSummaryCard(container, key, value) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<div class="k">${key}</div><div class="v">${value}</div>`;
  container.appendChild(card);
}

function renderSummary(result) {
  const cards = el("summary_cards");
  cards.innerHTML = "";
  const s = result.summary || {};
  addSummaryCard(cards, "Transactions", s.transactions_total ?? 0);
  addSummaryCard(cards, "Owned Addresses", s.owned_addresses_total ?? 0);
  addSummaryCard(cards, "Reward Accounts", s.reward_addresses_total ?? 0);
  addSummaryCard(cards, "Shelley Accounts", s.shelley_accounts_total ?? 0);
  addSummaryCard(cards, "Byron XPUBs", s.byron_xpubs_total ?? 0);
  addSummaryCard(cards, "Earliest TX", s.earliest_tx_utc || "n/a");
  addSummaryCard(cards, "Latest TX", s.latest_tx_utc || "n/a");

  const typeCounts = s.tx_type_counts || {};
  Object.keys(typeCounts)
    .sort()
    .forEach((k) => addSummaryCard(cards, `Type: ${k}`, typeCounts[k]));
}

function populateTypeFilter(transactions) {
  const select = el("type_filter");
  const current = select.value;
  const types = [...new Set(transactions.map((t) => t.type))].sort();
  select.innerHTML = `<option value="">All</option>${types
    .map((t) => `<option value="${t}">${t}</option>`)
    .join("")}`;
  if (types.includes(current)) select.value = current;
}

function renderTransactions() {
  const result = state.result;
  if (!result) return;
  const tbody = document.querySelector("#tx_table tbody");
  tbody.innerHTML = "";
  const typeFilter = el("type_filter").value.trim();
  const textFilter = el("text_filter").value.trim().toLowerCase();
  const limit = Math.max(10, Number(el("limit_filter").value || 500));

  let rows = [...(result.transactions || [])];
  if (typeFilter) rows = rows.filter((r) => r.type === typeFilter);
  if (textFilter) {
    rows = rows.filter((r) => {
      const full = [
        r.tx_hash,
        ...(r.origin_wallets || []),
        ...(r.destination_wallets || []),
        ...(r.tags || []),
      ]
        .join(" ")
        .toLowerCase();
      return full.includes(textFilter);
    });
  }
  rows = rows.slice(0, limit);

  for (const tx of rows) {
    const tr = document.createElement("tr");
    const netClass = Number(tx.net_lovelace) < 0 ? "neg" : "pos";
    tr.innerHTML = `
      <td>${tx.timestamp_utc}</td>
      <td>${(tx.tags || []).map((t) => `<span class="tag">${t}</span>`).join("")}</td>
      <td class="${netClass}">${tx.net_ada}</td>
      <td>${tx.fee}</td>
      <td>${tx.inputs_count}</td>
      <td>${tx.outputs_count}</td>
      <td title="${(tx.origin_wallets || []).join("\n")}">${fmtList(tx.origin_wallets, 3)}</td>
      <td title="${(tx.destination_wallets || []).join("\n")}">${fmtList(tx.destination_wallets, 3)}</td>
      <td class="mono" title="${tx.tx_hash}">${tx.tx_hash.slice(0, 16)}...</td>
    `;
    tbody.appendChild(tr);
  }
}

async function loadDefaults() {
  const res = await fetch("/api/defaults");
  if (!res.ok) return;
  const data = await res.json();
  if (data.keys_file) el("keys_file").value = data.keys_file;
  if (data.deposit_file) el("deposit_file").value = data.deposit_file;
  if (data.withdraw_file) el("withdraw_file").value = data.withdraw_file;
}

async function runReconcile() {
  const btn = el("run_btn");
  btn.disabled = true;
  setStatus("Running reconciliation... this can take a while.");
  try {
    const res = await fetch("/api/reconcile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readPayload()),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Reconciliation failed");
    }
    state.result = data;
    renderSummary(data);
    populateTypeFilter(data.transactions || []);
    renderTransactions();
    setStatus(
      `Done. ${data.summary.transactions_total} transactions reconciled. Saved to output/latest-reconciliation.json`
    );
  } catch (err) {
    setStatus(err.message || String(err), true);
  } finally {
    btn.disabled = false;
  }
}

el("run_btn").addEventListener("click", runReconcile);
el("type_filter").addEventListener("change", renderTransactions);
el("text_filter").addEventListener("input", renderTransactions);
el("limit_filter").addEventListener("input", renderTransactions);

loadDefaults().catch(() => {});
