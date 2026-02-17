const state = {
  result: null,
  graph: {
    simulation: null,
    zoom: null,
    svg: null,
    viewport: null,
  },
};

const el = (id) => document.getElementById(id);

function fmtList(values, max = 4) {
  if (!values || values.length === 0) return "";
  if (values.length <= max) return values.join(", ");
  return `${values.slice(0, max).join(", ")} ... (+${values.length - max})`;
}

function fmtAdaFromLovelace(value) {
  const num = Number(value || 0) / 1_000_000;
  return Number.isFinite(num)
    ? num.toLocaleString(undefined, { maximumFractionDigits: 6 })
    : "0";
}

function shortAddr(addr) {
  if (!addr) return "";
  if (addr.length <= 18) return addr;
  return `${addr.slice(0, 10)}...${addr.slice(-8)}`;
}

function addrPrefix(addr) {
  if (addr.startsWith("addr1")) return "addr1";
  if (addr.startsWith("stake1")) return "stake1";
  if (addr.startsWith("Ae2")) return "Ae2";
  if (addr.startsWith("Ddz")) return "Ddz";
  return "other";
}

function setStatus(text, isError = false) {
  const node = el("status");
  node.textContent = text;
  node.style.color = isError ? "#ff8080" : "";
}

function setVizStatus(text, isError = false) {
  const node = el("viz_status");
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

function populateTypeFilters(transactions) {
  const txTypes = [...new Set(transactions.map((t) => t.type))].sort();
  const txSelect = el("type_filter");
  const txCurrent = txSelect.value;
  txSelect.innerHTML = `<option value="">All</option>${txTypes
    .map((t) => `<option value="${t}">${t}</option>`)
    .join("")}`;
  if (txTypes.includes(txCurrent)) txSelect.value = txCurrent;

  const graphSelect = el("graph_type_filter");
  const graphCurrent = graphSelect.value;
  graphSelect.innerHTML = `<option value="">All</option>${txTypes
    .map((t) => `<option value="${t}">${t}</option>`)
    .join("")}`;
  if (txTypes.includes(graphCurrent)) graphSelect.value = graphCurrent;
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
      <td title="${(tx.destination_wallets || []).join("\n")}">${fmtList(
      tx.destination_wallets,
      3
    )}</td>
      <td class="mono" title="${tx.tx_hash}">${tx.tx_hash.slice(0, 16)}...</td>
    `;
    tbody.appendChild(tr);
  }
}

function setDetail(title, metaLines, rows) {
  el("viz_detail_title").textContent = title;
  el("viz_detail_meta").innerHTML = metaLines.map((x) => `<div>${x}</div>`).join("");
  const wrap = el("viz_detail_rows");
  wrap.innerHTML = "";
  rows.forEach((row) => {
    const div = document.createElement("div");
    div.className = "viz-row";
    div.innerHTML = `<div class="k">${row.k}</div><div class="v">${row.v}</div>`;
    wrap.appendChild(div);
  });
}

function aggregateByAddress(details) {
  const m = new Map();
  (details || []).forEach((d) => {
    const addr = d.address;
    if (!addr) return;
    const current = m.get(addr) || {
      address: addr,
      value_lovelace: 0,
      owned: Boolean(d.owned),
    };
    current.value_lovelace += Number(d.value_lovelace || 0);
    current.owned = current.owned || Boolean(d.owned);
    m.set(addr, current);
  });
  return [...m.values()];
}

function topAddressesForPairing(list, take, ownedSet) {
  return [...list]
    .sort((a, b) => {
      const ownDiff = Number(ownedSet.has(b.address)) - Number(ownedSet.has(a.address));
      if (ownDiff !== 0) return ownDiff;
      return Number(b.value_lovelace || 0) - Number(a.value_lovelace || 0);
    })
    .slice(0, take);
}

function parseGraphDateInput(value, toEndOfDay) {
  if (!value) return null;
  const suffix = toEndOfDay ? "T23:59:59Z" : "T00:00:00Z";
  const date = new Date(`${value}${suffix}`);
  return Number.isFinite(date.getTime()) ? Math.floor(date.getTime() / 1000) : null;
}

function initializeGraphRange(transactions) {
  if (!transactions.length) return;
  const first = transactions[0];
  const last = transactions[transactions.length - 1];
  const dateFrom = el("graph_date_from");
  const dateTo = el("graph_date_to");
  if (!dateFrom.value && first?.timestamp_utc) {
    dateFrom.value = first.timestamp_utc.slice(0, 10);
  }
  if (!dateTo.value && last?.timestamp_utc) {
    dateTo.value = last.timestamp_utc.slice(0, 10);
  }
}

function buildGraphData(result) {
  const txs = result.transactions || [];
  const ownedSet = new Set(result.ownership?.owned_addresses || []);
  const typeFilter = el("graph_type_filter").value.trim();
  const minAda = Math.max(0, Number(el("graph_min_ada").value || 0));
  const minLovelace = Math.floor(minAda * 1_000_000);
  const includeExternal = el("graph_include_external").checked;
  const maxPairings = Math.max(10, Number(el("graph_max_pairings").value || 250));
  const dateFromUnix = parseGraphDateInput(el("graph_date_from").value, false);
  const dateToUnix = parseGraphDateInput(el("graph_date_to").value, true);
  const search = el("graph_search").value.trim().toLowerCase();

  const nodeMap = new Map();
  const linkMap = new Map();
  let filteredTxCount = 0;

  function getNode(addr) {
    if (!nodeMap.has(addr)) {
      nodeMap.set(addr, {
        id: addr,
        address: addr,
        owned: ownedSet.has(addr),
        prefix: addrPrefix(addr),
        in_lovelace: 0,
        out_lovelace: 0,
        volume_lovelace: 0,
        net_lovelace: 0,
        first_ts: null,
        last_ts: null,
        txHashes: new Set(),
      });
    }
    return nodeMap.get(addr);
  }

  for (const tx of txs) {
    const ts = Number(tx.timestamp_unix || 0);
    if (typeFilter && tx.type !== typeFilter) continue;
    if (dateFromUnix !== null && ts < dateFromUnix) continue;
    if (dateToUnix !== null && ts > dateToUnix) continue;

    let src = aggregateByAddress(tx.input_details || []);
    let dst = aggregateByAddress(tx.output_details || []);
    if (!src.length || !dst.length) continue;

    const hasOwned =
      src.some((x) => ownedSet.has(x.address) || x.owned) ||
      dst.some((x) => ownedSet.has(x.address) || x.owned);
    if (!hasOwned) continue;

    if (!includeExternal) {
      src = src.filter((x) => ownedSet.has(x.address) || x.owned);
      dst = dst.filter((x) => ownedSet.has(x.address) || x.owned);
      if (!src.length || !dst.length) continue;
    }

    if (src.length * dst.length > maxPairings) {
      const limit = Math.max(1, Math.floor(Math.sqrt(maxPairings)));
      src = topAddressesForPairing(src, limit, ownedSet);
      dst = topAddressesForPairing(dst, limit, ownedSet);
      const adjustedMaxDst = Math.max(1, Math.floor(maxPairings / Math.max(src.length, 1)));
      dst = dst.slice(0, adjustedMaxDst);
      if (!src.length || !dst.length) continue;
    }

    filteredTxCount += 1;
    src.forEach((s) => {
      const n = getNode(s.address);
      n.out_lovelace += Number(s.value_lovelace || 0);
      n.txHashes.add(tx.tx_hash);
      n.first_ts = n.first_ts === null ? ts : Math.min(n.first_ts, ts);
      n.last_ts = n.last_ts === null ? ts : Math.max(n.last_ts, ts);
    });
    dst.forEach((d) => {
      const n = getNode(d.address);
      n.in_lovelace += Number(d.value_lovelace || 0);
      n.txHashes.add(tx.tx_hash);
      n.first_ts = n.first_ts === null ? ts : Math.min(n.first_ts, ts);
      n.last_ts = n.last_ts === null ? ts : Math.max(n.last_ts, ts);
    });

    dst.forEach((d) => {
      const outputValue = Number(d.value_lovelace || 0);
      const share = outputValue / Math.max(src.length, 1);
      src.forEach((s) => {
        const key = `${s.address}->${d.address}`;
        const existing = linkMap.get(key) || {
          source: s.address,
          target: d.address,
          amount_lovelace: 0,
          tx_count: 0,
          first_ts: null,
          last_ts: null,
          types: new Set(),
          tx_samples: [],
        };
        existing.amount_lovelace += share;
        existing.tx_count += 1;
        existing.first_ts = existing.first_ts === null ? ts : Math.min(existing.first_ts, ts);
        existing.last_ts = existing.last_ts === null ? ts : Math.max(existing.last_ts, ts);
        existing.types.add(tx.type);
        if (existing.tx_samples.length < 14) {
          existing.tx_samples.push({
            tx_hash: tx.tx_hash,
            timestamp_utc: tx.timestamp_utc,
            type: tx.type,
            amount_ada: fmtAdaFromLovelace(share),
          });
        }
        linkMap.set(key, existing);
      });
    });
  }

  let links = [...linkMap.values()].filter((l) => Number(l.amount_lovelace) >= minLovelace);
  if (search) {
    links = links.filter((l) =>
      `${l.source} ${l.target} ${[...l.types].join(" ")}`.toLowerCase().includes(search)
    );
  }

  const connected = new Set();
  links.forEach((l) => {
    connected.add(l.source);
    connected.add(l.target);
  });

  let nodes = [...nodeMap.values()]
    .filter((n) => connected.has(n.id))
    .map((n) => {
      const volume = Number(n.in_lovelace || 0) + Number(n.out_lovelace || 0);
      const net = Number(n.in_lovelace || 0) - Number(n.out_lovelace || 0);
      return {
        ...n,
        tx_count: n.txHashes.size,
        volume_lovelace: volume,
        net_lovelace: net,
        txHashes: undefined,
        searchMatch: search
          ? `${n.address} ${n.prefix} ${n.owned ? "owned" : "external"}`
              .toLowerCase()
              .includes(search)
          : false,
      };
    });

  if (search) {
    const matchingIds = new Set(nodes.filter((n) => n.searchMatch).map((n) => n.id));
    if (matchingIds.size > 0) {
      links = links.filter((l) => matchingIds.has(l.source) || matchingIds.has(l.target));
      const filteredConnected = new Set();
      links.forEach((l) => {
        filteredConnected.add(l.source);
        filteredConnected.add(l.target);
      });
      nodes = nodes.filter((n) => filteredConnected.has(n.id));
    }
  }

  return {
    nodes,
    links,
    filteredTxCount,
  };
}

function graphNodeColor(node) {
  const mode = el("graph_color_mode").value;
  if (mode === "prefix") {
    const map = {
      addr1: "#59d6ff",
      Ae2: "#f2c265",
      Ddz: "#db88ff",
      stake1: "#8ee19d",
      other: "#c4d6dd",
    };
    return map[node.prefix] || map.other;
  }
  if (mode === "net") {
    return Number(node.net_lovelace) >= 0 ? "#6be1b7" : "#ff9f86";
  }
  return node.owned ? "#58e0aa" : "#ffb872";
}

function graphNodeMetric(node) {
  const metric = el("graph_size_metric").value;
  if (metric === "tx_count") return Number(node.tx_count || 0);
  if (metric === "net") return Math.abs(Number(node.net_lovelace || 0));
  return Number(node.volume_lovelace || 0);
}

function resetGraphSelection() {
  d3.selectAll(".graph-node").classed("active", false);
  d3.selectAll(".graph-link").classed("active", false);
}

function renderNodeDetail(node) {
  setDetail(
    `Wallet ${shortAddr(node.address)}`,
    [
      `Address: ${node.address}`,
      `Owned: ${node.owned ? "yes" : "no"}`,
      `Prefix: ${node.prefix}`,
      `First Seen: ${
        node.first_ts ? new Date(node.first_ts * 1000).toISOString() : "n/a"
      }`,
      `Last Seen: ${
        node.last_ts ? new Date(node.last_ts * 1000).toISOString() : "n/a"
      }`,
    ],
    [
      { k: "Transactions", v: String(node.tx_count) },
      { k: "Total In (ADA)", v: fmtAdaFromLovelace(node.in_lovelace) },
      { k: "Total Out (ADA)", v: fmtAdaFromLovelace(node.out_lovelace) },
      { k: "Net (ADA)", v: fmtAdaFromLovelace(node.net_lovelace) },
      { k: "Volume (ADA)", v: fmtAdaFromLovelace(node.volume_lovelace) },
    ]
  );
}

function renderLinkDetail(link) {
  const sample = (link.tx_samples || [])
    .slice(0, 12)
    .map((t) => `${t.timestamp_utc} | ${t.type} | ${t.amount_ada} ADA | ${t.tx_hash.slice(0, 16)}...`)
    .join("<br/>");

  setDetail(
    `Flow ${shortAddr(link.source.id || link.source)} -> ${shortAddr(
      link.target.id || link.target
    )}`,
    [
      `Source: ${link.source.id || link.source}`,
      `Target: ${link.target.id || link.target}`,
      `First Seen: ${
        link.first_ts ? new Date(link.first_ts * 1000).toISOString() : "n/a"
      }`,
      `Last Seen: ${link.last_ts ? new Date(link.last_ts * 1000).toISOString() : "n/a"}`,
    ],
    [
      { k: "Approx Total Flow (ADA)", v: fmtAdaFromLovelace(link.amount_lovelace) },
      { k: "Flow Events", v: String(link.tx_count) },
      { k: "Types", v: [...(link.types || [])].join(", ") || "n/a" },
      { k: "Sample Transactions", v: sample || "n/a" },
    ]
  );
}

function renderGraph() {
  if (!state.result) return;
  if (!window.d3) {
    setVizStatus("D3 failed to load. Check your internet connection and refresh.", true);
    return;
  }

  const graph = buildGraphData(state.result);
  const { nodes, links, filteredTxCount } = graph;
  const stats = `Visible wallets: ${nodes.length} | visible flows: ${links.length} | filtered txs: ${filteredTxCount}`;
  el("viz_stats").textContent = stats;

  const svg = d3.select("#wallet_graph");
  const parent = el("wallet_graph");
  const width = Math.max(parent.clientWidth, 300);
  const height = Math.max(parent.clientHeight, 280);

  svg.selectAll("*").remove();
  if (state.graph.simulation) {
    state.graph.simulation.stop();
    state.graph.simulation = null;
  }

  if (nodes.length === 0 || links.length === 0) {
    setVizStatus("No graph data for current filters.");
    setDetail("Selection", ["Adjust graph filters and refresh."], []);
    return;
  }

  setVizStatus("Graph updated.");

  svg.attr("viewBox", `0 0 ${width} ${height}`);

  const defs = svg.append("defs");
  defs
    .append("marker")
    .attr("id", "arrow")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 16)
    .attr("refY", 0)
    .attr("markerWidth", 5)
    .attr("markerHeight", 5)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "rgba(152,205,221,0.6)");

  const viewport = svg.append("g");
  state.graph.viewport = viewport;
  state.graph.svg = svg;

  const metricValues = nodes.map(graphNodeMetric).filter((v) => Number.isFinite(v));
  const nodeRadius = d3
    .scaleSqrt()
    .domain([Math.max(0, d3.min(metricValues) || 0), Math.max(1, d3.max(metricValues) || 1)])
    .range([4, 20]);
  const linkWidth = d3
    .scaleSqrt()
    .domain([0, Math.max(1, d3.max(links, (d) => Number(d.amount_lovelace || 0)) || 1)])
    .range([0.5, 4.5]);

  const link = viewport
    .append("g")
    .attr("stroke-linecap", "round")
    .selectAll("line")
    .data(links)
    .enter()
    .append("line")
    .attr("class", "graph-link")
    .attr("stroke-width", (d) => linkWidth(Number(d.amount_lovelace || 0)))
    .attr("stroke-opacity", 0.6)
    .attr("marker-end", "url(#arrow)");

  link
    .append("title")
    .text(
      (d) =>
        `${d.source} -> ${d.target}\nflow: ${fmtAdaFromLovelace(d.amount_lovelace)} ADA\nevents: ${
          d.tx_count
        }`
    );

  const node = viewport
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("class", "graph-node")
    .attr("r", (d) => nodeRadius(graphNodeMetric(d)))
    .attr("fill", (d) => graphNodeColor(d))
    .attr("opacity", (d) =>
      el("graph_search").value.trim() && !d.searchMatch ? 0.35 : 0.95
    );

  node
    .append("title")
    .text(
      (d) =>
        `${d.address}\nowned: ${d.owned}\ntransactions: ${d.tx_count}\nin: ${fmtAdaFromLovelace(
          d.in_lovelace
        )} ADA\nout: ${fmtAdaFromLovelace(d.out_lovelace)} ADA`
    );

  const showLabels = el("graph_show_labels").checked;
  const label = viewport
    .append("g")
    .selectAll("text")
    .data(showLabels ? nodes : [])
    .enter()
    .append("text")
    .attr("class", "graph-label")
    .text((d) => shortAddr(d.address));

  const simulation = d3
    .forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(98).strength(0.18))
    .force("charge", d3.forceManyBody().strength(-180))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius((d) => nodeRadius(graphNodeMetric(d)) + 4));

  node.call(
    d3
      .drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
  );

  node.on("click", (_event, d) => {
    resetGraphSelection();
    d3.select(_event.currentTarget).classed("active", true);
    renderNodeDetail(d);
  });

  link.on("click", (_event, d) => {
    resetGraphSelection();
    d3.select(_event.currentTarget).classed("active", true);
    renderLinkDetail(d);
  });

  svg.on("click", (event) => {
    if (event.target === svg.node()) {
      resetGraphSelection();
      setDetail(
        "Selection",
        ["Click a wallet bubble or flow line for full details."],
        []
      );
    }
  });

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

    label.attr("x", (d) => d.x + 6).attr("y", (d) => d.y - 6);
  });

  const zoom = d3
    .zoom()
    .scaleExtent([0.2, 6])
    .on("zoom", (event) => viewport.attr("transform", event.transform));
  svg.call(zoom);

  state.graph.simulation = simulation;
  state.graph.zoom = zoom;
}

function resetGraphZoom() {
  if (!state.graph.svg || !state.graph.zoom) return;
  state.graph.svg.transition().duration(250).call(state.graph.zoom.transform, d3.zoomIdentity);
}

function refreshGraph() {
  try {
    renderGraph();
  } catch (err) {
    setVizStatus(err.message || String(err), true);
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
  setVizStatus("Waiting for reconciliation...");
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
    populateTypeFilters(data.transactions || []);
    initializeGraphRange(data.transactions || []);
    renderTransactions();
    refreshGraph();
    setStatus(
      `Done. ${data.summary.transactions_total} transactions reconciled. Saved to output/latest-reconciliation.json`
    );
  } catch (err) {
    setStatus(err.message || String(err), true);
    setVizStatus("Reconciliation failed. Graph unavailable.", true);
  } finally {
    btn.disabled = false;
  }
}

el("run_btn").addEventListener("click", runReconcile);
el("type_filter").addEventListener("change", renderTransactions);
el("text_filter").addEventListener("input", renderTransactions);
el("limit_filter").addEventListener("input", renderTransactions);

el("graph_refresh_btn").addEventListener("click", refreshGraph);
el("graph_reset_zoom_btn").addEventListener("click", resetGraphZoom);

[
  "graph_type_filter",
  "graph_date_from",
  "graph_date_to",
  "graph_min_ada",
  "graph_include_external",
  "graph_size_metric",
  "graph_color_mode",
  "graph_show_labels",
  "graph_search",
  "graph_max_pairings",
].forEach((id) => {
  el(id).addEventListener("input", refreshGraph);
  el(id).addEventListener("change", refreshGraph);
});

loadDefaults().catch(() => {});

window.addEventListener("resize", () => {
  if (state.result) refreshGraph();
});
