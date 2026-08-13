import { useEffect, useMemo, useState } from "react";
import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2, GitBranch, Pause, Play, RotateCcw, ShieldCheck } from "lucide-react";
import type { ExplorerPayload } from "../types";

type Scenario = "official" | "invalid";

const virtualPositions: Record<string, { x: number; y: number }> = {
  START: { x: -170, y: 210 }, END: { x: 1730, y: 210 },
};

export function LangGraphLab({ graph }: { graph: ExplorerPayload["langgraph"] }) {
  const [scenario, setScenario] = useState<Scenario>("official");
  const [stepIndex, setStepIndex] = useState(0);
  const [selectedNode, setSelectedNode] = useState(graph.nodes[0].id);
  const [playing, setPlaying] = useState(false);
  const trace = scenario === "official" ? graph.official_trace : graph.invalid_trace;
  const active = trace[Math.min(stepIndex, trace.length - 1)];
  const selected = graph.nodes.find((node) => node.id === selectedNode);

  const reset = (nextScenario = scenario) => {
    setScenario(nextScenario); setStepIndex(0); setPlaying(false);
    const nextTrace = nextScenario === "official" ? graph.official_trace : graph.invalid_trace;
    setSelectedNode(nextTrace[0].node);
  };

  // A short timer gives a presentation-friendly playback without executing the Agent.
  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setTimeout(() => {
      if (stepIndex >= trace.length - 1) setPlaying(false);
      else { const next = stepIndex + 1; setStepIndex(next); setSelectedNode(trace[next].node); }
    }, 1800);
    return () => window.clearTimeout(timer);
  }, [playing, stepIndex, trace]);

  const nodes = useMemo<Node[]>(() => {
    const domainNodes = graph.nodes.map((item) => ({
      id: item.id,
      position: { x: item.x, y: item.y },
      data: { label: item.label },
      className: `lang-node ${item.kind} ${active.node === item.id ? "active" : ""}`,
    }));
    return [
      { id: "START", position: virtualPositions.START, data: { label: "START" }, className: `lang-node terminal ${active.node === "START" ? "active" : ""}` },
      ...domainNodes,
      { id: "END", position: virtualPositions.END, data: { label: "END" }, className: `lang-node terminal ${active.node === "END" ? "active" : ""}` },
    ];
  }, [graph.nodes, active.node]);
  const edges = useMemo<Edge[]>(() => graph.edges.map((item, index) => ({
    id: `lang-edge-${index}`, source: item.source, target: item.target, label: item.label,
    animated: item.source === trace[Math.max(0, stepIndex - 1)]?.node && item.target === active.node,
    markerEnd: { type: MarkerType.ArrowClosed },
    className: `lang-edge ${item.condition ? "conditional" : ""}`,
    style: { stroke: item.label === "invalid" ? "#b45747" : item.label === "RETRY" ? "#b77a3d" : "#477665", strokeWidth: 2 },
  })), [graph.edges, trace, stepIndex, active.node]);

  const go = (direction: -1 | 1) => {
    const next = Math.min(Math.max(stepIndex + direction, 0), trace.length - 1);
    setStepIndex(next); setSelectedNode(trace[next].node); setPlaying(false);
  };

  return <div className="langgraph-lab">
    <div className="lab-toolbar">
      <div><span className="eyebrow">Guided state simulation</span><strong>{graph.library}</strong></div>
      <div className="scenario-tabs">
        <button className={scenario === "official" ? "active" : ""} onClick={() => reset("official")}><CheckCircle2 /> Επίσημο ACCEPT run</button>
        <button className={scenario === "invalid" ? "active" : ""} onClick={() => reset("invalid")}><AlertTriangle /> Invalid → RETRY</button>
      </div>
    </div>
    <div className="langgraph-canvas">
      <ReactFlow nodes={nodes} edges={edges} onNodeClick={(_, node) => setSelectedNode(node.id)} nodesDraggable={false} nodesConnectable={false} fitView fitViewOptions={{ padding: .1 }} minZoom={.35}>
        <Background gap={20} color="#d9ddd6" /><Controls showInteractive={false} />
      </ReactFlow>
    </div>
    <div className="playback-panel">
      <div className="playback-controls"><button onClick={() => reset()} title="Επανεκκίνηση"><RotateCcw /></button><button onClick={() => go(-1)} disabled={stepIndex === 0}><ArrowLeft /></button><button className="play" onClick={() => setPlaying((value) => !value)}>{playing ? <Pause /> : <Play />}</button><button onClick={() => go(1)} disabled={stepIndex === trace.length - 1}><ArrowRight /></button><span>Βήμα {stepIndex + 1}/{trace.length}</span></div>
      <div className="trace-card"><span className="node-chip">{active.node}</span><h3>{active.title}</h3><p>{active.detail}</p><code>{active.state_change}</code></div>
    </div>
    {selected && <div className="node-inspector"><header><GitBranch /><div><span className="eyebrow">Node inspector</span><h3>{selected.label}</h3><code>{selected.implementation}</code></div></header><p>{selected.responsibility}</p><div className="node-why"><ShieldCheck /><span><strong>Γιατί υπάρχει ως ξεχωριστό node:</strong> {selected.why}</span></div><div className="io-grid"><div><strong>Διαβάζει από το state</strong>{selected.reads.length ? selected.reads.map((item) => <code key={item}>{item}</code>) : <span>Κανένα πεδίο</span>}</div><div><strong>Γράφει στο state</strong>{selected.writes.length ? selected.writes.map((item) => <code key={item}>{item}</code>) : <span>Καμία μεταβολή</span>}</div></div></div>}
  </div>;
}
