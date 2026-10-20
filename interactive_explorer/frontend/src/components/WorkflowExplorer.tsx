import { useMemo, useState, type ReactNode } from "react";
import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { ArrowRight, Bot, Braces, Database, FileCode2, GitCompareArrows, LockKeyhole, Network, ScanSearch, ShieldCheck, Split, TestTube2 } from "lucide-react";
import type { WorkflowStep } from "../types";

interface WorkflowExplorerProps {
  steps: WorkflowStep[];
}

type ArchitectureNode = {
  id: string;
  detailId: string;
  x: number;
  y: number;
  phase: string;
  title: string;
  subtitle: string;
  badge?: string;
  variant: string;
  icon: ReactNode;
};

/** Rich card displayed inside a React Flow node. */
function ArchitectureCard({ item, active }: { item: ArchitectureNode; active: boolean }) {
  return <div className={`architecture-card ${item.variant} ${active ? "active" : ""}`}>
    <div className="architecture-card-top"><span className="phase-label">{item.phase}</span>{item.badge && <span className="architecture-badge">{item.badge}</span>}</div>
    <div className="architecture-card-body"><div className="architecture-icon">{item.icon}</div><div><strong>{item.title}</strong><span>{item.subtitle}</span></div></div>
  </div>;
}

/**
 * Vertical architectural map of the real experiment.
 *
 * The diagram deliberately contains one virtual "Locked test dataset" storage node:
 * it makes the test branch visible without pretending that another
 * executable pipeline stage exists. Clicking it opens the chronological split detail.
 */
export function WorkflowExplorer({ steps }: WorkflowExplorerProps) {
  const [selectedId, setSelectedId] = useState(steps[0]?.id);
  const selected = steps.find((step) => step.id === selectedId) ?? steps[0];
  const detailById = useMemo(() => new Map(steps.map((step) => [step.id, step])), [steps]);

  const architecture = useMemo<ArchitectureNode[]>(() => [
    { id: "raw", detailId: "raw", x: 260, y: 0, phase: "01 · DATA SOURCE", title: "Raw PaySim CSV", subtitle: "6.362.620 συναλλαγές", badge: "INPUT", variant: "source", icon: <Database /> },
    { id: "streaming", detailId: "streaming", x: 260, y: 135, phase: "02 · INGESTION", title: "Streaming & schema", subtitle: "Lazy CSV → validated Parquet", variant: "data", icon: <ScanSearch /> },
    { id: "features", detailId: "features", x: 260, y: 270, phase: "03 · PREPARATION", title: "Leakage-safe features", subtitle: "Row-wise modeling table", variant: "data", icon: <Braces /> },
    { id: "test", detailId: "test", x: 260, y: 405, phase: "04 · TEST BOUNDARY", title: "Chronological split", subtitle: "Πλήρη time steps · όχι random rows", badge: "CRITICAL", variant: "split", icon: <Split /> },
    { id: "sample", detailId: "sample", x: 35, y: 575, phase: "TRAINING PATH", title: "Reproducible sample", subtitle: "72.149 rows · normal:fraud 10:1", variant: "training", icon: <TestTube2 /> },
    { id: "locked_test_dataset", detailId: "test", x: 485, y: 575, phase: "FINAL TEST PATH", title: "Locked test dataset", subtitle: "123.580 rows · steps 595–743", badge: "LOCKED", variant: "locked", icon: <LockKeyhole /> },
    { id: "mcar", detailId: "mcar", x: 35, y: 730, phase: "CONTROLLED EXPERIMENT", title: "5% MCAR degradation", subtitle: "3.607 missing values στη type", variant: "degradation", icon: <Network /> },
    { id: "conventional", detailId: "pipelines", x: -55, y: 900, phase: "PIPELINE A", title: "Conventional benchmark", subtitle: "Fixed preprocessing + 3 models", variant: "conventional", icon: <ShieldCheck /> },
    { id: "agentic", detailId: "agent", x: 220, y: 900, phase: "PIPELINE B", title: "LangGraph Agent", subtitle: "Plan → validate → execute → feedback", variant: "agentic", icon: <Bot /> },
    { id: "paired", detailId: "paired", x: 82, y: 1070, phase: "TRAINING EVIDENCE", title: "Paired 5×5 comparison", subtitle: "3 models · 25 κοινά folds", variant: "comparison", icon: <GitCompareArrows /> },
    { id: "freeze", detailId: "freeze", x: 82, y: 1215, phase: "DECISION GATE", title: "Freeze selected plan", subtitle: detailById.get("freeze")?.node_subtitle ?? "Selected plan · frozen", badge: "FROZEN", variant: "freeze", icon: <LockKeyhole /> },
    { id: "test_dataset", detailId: "test_dataset", x: 320, y: 1390, phase: "FINAL EVALUATION", title: "Test dataset", subtitle: "Conventional vs Agentic · once", variant: "final", icon: <TestTube2 /> },
    { id: "statistics", detailId: "statistics", x: 320, y: 1540, phase: "THESIS EVIDENCE", title: "Results & interpretation", subtitle: "Metrics · statistics · cost · figures", badge: "OUTPUT", variant: "output", icon: <FileCode2 /> },
  ], [detailById]);

  const nodes = useMemo<Node[]>(() => architecture.map((item) => ({
    id: item.id,
    position: { x: item.x, y: item.y },
    data: { label: <ArchitectureCard item={item} active={selectedId === item.detailId} /> },
    className: "architecture-flow-node",
  })), [architecture, selectedId]);

  const edge = (id: string, source: string, target: string, label?: string, tone = "main"): Edge => ({
    id, source, target, label, type: "smoothstep", animated: tone === "agentic",
    markerEnd: { type: MarkerType.ArrowClosed },
    className: `architecture-edge ${tone}`,
    labelStyle: { fontSize: 10, fontWeight: 700, fill: "#56655f" },
    labelBgStyle: { fill: "#f8f7f2", fillOpacity: .95 },
    style: { strokeWidth: 2.3 },
  });
  const edges = useMemo<Edge[]>(() => [
    edge("e1", "raw", "streaming"), edge("e2", "streaming", "features"), edge("e3", "features", "test"),
    edge("e4", "test", "sample", "steps 1–594", "training"), edge("e5", "test", "locked_test_dataset", "steps 595–743", "locked"),
    edge("e6", "sample", "mcar", "training copy", "training"),
    edge("e7", "mcar", "conventional", "ίδιες missing θέσεις", "conventional"), edge("e8", "mcar", "agentic", "ίδιες missing θέσεις", "agentic"),
    edge("e9", "conventional", "paired", "fold metrics", "conventional"), edge("e10", "agentic", "paired", "fold metrics", "agentic"),
    edge("e11", "paired", "freeze", "training-only decision", "training"),
    edge("e12", "freeze", "test_dataset", "frozen plan", "agentic"), edge("e13", "locked_test_dataset", "test_dataset", "opens once", "locked"),
    edge("e14", "test_dataset", "statistics", "final evidence", "final"),
  ], []);

  return (
    <section className="workflow-layout architectural">
      <div className="architecture-stage">
        <div className="architecture-legend"><span><i className="legend-training" />Training</span><span><i className="legend-agentic" />Agentic</span><span><i className="legend-locked" />Locked test</span></div>
        <div className="workflow-canvas architectural" aria-label="Κάθετο αρχιτεκτονικό workflow">
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodeClick={(_, node) => {
              const item = architecture.find((candidate) => candidate.id === node.id);
              if (item && detailById.has(item.detailId)) setSelectedId(item.detailId);
            }}
            nodesDraggable={false} nodesConnectable={false} elementsSelectable
            fitView fitViewOptions={{ padding: 0.07, minZoom: .58 }} minZoom={0.42} maxZoom={1.35}
          >
            <Background color="#d7ddd7" gap={22} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </div>
      {selected && (
        <aside className="step-panel architectural-detail">
          <span className="eyebrow">Μεθοδολογικό βήμα {selected.order} / {steps.length}</span>
          <h3>{selected.title}</h3>
          <p className="step-summary">{selected.short}</p>
          <dl>
            <div><dt>Είσοδος</dt><dd>{selected.input}</dd></div>
            <div><dt>Έξοδος</dt><dd>{selected.output}</dd></div>
            <div><dt>Γιατί γίνεται</dt><dd>{selected.why}</dd></div>
            <div><dt>Πώς ακριβώς γίνεται</dt><dd>{selected.how}</dd></div>
            <div><dt>Τι πρόβλημα αποφεύγουμε</dt><dd>{selected.avoids}</dd></div>
            <div className="takeaway"><dt>Αυτό να κρατήσεις</dt><dd>{selected.takeaway}</dd></div>
          </dl>
          <div className="source-list"><FileCode2 size={18} /><div><strong>Πραγματικός κώδικας</strong>{selected.source_files.map((file) => <code key={file}>{file}</code>)}</div></div>
          <button className="next-button" onClick={() => {
            const nextIndex = Math.min(selected.order, steps.length - 1);
            setSelectedId(steps[nextIndex].id);
          }} disabled={selected.order === steps.length}>Επόμενο μεθοδολογικό βήμα <ArrowRight size={17} /></button>
        </aside>
      )}
    </section>
  );
}
