export type ModelId = "logistic_regression" | "decision_tree" | "random_forest";
export type PipelineId = "conventional" | "agentic";

export interface WorkflowStep {
  id: string;
  order: number;
  title: string;
  short: string;
  why: string;
  how: string;
  avoids: string;
  takeaway: string;
  input: string;
  output: string;
  source_files: string[];
}

export interface MetricRow {
  pipeline?: PipelineId;
  pipeline_label?: string;
  model: ModelId;
  model_label: string;
  [key: string]: string | number | boolean | undefined;
}

export interface CurveRow {
  pipeline: PipelineId;
  pipeline_label: string;
  model: ModelId;
  model_label: string;
  pr_auc: number;
  roc_auc: number;
  pr_points: { recall: number; precision: number }[];
  roc_points: { fpr: number; tpr: number }[];
}

export interface ConfusionMatrixRow {
  pipeline: PipelineId;
  pipeline_label: string;
  model: ModelId;
  model_label: string;
  threshold: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface ExplorerPayload {
  snapshot_version: string;
  generated_at_utc: string;
  source_run_id: string;
  methodology: {
    title: string;
    primary_metric: string;
    decision_threshold: number;
    models: { id: ModelId; label: string }[];
    pipelines: { id: PipelineId; label: string }[];
    cross_validation: { folds: number; repetitions: number; pairs_per_model: number };
  };
  learning: {
    central_question: string;
    experiment_layers: { title: string; question: string; evidence: string; allowed_conclusion: string }[];
    glossary: { term: string; plain: string; why: string }[];
  };
  langgraph: {
    library: string;
    source_file: string;
    graph_role: string;
    nodes: { id: string; label: string; kind: string; x: number; y: number; reads: string[]; writes: string[]; responsibility: string; why: string; implementation: string }[];
    edges: { source: string; target: string; label: string; condition?: boolean }[];
    state_fields: { name: string; lifetime: string; contains: string; excludes: string }[];
    official_trace: { node: string; title: string; detail: string; state_change: string }[];
    invalid_trace: { node: string; title: string; detail: string; state_change: string }[];
    design_decisions: { choice: string; reason: string }[];
    official_run: { run_id: string; iterations: number; termination: string; max_iterations: number };
  };
  workflow: WorkflowStep[];
  datasets: Record<string, any>;
  quality: Record<string, any>;
  cross_validation: {
    conventional_metrics: MetricRow[];
    agentic_metrics: MetricRow[];
    paired_tests: MetricRow[];
    significant_tests: MetricRow[];
    oof_curves: CurveRow[];
    curve_note: string;
  };
  temporal_evaluation: {
    metrics: MetricRow[];
    curves: CurveRow[];
    confusion_matrices: ConfusionMatrixRow[];
    interpretation: string;
  };
  resources: Record<string, any>;
  selected_plan: Record<string, any>;
  caveats: string[];
}
