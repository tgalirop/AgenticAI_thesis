import type { ConfusionMatrixRow } from "../types";

interface ConfusionMatrixProps {
  matrix: ConfusionMatrixRow;
}

function Cell({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className={`matrix-cell ${tone}`}>
      <strong>{value.toLocaleString("el-GR")}</strong>
      <span>{label}</span>
    </div>
  );
}

/** A labelled 2×2 matrix that keeps predicted and actual axes explicit. */
export function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  const total = matrix.tn + matrix.fp + matrix.fn + matrix.tp;
  return (
    <article className="matrix-card">
      <header>
        <span className={`pipeline-pill ${matrix.pipeline}`}>{matrix.pipeline_label}</span>
        <h4>{matrix.model_label}</h4>
        <small>Threshold {matrix.threshold.toFixed(1)} · n={total.toLocaleString("el-GR")}</small>
      </header>
      <div className="matrix-axis top">Πραγματική κλάση →</div>
      <div className="matrix-grid">
        <div className="matrix-axis side">Πρόβλεψη</div>
        <div className="matrix-label">Normal</div>
        <div className="matrix-label">Fraud</div>
        <div className="matrix-label">Normal</div>
        <Cell label="True Negative" value={matrix.tn} tone="correct" />
        <Cell label="False Negative" value={matrix.fn} tone="error" />
        <div className="matrix-label">Fraud</div>
        <Cell label="False Positive" value={matrix.fp} tone="warning" />
        <Cell label="True Positive" value={matrix.tp} tone="correct strong" />
      </div>
    </article>
  );
}
