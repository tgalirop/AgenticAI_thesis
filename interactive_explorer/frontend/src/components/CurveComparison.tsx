import { useMemo } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CurveRow, ModelId } from "../types";
import { ChartFrame } from "./ChartFrame";

interface CurveComparisonProps {
  curves: CurveRow[];
  model: ModelId;
  kind: "pr" | "roc";
  scopeLabel: string;
}

const colors = { conventional: "#315f78", agentic: "#dc7655" };

function interpolate(points: { x: number; y: number }[], x: number): number | null {
  if (!points.length || x < points[0].x || x > points[points.length - 1].x) return null;
  for (let index = 1; index < points.length; index += 1) {
    const left = points[index - 1];
    const right = points[index];
    if (x <= right.x) {
      const width = right.x - left.x;
      return width === 0 ? right.y : left.y + ((x - left.x) / width) * (right.y - left.y);
    }
  }
  return points.at(-1)?.y ?? null;
}

/** Draw both pipelines on a common x-grid so tooltips compare equal coordinates. */
export function CurveComparison({ curves, model, kind, scopeLabel }: CurveComparisonProps) {
  const selected = curves.filter((curve) => curve.model === model);
  const chartData = useMemo(() => {
    const grid = Array.from({ length: 101 }, (_, index) => index / 100);
    return grid.map((x) => {
      const row: Record<string, number | null> = { x };
      selected.forEach((curve) => {
        const points = kind === "pr"
          ? curve.pr_points.map((point) => ({ x: point.recall, y: point.precision }))
          : curve.roc_points.map((point) => ({ x: point.fpr, y: point.tpr }));
        row[curve.pipeline] = interpolate(points, x);
      });
      return row;
    });
  }, [selected, kind]);

  const isPr = kind === "pr";
  const title = isPr ? "Precision–Recall curve" : "ROC curve";
  const aucName = isPr ? "PR-AUC" : "ROC-AUC";
  const subtitle = selected.map((curve) => `${curve.pipeline_label} ${aucName}: ${(isPr ? curve.pr_auc : curve.roc_auc).toFixed(4)}`).join(" · ");
  const insight = isPr
    ? "Κάθε σημείο αντιστοιχεί σε διαφορετικό threshold: αν χαμηλώσουμε το threshold, συνήθως βρίσκουμε περισσότερα fraud (Recall ↑), αλλά χαρακτηρίζουμε περισσότερα normal ως fraud (Precision ↓). Πιο πάνω και δεξιά είναι γενικά καλύτερα."
    : "Κάθε σημείο αντιστοιχεί σε threshold και δείχνει πόσο αυξάνεται το True Positive Rate σε σχέση με το False Positive Rate. Η διαγώνιος είναι τυχαίο ranking και καμπύλη πιο κοντά στην πάνω αριστερή γωνία είναι καλύτερη.";
  const caution = isPr
    ? "η καμπύλη επιλέγει από μόνη της επιχειρησιακό threshold ή ότι μία μικρή AUC διαφορά είναι αυτομάτως σημαντική. Η fraud prevalence επηρεάζει έντονα την PR-AUC."
    : "υψηλό ROC-AUC συνεπάγεται καλό Precision. Με εκατομμύρια normal συναλλαγές, μικρό FPR μπορεί ακόμη να δημιουργήσει πολλούς false positives.";

  return (
    <ChartFrame title={`${title} — ${scopeLabel}`} subtitle={subtitle} insight={insight} caution={caution}>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={chartData} margin={{ top: 10, right: 15, left: 0, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e4e5df" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(value) => value.toFixed(1)} label={{ value: isPr ? "Recall" : "False Positive Rate", position: "insideBottom", offset: -8 }} />
          <YAxis domain={[0, 1]} tickFormatter={(value) => value.toFixed(1)} label={{ value: isPr ? "Precision" : "True Positive Rate", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(value) => Number(value ?? 0).toFixed(4)} labelFormatter={(value) => `${isPr ? "Recall" : "FPR"}: ${Number(value).toFixed(2)}`} />
          <Legend verticalAlign="top" height={32} />
          {!isPr && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#999" strokeDasharray="5 5" />}
          <Line type="monotone" dataKey="conventional" name="Conventional" dot={false} stroke={colors.conventional} strokeWidth={2.5} connectNulls />
          <Line type="monotone" dataKey="agentic" name="Agentic" dot={false} stroke={colors.agentic} strokeWidth={2.5} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
