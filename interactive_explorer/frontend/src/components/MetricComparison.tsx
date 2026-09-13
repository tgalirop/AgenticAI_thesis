import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricRow } from "../types";
import { ChartFrame } from "./ChartFrame";

interface MetricComparisonProps {
  rows: MetricRow[];
  metric: string;
  title: string;
  insight: string;
}

export function MetricComparison({ rows, metric, title, insight }: MetricComparisonProps) {
  const data = Array.from(new Set(rows.map((row) => row.model))).map((model) => {
    const matching = rows.filter((row) => row.model === model);
    return {
      model: matching[0].model_label.replace("Logistic Regression", "Logistic Reg."),
      conventional: Number(matching.find((row) => row.pipeline === "conventional")?.[metric] ?? 0),
      agentic: Number(matching.find((row) => row.pipeline === "agentic")?.[metric] ?? 0),
    };
  });
  return (
    <ChartFrame title={title} subtitle="Conventional και Agentic αξιολογούνται στις ίδιες 123.580 test γραμμές." insight={insight} caution="η ψηλότερη μπάρα αποδεικνύει καθολική υπεροχή. Πρέπει να εξετάσουμε το μέγεθος delta, το μοντέλο, τη μετρική και τα false positives/negatives.">
      <ResponsiveContainer width="100%" height={310}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="model" />
          <YAxis domain={[0, "auto"]} />
          <Tooltip formatter={(value) => Number(value ?? 0).toFixed(6)} />
          <Legend />
          <Bar name="Conventional" dataKey="conventional" fill="#315f78" radius={[5, 5, 0, 0]} />
          <Bar name="Agentic" dataKey="agentic" fill="#dc7655" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
