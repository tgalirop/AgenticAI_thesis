import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricRow } from "../types";
import { ChartFrame } from "./ChartFrame";

export function StatisticsPanel({ tests }: { tests: MetricRow[] }) {
  const data = tests.map((test) => ({
    name: `${test.model_label.replace("Logistic Regression", "Logistic").replace("Random Forest", "RF")} · ${String(test.metric).toUpperCase()}`,
    delta: Number(test.mean_delta),
    significant: Boolean(test.significant_0_05),
    p: Number(test.p_value_holm),
  }));
  return (
    <>
      <ChartFrame
        title="Μέση paired διαφορά ανά model και metric"
        subtitle="Διαφορά = Agentic − Conventional στα ίδια 25 folds. Πορτοκαλί: significant μετά Holm."
        insight="Θετική μπάρα ευνοεί το Agentic και αρνητική το Conventional. Η στατιστική σημαντικότητα δείχνει συνέπεια, όχι απαραίτητα μεγάλο πρακτικό όφελος."
      >
        <ResponsiveContainer width="100%" height={430}>
          <BarChart data={data} layout="vertical" margin={{ left: 45, right: 25 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={["dataMin", "dataMax"]} tickFormatter={(value) => value.toFixed(3)} />
            <YAxis type="category" dataKey="name" width={145} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value) => Number(value ?? 0).toFixed(6)} />
            <ReferenceLine x={0} stroke="#333" />
            <Bar dataKey="delta" name="Mean delta" radius={[0, 5, 5, 0]}>
              {data.map((row, index) => <Cell key={index} fill={row.significant ? "#dc7655" : "#aeb8b0"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Μοντέλο</th><th>Μετρική</th><th>Mean delta</th><th>Holm p</th><th>Συμπέρασμα</th></tr></thead>
          <tbody>{tests.map((test) => (
            <tr key={`${test.model}-${test.metric}`}>
              <td>{test.model_label}</td><td>{String(test.metric).toUpperCase()}</td>
              <td className={Number(test.mean_delta) >= 0 ? "positive" : "negative"}>{Number(test.mean_delta).toFixed(6)}</td>
              <td>{Number(test.p_value_holm).toExponential(3)}</td>
              <td><span className={test.significant_0_05 ? "status significant" : "status"}>{test.significant_0_05 ? "Σημαντικό" : "Μη σημαντικό"}</span></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}
