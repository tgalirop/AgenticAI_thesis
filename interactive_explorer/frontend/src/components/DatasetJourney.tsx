import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ExplorerPayload } from "../types";
import { ChartFrame } from "./ChartFrame";

interface DatasetJourneyProps {
  data: ExplorerPayload["datasets"];
  quality: ExplorerPayload["quality"];
}

const stageColors = ["#315f78", "#3d7a68", "#dc7655", "#8e6c88"];

export function DatasetJourney({ data, quality }: DatasetJourneyProps) {
  const sizes = [
    { name: "Raw", rows: data.raw.rows },
    { name: "Training", rows: data.training.rows },
    { name: "10:1 sample", rows: data.sample.rows },
    { name: "Test", rows: data.test_dataset.rows },
  ];
  const prevalence = data.prevalence.map((item: { dataset: string; value: number }) => ({
    ...item,
    percentage: item.value * 100,
  }));

  return (
    <div className="chart-grid">
      <ChartFrame
        title="Πορεία μεγέθους των datasets"
        subtitle="Οι μπάρες είναι λογαριθμικές για να παραμένει ορατό το sample."
        insight="Το test dataset δεν προκύπτει από το sample. Αποκόπτεται χρονικά από το πλήρες dataset και μένει κλειδωμένο, ενώ το sample προέρχεται μόνο από το training partition."
        caution="το test dataset είναι το 20% των γραμμών. Είναι περίπου το 20% των διακριτών χρονικών steps."
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sizes} margin={{ top: 25, right: 10, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" />
            <YAxis scale="log" domain={[10_000, 10_000_000]} tickFormatter={(value) => Intl.NumberFormat("el-GR", { notation: "compact" }).format(value)} />
            <Tooltip formatter={(value) => Number(value ?? 0).toLocaleString("el-GR")} />
            <Bar dataKey="rows" radius={[7, 7, 0, 0]}>
              {sizes.map((_, index) => <Cell key={index} fill={stageColors[index]} />)}
              <LabelList dataKey="rows" position="top" formatter={(value) => Number(value ?? 0).toLocaleString("el-GR")} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>

      <ChartFrame
        title="Fraud prevalence ανά σύνολο"
        subtitle="Ποσοστό fraud μέσα σε κάθε dataset."
        insight="Το 10:1 sample έχει 9,09% fraud, ενώ το test dataset 1,34%. Γι’ αυτό τα απόλυτα CV και test PR-AUC/Precision δεν συγκρίνονται απευθείας."
        caution="το μοντέλο θα συναντά 9,09% fraud στην πραγματικότητα. Η αναλογία δημιουργήθηκε μόνο για το controlled training experiment."
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={prevalence} layout="vertical" margin={{ top: 5, right: 40, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(value) => `${value}%`} />
            <YAxis type="category" dataKey="dataset" width={110} />
            <Tooltip formatter={(value) => `${Number(value ?? 0).toFixed(3)}%`} />
            <Bar dataKey="percentage" fill="#dc7655" radius={[0, 7, 7, 0]}>
              <LabelList dataKey="percentage" position="right" formatter={(value) => `${Number(value ?? 0).toFixed(3)}%`} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>

      <article className="degradation-card">
        <div>
          <span className="eyebrow">Controlled degradation</span>
          <h3>5% MCAR στη <code>type</code></h3>
          <p>Από τις {data.sample.rows.toLocaleString("el-GR")} γραμμές, ακριβώς <strong>{quality.degradation.affected_rows.toLocaleString("el-GR")}</strong> θέσεις επιλέχθηκαν τυχαία με seed {quality.degradation.seed}.</p>
        </div>
        <div className="missing-demo" aria-label="Παράδειγμα MCAR missingness">
          <div><span>TRANSFER</span><span className="arrow">→</span><span className="missing">missing</span></div>
          <div><span>CASH_OUT</span><span className="arrow">→</span><span>CASH_OUT</span></div>
          <div><span>PAYMENT</span><span className="arrow">→</span><span className="missing">missing</span></div>
        </div>
        <p className="callout">Το test dataset δεν υποβαθμίστηκε. Το πείραμα ελέγχει training σε degraded ιστορικό sample και inference σε καθαρά μεταγενέστερα δεδομένα.</p>
      </article>
    </div>
  );
}
