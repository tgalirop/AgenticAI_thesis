import type { ReactNode } from "react";
import { Info } from "lucide-react";

interface ChartFrameProps {
  title: string;
  subtitle: string;
  insight: string;
  caution?: string;
  children: ReactNode;
}

/** Consistent scientific chart container with an explicit interpretation. */
export function ChartFrame({ title, subtitle, insight, caution, children }: ChartFrameProps) {
  return (
    <article className="chart-card">
      <header>
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </header>
      <div className="chart-area">{children}</div>
      <div className="chart-insight">
        <Info size={17} aria-hidden="true" />
        <span>{insight}</span>
      </div>
      {caution && <div className="chart-caution"><strong>Δεν σημαίνει ότι:</strong> {caution}</div>}
    </article>
  );
}
