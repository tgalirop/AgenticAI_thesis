import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BrainCircuit, Clock3, Coins, Cpu, MemoryStick } from "lucide-react";
import { ChartFrame } from "./ChartFrame";

export function ResourcesPanel({ resources }: { resources: Record<string, any> }) {
  const conventional = resources.memory.degraded_conventional_benchmark.peak_increase_megabytes;
  const agentic = resources.memory.selected_agentic_candidate.peak_increase_megabytes;
  const memory = [{ name: "Conventional", value: conventional }, { name: "Agentic", value: agentic }];
  const llm = resources.llm;
  return (
    <div className="resource-grid">
      <div className="resource-cards">
        <div className="resource-card"><Clock3 /><span>Συνολικός χρόνος</span><strong>{(resources.elapsed_seconds / 60).toFixed(2)} min</strong></div>
        <div className="resource-card"><Cpu /><span>Runtime multiplier</span><strong>{resources.runtime_multiplier.toFixed(3)}×</strong></div>
        <div className="resource-card"><BrainCircuit /><span>LLM requests</span><strong>{llm.api_requests}</strong></div>
        <div className="resource-card"><MemoryStick /><span>Total tokens</span><strong>{llm.total_tokens.toLocaleString("el-GR")}</strong></div>
        <div className="resource-card"><Coins /><span>Observed API cost</span><strong>${llm.observed_api_cost_usd.toFixed(2)}</strong></div>
      </div>
      <ChartFrame title="Peak RSS increase" subtitle="Sampled process-tree memory κατά τα συγκρινόμενα runs." insight="Η Agentic υποψήφια εκτέλεση αύξησε περισσότερο το peak RSS. Οι μετρήσεις ξεκίνησαν από διαφορετικό αρχικό RSS, γι’ αυτό συγκρίνουμε την αύξηση και όχι το απόλυτο peak.">
        <ResponsiveContainer width="100%" height={290}>
          <BarChart data={memory} margin={{ top: 20, right: 15, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" /><YAxis unit=" MiB" />
            <Tooltip formatter={(value) => `${Number(value ?? 0).toFixed(3)} MiB`} />
            <Bar dataKey="value" radius={[7, 7, 0, 0]}>
              <Cell fill="#315f78" /><Cell fill="#dc7655" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>
      <article className="llm-card">
        <span className="eyebrow">LLM telemetry</span>
        <h3>{llm.model}</h3>
        <div className="token-bar"><span style={{ width: `${(llm.prompt_tokens / llm.total_tokens) * 100}%` }}>Prompt {llm.prompt_tokens}</span><span>Completion {llm.completion_tokens}</span></div>
        <dl><div><dt>Provider time</dt><dd>{llm.provider_total_seconds.toFixed(3)} sec</dd></div><div><dt>Retries</dt><dd>{llm.retries}</dd></div><div><dt>Tier</dt><dd>{llm.configured_billing_tier}</dd></div></dl>
      </article>
    </div>
  );
}
