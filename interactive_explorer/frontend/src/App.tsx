import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, BarChart3, BookOpen, BrainCircuit, CheckCircle2, Database, FlaskConical, GitBranch, GitCompareArrows, Menu, Network, ShieldCheck, X } from "lucide-react";
import { fetchExplorer } from "./api";
import { ConfusionMatrix } from "./components/ConfusionMatrix";
import { ConceptGuide } from "./components/ConceptGuide";
import { CurveComparison } from "./components/CurveComparison";
import { DatasetJourney } from "./components/DatasetJourney";
import { ExplanationBlock } from "./components/ExplanationBlock";
import { LangGraphLab } from "./components/LangGraphLab";
import { MetricComparison } from "./components/MetricComparison";
import { ResourcesPanel } from "./components/ResourcesPanel";
import { StatisticsPanel } from "./components/StatisticsPanel";
import { WorkflowExplorer } from "./components/WorkflowExplorer";
import type { ExplorerPayload, IterationSummary, ModelId } from "./types";

type SectionId = "overview" | "workflow" | "data" | "langgraph" | "training" | "test" | "statistics" | "resources" | "plan";

const sections: { id: SectionId; label: string; icon: typeof BookOpen }[] = [
  { id: "overview", label: "Με μία ματιά", icon: BookOpen },
  { id: "workflow", label: "Pipeline βήμα-βήμα", icon: Network },
  { id: "data", label: "Δεδομένα & MCAR", icon: Database },
  { id: "langgraph", label: "LangGraph εργαστήριο", icon: GitBranch },
  { id: "training", label: "25 paired folds", icon: FlaskConical },
  { id: "test", label: "Final test dataset", icon: GitCompareArrows },
  { id: "statistics", label: "Στατιστική ανάλυση", icon: BarChart3 },
  { id: "resources", label: "Κόστος & πόροι", icon: BrainCircuit },
  { id: "plan", label: "Agent plan", icon: ShieldCheck },
];

function modelTabs(data: ExplorerPayload, model: ModelId, setModel: (model: ModelId) => void) {
  return <div className="model-tabs">{data.methodology.models.map((item) => <button key={item.id} className={model === item.id ? "active" : ""} onClick={() => setModel(item.id)}>{item.label}</button>)}</div>;
}

const EPSILON = 1e-6;
const greek = (value: number, digits: number) => value.toFixed(digits).replace(".", ",");

/** Per-model test PR-AUC deltas (Agentic − Conventional) derived from the snapshot. */
function testDeltas(data: ExplorerPayload) {
  return data.methodology.models.map((item) => {
    const row = (pipeline: string) => data.test_evaluation.metrics.find((metric) => metric.model === item.id && metric.pipeline === pipeline);
    return { label: item.label, delta: Number(row("agentic")?.pr_auc ?? 0) - Number(row("conventional")?.pr_auc ?? 0) };
  });
}

function testConclusion(data: ExplorerPayload) {
  const deltas = testDeltas(data);
  if (deltas.every((item) => Math.abs(item.delta) < EPSILON)) {
    return "Τα δύο pipelines έχουν ταυτόσημο test PR-AUC σε όλα τα μοντέλα. Δεν υπάρχει Agentic υπεροχή: το baseline χειριζόταν ήδη τα missing values.";
  }
  const better = deltas.filter((item) => item.delta >= EPSILON).map((item) => item.label);
  const worse = deltas.filter((item) => item.delta <= -EPSILON).map((item) => item.label);
  return `Θετικό PR-AUC delta: ${better.join(", ") || "κανένα μοντέλο"}· αρνητικό: ${worse.join(", ") || "κανένα μοντέλο"}. Δεν υπάρχει καθολική Agentic υπεροχή.`;
}

function statisticsConclusion(data: ExplorerPayload) {
  const tests = data.cross_validation.paired_tests;
  if (tests.every((test) => Math.abs(Number(test.mean_delta)) < EPSILON)) {
    return `Και οι ${tests.length} model/metric συγκρίσεις έχουν μηδενική διαφορά σε όλα τα 25 folds (μόνο ties)· Wilcoxon και Nadeau–Bengio δίνουν p = 1.`;
  }
  const wilcoxon = tests.filter((test) => test.significant_0_05).length;
  const nadeau = tests.filter((test) => test.nb_significant_0_05).length;
  return `Σημαντικές μετά Holm: ${wilcoxon}/${tests.length} με Wilcoxon και ${nadeau}/${tests.length} με Nadeau–Bengio.`;
}

function resourcesConclusion(data: ExplorerPayload) {
  const llm = data.resources.llm;
  return `Στο επίσημο run ο runtime multiplier του επιλεγμένου plan ήταν ${greek(Number(data.resources.runtime_multiplier), 3)}, με ${llm.api_requests} Groq κλήσεις και ${Number(llm.total_tokens).toLocaleString("el-GR")} tokens σε ${data.selected_plan.iterations} iterations.`;
}

function historyTable(data: ExplorerPayload) {
  const history = (data.selected_plan.history ?? []) as IterationSummary[];
  if (!history.length) return null;
  const signed = (value: number | null, digits = 6) => (value === null ? "—" : `${value >= 0 ? "+" : ""}${greek(value, digits)}`);
  return <>
    <div className="section-intro minor"><span className="eyebrow">{history.length} iterations</span><h2>Πώς έφτασε ο Agent στο τελικό plan</h2><p>Κάθε γραμμή είναι ένα iteration του επίσημου checkpoint. Η απόφαση προέρχεται από τη deterministic FeedbackPolicy, όχι από το LLM.</p></div>
    <div className="table-wrap"><table>
      <thead><tr><th>#</th><th>Plan</th><th>Ενέργειες</th><th>Δ PR-AUC</th><th>Δ quality</th><th>Runtime</th><th>Απόφαση</th></tr></thead>
      <tbody>{history.map((item) => <tr key={item.iteration}>
        <td>{item.iteration}</td><td><code>{item.plan_id}</code></td><td><code>{item.actions_summary}</code></td>
        <td className={(item.primary_metric_delta ?? 0) >= 0 ? "positive" : "negative"}>{signed(item.primary_metric_delta)}</td>
        <td>{signed(item.quality_delta)}</td>
        <td>{item.runtime_multiplier === null ? "—" : `${greek(item.runtime_multiplier, 3)}×`}</td>
        <td><span className={item.decision === "ACCEPT" ? "status significant" : "status"}>{item.decision}</span><br /><small>{item.reason_code}</small></td>
      </tr>)}</tbody>
    </table></div>
  </>;
}

function App() {
  const [data, setData] = useState<ExplorerPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [section, setSection] = useState<SectionId>("overview");
  const [model, setModel] = useState<ModelId>("random_forest");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    fetchExplorer().then(setData).catch((reason: Error) => setError(reason.message));
  }, []);

  const selectedModelLabel = useMemo(() => data?.methodology.models.find((item) => item.id === model)?.label ?? "", [data, model]);

  if (error) return <main className="state-screen"><AlertTriangle size={42} /><h1>Δεν φορτώθηκε το experiment snapshot</h1><p>{error}</p></main>;
  if (!data) return <main className="state-screen"><div className="loader" /><h1>Φόρτωση πραγματικών experiment artifacts…</h1></main>;

  const testRows = data.test_evaluation.metrics;
  const currentSection = sections.find((item) => item.id === section)!;
  return (
    <div className="app-shell">
      <aside className={menuOpen ? "sidebar open" : "sidebar"}>
        <div className="brand"><div className="brand-mark">AI</div><div><strong>Thesis Explorer</strong><span>Interactive evidence</span></div><button className="close-menu" onClick={() => setMenuOpen(false)}><X /></button></div>
        <nav>{sections.map((item) => { const Icon = item.icon; return <button key={item.id} className={section === item.id ? "active" : ""} onClick={() => { setSection(item.id); setMenuOpen(false); }}><Icon size={19} />{item.label}</button>; })}</nav>
        <div className="run-stamp"><span>Official run</span><code>{data.source_run_id}</code><small>Snapshot v{data.snapshot_version}</small></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><button className="menu-button" onClick={() => setMenuOpen(true)}><Menu /></button><div><span className="eyebrow">Agentic AI · PaySim fraud detection</span><h1>{currentSection.label}</h1></div><span className="evidence-badge"><CheckCircle2 size={17} /> Real experiment evidence</span></header>

        {section === "overview" && <section className="page-section">
          <div className="hero"><div><span className="eyebrow">Διαδραστική τεκμηρίωση διπλωματικής</span><h2>Από 6,3 εκατομμύρια συναλλαγές σε ένα ελεγχόμενο Agentic AI πείραμα.</h2><p>Ακολούθησε τη ροή, πάτησε κάθε στάδιο και διάβασε τα πραγματικά αποτελέσματα χωρίς να μπερδεύεις training cross-validation με final test evaluation.</p><button className="primary-button" onClick={() => setSection("workflow")}>Ξεκίνα από το pipeline</button></div><div className="hero-orbit"><span>Raw data</span><span>LangGraph</span><strong>PR-AUC</strong><span>25 folds</span><span>Test dataset</span></div></div>
          <div className="research-question"><span className="eyebrow">Το κεντρικό ερώτημα</span><p>{data.learning.central_question}</p></div>
          <div className="kpi-grid"><div><span>Raw rows</span><strong>{data.datasets.raw.rows.toLocaleString("el-GR")}</strong><small>8.213 fraud</small></div><div><span>Training sample</span><strong>{data.datasets.sample.rows.toLocaleString("el-GR")}</strong><small>10 normal : 1 fraud</small></div><div><span>MCAR missing</span><strong>{data.quality.degradation.affected_rows.toLocaleString("el-GR")}</strong><small>5% της type</small></div><div><span>Paired comparisons</span><strong>3 × 25</strong><small>μοντέλα × κοινά folds</small></div><div><span>Test rows</span><strong>{data.datasets.test_dataset.rows.toLocaleString("el-GR")}</strong><small>steps 595–743</small></div></div>
          <div className="two-question-grid">{data.learning.experiment_layers.map((layer, index) => <article key={layer.title}><span className="number">0{index + 1}</span><h3>{layer.title}</h3><p><strong>Ερώτηση:</strong> {layer.question}</p><p><strong>Απόδειξη:</strong> {layer.evidence}</p><p><strong>Επιτρεπτό συμπέρασμα:</strong> {layer.allowed_conclusion}</p></article>)}</div>
          <div className="warning-banner"><AlertTriangle /><div><strong>Δεν συγκρίνουμε το sample με το test dataset.</strong><span>Σε κάθε στάδιο συγκρίνουμε τα δύο pipelines πάνω στα ίδια δεδομένα. Τα δύο στάδια απαντούν διαφορετικά ερευνητικά ερωτήματα.</span></div></div>
        </section>}

        {section === "workflow" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Κάνε click σε κάθε node</span><h2>Ολόκληρο το pipeline, βήμα προς βήμα</h2><p>Κάθε node δείχνει input, output, αιτιολόγηση και το πραγματικό αρχείο που το υλοποιεί.</p></div><WorkflowExplorer steps={data.workflow} /></section>}
        {section === "data" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Data journey</span><h2>Πώς δημιουργούνται τα διαφορετικά σύνολα</h2><p>Το sample και το test dataset είναι διαφορετικοί κλάδοι του training/chronological split.</p></div><ExplanationBlock why="Χρειαζόμαστε οικονομικά εφικτή ανάπτυξη χωρίς να αγγίξουμε την τελική χρονική αξιολόγηση." how="Χωρίζουμε πρώτα όλο το dataset χρονικά. Μόνο από το training δημιουργούμε το 10:1 sample και μόνο σε deep copy αυτού εισάγουμε MCAR missingness." conclusion="Μπορούμε να συγκρίνουμε δίκαια τα δύο pipelines στο controlled sample και αργότερα να ελέγξουμε out-of-time γενίκευση." limitation="Το sample δεν έχει τη φυσική fraud prevalence και το καθαρό test dataset δεν δοκιμάζει missing values κατά το inference." /><ConceptGuide glossary={data.learning.glossary} terms={["Test dataset", "MCAR", "Data leakage"]} /><DatasetJourney data={data.datasets} quality={data.quality} /></section>}

        {section === "langgraph" && <section className="page-section">
          <div className="section-intro"><span className="eyebrow">Η πραγματική Agentic μηχανή</span><h2>Πώς και γιατί χρησιμοποιήσαμε το LangGraph</h2><p>{data.langgraph.graph_role}</p></div>
          <ExplanationBlock why="Η διαδικασία δεν είναι μία απλή LLM απάντηση. Χρειάζεται ασφαλής κύκλος proposal → validation → execution → measurement → feedback, με κλάδους για invalid plan, retry και terminal stop." how="Το Python LangGraph StateGraph συνδέει επτά μικρά callable node objects μέσω σταθερών και conditional edges. Κάθε node διαβάζει/γράφει συγκεκριμένα typed state fields και οι domain services εισάγονται ως dependencies." conclusion="Η ροή είναι επεκτάσιμη, audit-ready και bounded: μπορούμε να προσθέσουμε node ή transformation χωρίς να μετατρέψουμε το σύστημα σε monolithic script." limitation="Το LangGraph δεν κάνει από μόνο του preprocessing ή ML. Ενορχηστρώνει τις κλάσεις που τα κάνουν. Επίσης δεν κάνει το LLM αυτόματα ασφαλές· γι’ αυτό υπάρχουν Validator, allowlist και deterministic Executor." />
          <div className="simulation-notice"><GitBranch /><div><strong>Τι είναι διαδραστικό εδώ;</strong><span>Το παρακάτω εργαστήριο αναπαράγει βήμα-βήμα το επίσημο checkpoint και έναν τεκμηριωμένο invalid/RETRY κλάδο. Δεν ξανακαλεί το Groq ούτε ξανατρέχει 150 model fits όταν πατάς τα κουμπιά.</span></div></div>
          <LangGraphLab graph={data.langgraph} />
          <div className="section-intro minor"><span className="eyebrow">Architectural reasoning</span><h2>Γιατί το σχεδιάσαμε έτσι</h2><p>Κάθε επιλογή αντιστοιχεί σε συγκεκριμένο πρόβλημα ασφάλειας, επεκτασιμότητας ή αναπαραγωγιμότητας.</p></div>
          <div className="decision-grid">{data.langgraph.design_decisions.map((decision, index) => <article key={decision.choice}><span>0{index + 1}</span><h3>{decision.choice}</h3><p>{decision.reason}</p></article>)}</div>
          <div className="section-intro minor"><span className="eyebrow">Typed state contract</span><h2>Τι ταξιδεύει μεταξύ των nodes;</h2><p>Το state λειτουργεί σαν ελεγχόμενος κοινός φάκελος. Δεν περνάμε ολόκληρα dataframes ή fitted μοντέλα από node σε node.</p></div>
          <div className="state-table">{data.langgraph.state_fields.map((field) => <article key={field.name}><header><code>{field.name}</code><span>{field.lifetime}</span></header><p><strong>Περιέχει:</strong> {field.contains}</p><p><strong>Σκόπιμα δεν περιέχει:</strong> {field.excludes}</p></article>)}</div>
          <div className="oral-script"><strong>Πώς το εξηγείς προφορικά:</strong><p>«Χρησιμοποιώ το LangGraph ως stateful orchestrator. Το LLM είναι μόνο ένα node που προτείνει typed plan. Η πρόταση περνά από deterministic Validator, εκτελείται μόνο μέσω allowlisted factories, αξιολογείται με κοινό ML protocol και η FeedbackPolicy αποφασίζει ACCEPT, RETRY ή STOP. Κάθε iteration αποθηκεύεται σε immutable checkpoint, άρα η διαδικασία είναι ελέγξιμη και επεκτάσιμη.»</p></div>
        </section>}

        {section === "training" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Training evidence</span><h2>5 folds × 5 repetitions = 25 κοινά ζεύγη</h2><p>Οι καμπύλες προέρχονται από πραγματικές out-of-fold προβλέψεις στο degraded 10:1 sample. {data.cross_validation.curve_note}</p></div><ExplanationBlock why="Ένας μόνο train/validation χωρισμός μπορεί τυχαία να ευνοήσει ένα pipeline." how="Κάθε pipeline κάνει fit στα ίδια training indices και αξιολογείται στα ίδια validation indices, 5 folds επί 5 νέους χωρισμούς." conclusion="Η fold-by-fold διαφορά δείχνει αν η Agentic επίδραση επαναλαμβάνεται με σχετική συνέπεια μέσα στο controlled sample." limitation="Τα 25 folds δεν είναι ανεξάρτητα datasets και οι επιδόσεις τους δεν προβλέπουν αυτομάτως τη φυσική test επίδοση." /><ConceptGuide glossary={data.learning.glossary} terms={["Fold", "Repetition", "Paired comparison", "PR curve / PR-AUC", "ROC curve / ROC-AUC", "PR-AUC baseline"]} />{modelTabs(data, model, setModel)}<div className="chart-grid"><CurveComparison curves={data.cross_validation.oof_curves} model={model} kind="pr" scopeLabel={`${selectedModelLabel} · OOF training`} /><CurveComparison curves={data.cross_validation.oof_curves} model={model} kind="roc" scopeLabel={`${selectedModelLabel} · OOF training`} /></div><div className="warning-banner compact"><AlertTriangle /><div><strong>Πρόσεξε την prevalence.</strong><span>Οι OOF καμπύλες αφορούν sample με 9,09% fraud. Δεν συγκρίνουμε το ύψος τους απευθείας με τις test καμπύλες.</span></div></div></section>}

        {section === "test" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Untouched out-of-time test dataset</span><h2>Τελική αξιολόγηση μετά το freeze</h2><p>{data.test_evaluation.interpretation}</p></div><ExplanationBlock why="Το cross-validation στο enriched sample δεν αρκεί για να δείξει συμπεριφορά σε μελλοντικές συναλλαγές φυσικής κατανομής." how="Μετά το freeze, και τα δύο pipelines κάνουν fit στο ίδιο degraded sample και predict στις ίδιες καθαρές test γραμμές χωρίς νέο tuning." conclusion={testConclusion(data)} limitation="Ένα test window δεν δίνει confidence interval και το καθαρό test dataset δεν ελέγχει Agentic αποκατάσταση missing values κατά το inference." /><ConceptGuide glossary={data.learning.glossary} terms={["Test dataset", "Precision", "Recall", "F1-score", "PR curve / PR-AUC", "ROC curve / ROC-AUC", "Confusion matrix"]} />{modelTabs(data, model, setModel)}<div className="chart-grid"><CurveComparison curves={data.test_evaluation.curves} model={model} kind="pr" scopeLabel={`${selectedModelLabel} · test`} /><CurveComparison curves={data.test_evaluation.curves} model={model} kind="roc" scopeLabel={`${selectedModelLabel} · test`} /><MetricComparison rows={testRows} metric="pr_auc" title="Final PR-AUC comparison" insight={testConclusion(data)} /><MetricComparison rows={testRows} metric="f1" title="Final F1 comparison" insight="Το F1 εξαρτάται από threshold 0,5 και ισορροπεί Precision/Recall. Δεν αντικαθιστά την threshold-independent PR-AUC." /></div><div className="section-intro minor"><span className="eyebrow">Threshold = 0,5</span><h2>Confusion matrices — {selectedModelLabel}</h2><p>Διάβασε ξεχωριστά τα false positives και false negatives: σε fraud detection έχουν διαφορετικό επιχειρησιακό κόστος.</p></div><div className="matrix-pair">{data.test_evaluation.confusion_matrices.filter((item) => item.model === model).map((matrix) => <ConfusionMatrix key={`${matrix.pipeline}-${matrix.model}`} matrix={matrix} />)}</div><div className="oral-script"><strong>Πώς το εξηγείς προφορικά:</strong><p>«Η PR-AUC αξιολογεί το ranking σε όλα τα thresholds. Η confusion matrix αντίθετα παγώνει το threshold στο 0,5 και μου δείχνει συγκεκριμένα πόσες απάτες βρήκα, πόσες έχασα και πόσους ψευδείς συναγερμούς δημιούργησα. Γι’ αυτό χρειάζομαι και τις δύο οπτικές.»</p></div></section>}

        {section === "statistics" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Wilcoxon + Nadeau–Bengio · Holm correction</span><h2>Τι συνέβη στα 25 κοινά folds;</h2><p>Οι έλεγχοι γίνονται μόνο στο training experiment. Το test dataset έχει μία μέτρηση και δεν συμμετέχει στα p-values.</p></div><ExplanationBlock why="Οι δύο μέσοι όροι μόνοι τους δεν δείχνουν αν η διαφορά επαναλαμβάνεται ή προέκυψε από λίγα τυχαία folds." how="Υπολογίζουμε Agentic − Conventional σε κάθε κοινό fold, εφαρμόζουμε paired Wilcoxon και Nadeau–Bengio corrected t-test (διορθώνει για την επικάλυψη των training sets) και διορθώνουμε τα 12 p-values κάθε test με Holm." conclusion={statisticsConclusion(data)} limitation="Το p-value δεν μετρά το μέγεθος ή την επιχειρησιακή αξία και δεν σημαίνει ότι το Agentic είναι καλύτερο παντού." /><ConceptGuide glossary={data.learning.glossary} terms={["Paired comparison", "p-value", "Holm correction", "Nadeau–Bengio test"]} /><StatisticsPanel tests={data.cross_validation.paired_tests} /></section>}
        {section === "resources" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Operational evidence</span><h2>Χρόνος, μνήμη και LLM usage</h2><p>Η predictive απόδοση διαβάζεται μαζί με το πρόσθετο κόστος της Agentic διαδικασίας.</p></div><ExplanationBlock why="Μία μικρή predictive βελτίωση μπορεί να μην αξίζει αν απαιτεί δυσανάλογο χρόνο, μνήμη ή πληρωμένες LLM κλήσεις." how="Μετράμε wall-clock runtime, sampled process-tree RSS και provider-reported requests, tokens και latency." conclusion={resourcesConclusion(data)} limitation="Ένα run δεν αποτελεί benchmark υποδομής και το observed $0 ισχύει για το συγκεκριμένο Free Tier, όχι για κάθε deployment." /><ResourcesPanel resources={data.resources} /></section>}

        {section === "plan" && <section className="page-section"><div className="section-intro"><span className="eyebrow">Frozen plan · {data.selected_plan.termination_action}</span><h2>Τι πρότεινε και τι εκτέλεσε ο Agent</h2><p>Το LLM δημιούργησε declarative JSON σε κάθε iteration. Ο Validator το έλεγξε, ο deterministic Executor το εφάρμοσε πάνω στο baseline preprocessing και η FeedbackPolicy αποφάσισε.</p></div><div className="agent-chain"><div><BrainCircuit /><strong>LLM proposal</strong><span>{data.resources.llm.api_requests} requests · χωρίς raw rows</span></div><div className="chain-arrow">→</div><div><ShieldCheck /><strong>Validator</strong><span>{data.selected_plan.validation.is_valid ? "Valid plan" : "Rejected"}</span></div><div className="chain-arrow">→</div><div><CheckCircle2 /><strong>Executor</strong><span>Safe sklearn objects</span></div><div className="chain-arrow">→</div><div><GitCompareArrows /><strong>Feedback</strong><span>{data.selected_plan.feedback.action}</span></div></div>{historyTable(data)}<div className="section-intro minor"><span className="eyebrow">Selected plan · {data.selected_plan.plan.plan_id}</span><h2>Οι ενέργειες του plan που κλείδωσε</h2></div><div className="action-grid">{data.selected_plan.plan.actions.map((action: any, index: number) => <article key={action.action_id}><span className="action-number">0{index + 1}</span><h3>{action.transformation}</h3><code>{action.columns.join(", ")}</code><p>{action.rationale}</p><dl><div><dt>Parameters</dt><dd>{JSON.stringify(action.parameters)}</dd></div><div><dt>Scope</dt><dd>{action.model_scope}</dd></div></dl></article>)}</div><div className="quality-strip"><div><span>Quality before</span><strong>{data.quality.before_score.toFixed(6)}</strong></div><div className="quality-arrow">→</div><div><span>Quality after</span><strong>{data.quality.after_score.toFixed(6)}</strong></div><div><span>Delta</span><strong className="positive">+{data.quality.delta.toFixed(6)}</strong></div></div><p className="callout">{data.quality.interpretation}</p><div className="caveats"><h3>Όρια που πρέπει να αναφέρουμε</h3>{data.caveats.map((caveat) => <div key={caveat}><AlertTriangle size={17} /><span>{caveat}</span></div>)}</div></section>}
      </main>
    </div>
  );
}

export default App;
