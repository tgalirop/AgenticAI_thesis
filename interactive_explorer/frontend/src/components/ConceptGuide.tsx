import { useState } from "react";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import type { ExplorerPayload } from "../types";

interface ConceptGuideProps {
  glossary: ExplorerPayload["learning"]["glossary"];
  terms?: string[];
  title?: string;
}

/** Expandable plain-language definitions placed next to the relevant evidence. */
export function ConceptGuide({ glossary, terms, title = "Έννοιες που χρειάζεσαι εδώ" }: ConceptGuideProps) {
  const [open, setOpen] = useState(false);
  const visible = terms?.length ? glossary.filter((item) => terms.includes(item.term)) : glossary;
  return (
    <aside className="concept-guide">
      <button onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <BookOpen size={19} /><span>{title}</span>{open ? <ChevronUp /> : <ChevronDown />}
      </button>
      {open && <div className="concept-grid">{visible.map((item) => (
        <article key={item.term}><h4>{item.term}</h4><p>{item.plain}</p><small><strong>Γιατί μας ενδιαφέρει:</strong> {item.why}</small></article>
      ))}</div>}
    </aside>
  );
}
