import { AlertCircle, CheckCircle2, HelpCircle, Lightbulb } from "lucide-react";

interface ExplanationBlockProps {
  why: string;
  how: string;
  conclusion: string;
  limitation: string;
}

/** Four-part narration template the student can reuse during presentation. */
export function ExplanationBlock({ why, how, conclusion, limitation }: ExplanationBlockProps) {
  return <div className="explanation-grid">
    <article><HelpCircle /><div><strong>Γιατί το κάναμε;</strong><p>{why}</p></div></article>
    <article><Lightbulb /><div><strong>Πώς λειτουργεί;</strong><p>{how}</p></div></article>
    <article><CheckCircle2 /><div><strong>Τι επιτρέπεται να συμπεράνουμε;</strong><p>{conclusion}</p></div></article>
    <article><AlertCircle /><div><strong>Τι δεν αποδεικνύει;</strong><p>{limitation}</p></div></article>
  </div>;
}
