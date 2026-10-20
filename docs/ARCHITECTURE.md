# Αρχιτεκτονικές Αρχές του Agentic AI Συστήματος

Το παρόν αρχείο καθορίζει τις δεσμευτικές αρχές σχεδιασμού για το LangGraph και
γενικότερα για ολόκληρο το Agentic AI μέρος της διπλωματικής. Στόχος δεν είναι
μόνο να λειτουργήσει το τελικό πείραμα, αλλά να προκύψει κώδικας υψηλής ποιότητας,
εύκολα ελέγξιμος, συντηρήσιμος και επεκτάσιμος.

## Βασική απαίτηση

Το Agentic AI υποσύστημα θα υλοποιηθεί με αντικειμενοστρεφή σχεδιασμό. Κάθε βασική
ευθύνη θα αντιστοιχεί σε σαφώς ορισμένη κλάση ή interface και δεν θα συγκεντρωθεί
σε ένα μεγάλο script ή σε ένα σύνολο στενά συνδεδεμένων functions.

Οι LangGraph nodes θα αποτελούν λεπτά orchestration adapters. Η επιχειρησιακή
λογική, το validation, η εκτέλεση transformations, η αξιολόγηση και η διαχείριση
του state θα υλοποιούνται σε ανεξάρτητες κλάσεις που μπορούν να χρησιμοποιηθούν
και να ελεγχθούν χωρίς να απαιτείται εκτέλεση ολόκληρου του graph.

## Αρχές σχεδιασμού

### Single Responsibility

Κάθε κλάση θα έχει μία σαφή ευθύνη. Ενδεικτικά:

- ο Profiler παράγει Data Quality Reports,
- ο Strategy Generator δημιουργεί transformation plans,
- ο Validator ελέγχει plans,
- ο Executor εφαρμόζει μόνο εγκεκριμένες ενέργειες,
- οι Evaluators υπολογίζουν ποιότητα και ML performance,
- ο Feedback Policy αποφασίζει `ACCEPT`, `RETRY` ή `STOP`,
- ο Graph Builder συνδέει τους κόμβους χωρίς να υλοποιεί την εσωτερική λογική τους.

### Dependency Injection

Οι εξαρτήσεις θα περνούν στους constructors και δεν θα δημιουργούνται κρυφά μέσα
στις κλάσεις. Για παράδειγμα, ο Strategy Generator θα λαμβάνει model client,
prompt provider και plan parser ως dependencies. Έτσι θα μπορούν να
αντικαθίστανται εύκολα σε tests ή σε μελλοντικές επεκτάσεις.

### Interfaces και Protocols

Όπου υπάρχουν εναλλακτικές υλοποιήσεις, θα ορίζονται typed interfaces ή Python
`Protocol` classes. Ενδεικτικά:

- `StrategyGeneratorProtocol`,
- `PlanValidatorProtocol`,
- `TransformationProtocol`,
- `EvaluatorProtocol`,
- `FeedbackPolicyProtocol`,
- `CheckpointStoreProtocol`.

Το graph θα εξαρτάται από τα interfaces και όχι από μία συγκεκριμένη υλοποίηση.

### Composition over inheritance

Θα προτιμάται η σύνθεση μικρών συνεργαζόμενων αντικειμένων έναντι βαθιών
ιεραρχιών κληρονομικότητας. Η κληρονομικότητα θα χρησιμοποιείται μόνο όταν
υπάρχει πραγματική και σταθερή σχέση εξειδίκευσης.

### Typed domain models

Τα δεδομένα που ανταλλάσσονται μεταξύ των components δεν θα είναι αυθαίρετα
dictionaries. Θα οριστούν typed domain models, ενδεικτικά:

- `DataQualityReport`,
- `TransformationAction`,
- `TransformationPlan`,
- `ValidationResult`,
- `EvaluationResult`,
- `IterationRecord`,
- `AgentState`,
- `TerminationDecision`.

Για structured LLM output θα χρησιμοποιούνται αυστηρά validated schemas. Invalid
ή ελλιπές output δεν θα φτάνει ποτέ στον Executor.

### Transformation registry

Οι επιτρεπόμενοι μετασχηματισμοί θα καταχωρίζονται σε registry. Κάθε transformation
θα αποτελεί ανεξάρτητη κλάση με κοινό interface, όπως:

```python
class TransformationProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def validate(self, action: TransformationAction, context: DatasetContext) -> None: ...

    def build(self, action: TransformationAction) -> object: ...
```

Η προσθήκη νέου transformation δεν θα απαιτεί αλλαγή στον πυρήνα του Executor ή
στο LangGraph graph. Θα απαιτεί μόνο νέα υλοποίηση και εγγραφή στο registry.

### Configuration-driven behavior

Iterations, stopping criteria, thresholds, model settings, επιτρεπόμενες ενέργειες
και budgets θα προέρχονται από validated configuration objects. Δεν θα υπάρχουν
διάσπαρτες hard-coded τιμές στον Agentic κώδικα.

### Testability

Κάθε κλάση θα μπορεί να ελεγχθεί αυτόνομα. Τα tests θα περιλαμβάνουν:

- unit tests ανά component,
- contract tests για κάθε interface,
- tests απόρριψης unsafe ή leakage-prone plans,
- integration tests του graph με fake LLM,
- deterministic replay tests αποθηκευμένων plans,
- end-to-end test χωρίς πραγματικές εξωτερικές κλήσεις.

Οι βασικοί έλεγχοι δεν θα εξαρτώνται από διαθέσιμο API key ή από live LLM.

### Observability και reproducibility

Κάθε iteration θα καταγράφει:

- input report και plan,
- validation αποτέλεσμα,
- εφαρμοσμένες ενέργειες,
- data-quality και ML metrics,
- χρόνους και χρήση πόρων,
- warnings και errors,
- feedback decision,
- random seeds και package/model versions.

Τα logs θα είναι structured και θα επιτρέπουν αναπαραγωγή και audit της ροής.

## Προτεινόμενα βασικά αντικείμενα

```text
AgentApplication
├── AgentGraphBuilder
├── AgentStateManager
├── DataProfiler
├── StrategyGenerator
│   ├── ModelClient
│   ├── PromptProvider
│   └── PlanParser
├── TransformationPlanValidator
│   └── TransformationRegistry
├── TransformationExecutor
│   └── TransformationRegistry
├── DataQualityEvaluator
├── MachineLearningEvaluator
├── FeedbackPolicy
├── ExperimentLogger
└── CheckpointStore
```

## Κανόνες ασφαλείας

- Το LLM επιστρέφει μόνο structured plans.
- Δεν παράγεται ή εκτελείται αυθαίρετος Python κώδικας.
- Ο Executor γνωρίζει μόνο transformations που υπάρχουν στη whitelist/registry.
- Κάθε plan περνά υποχρεωτικά από Validator.
- Oversampling εφαρμόζεται μόνο στο training μέρος κάθε fold.
- Το test dataset δεν είναι διαθέσιμο στους Agent components.
- Κάθε iteration ξεκινά από το ίδιο αρχικό training dataset.
- Ο `TransformationExecutor` δέχεται μόνο `ValidationResult(is_valid=True)`.
- Κάθε executable action επιλύεται μέσω injected `TransformationFactoryRegistry`.
- Validated παράμετρος δεν επιτρέπεται να αγνοείται από την execution factory.
- Επιτρέπεται το πολύ ένας sampler ανά model pipeline.
- Οι samplers ενσωματώνονται σε `imblearn.Pipeline` και ενεργοποιούνται μόνο στο
  `fit`, ποτέ στο validation ή στο prediction.

## Deterministic execution layer

Η εκτέλεση διαχωρίζεται σε τρία επίπεδα:

```text
Validated TransformationPlan
          ↓
TransformationFactoryRegistry
          ↓
ModelPipelineBuilder
          ↓
TransformationExecutor → ExecutionResult
```

Κάθε transformation factory παράγει ένα typed execution artifact:

- column transformer,
- fold-local sampler,
- ασφαλείς estimator parameters.

Ο `ModelPipelineBuilder` εφαρμόζει μόνο actions που αντιστοιχούν στο τρέχον model
scope, διατηρεί τη σειρά τους ανά στήλη και κλωνοποιεί τα sklearn components ώστε
να μην υπάρχει κοινό fitted state. Ο Executor εφαρμόζει ανεξάρτητο defence-in-depth
έλεγχο και αρνείται execution σε validation-fold ή test dataset context, ακόμη
και αν του δοθεί χειροκίνητα ένα φαινομενικά έγκυρο result.

### Το plan εφαρμόζεται πάνω στο conventional baseline

Ένα Agent plan **τροποποιεί** το conventional preprocessing· δεν το αντικαθιστά.
Ο `ModelPipelineBuilder` δέχεται injected `BaselinePreprocessingContract`, που
περιγράφει δηλωτικά τι κάνει το conventional pipeline:

- one-hot encoding (`handle_unknown="ignore"`) στη `type` για όλα τα μοντέλα·
- standard scaling στα numeric features μόνο για τη Logistic Regression.

Για κάθε στήλη, τα βήματα του plan μπαίνουν πρώτα. Μετά προστίθενται τα βήματα του
baseline **των οποίων το είδος δεν ζήτησε ήδη το plan** για τη συγκεκριμένη στήλη.
Παραδείγματα:

| Plan για τη στήλη | Τελικό pipeline της στήλης |
|---|---|
| τίποτα | ακριβώς το baseline |
| `impute_categorical` στη `type` | imputation → baseline one-hot |
| `one_hot_encode` στη `type` | μόνο το one-hot του plan (το baseline one-hot παραλείπεται) |
| `log_transform` στο `amount` (LR) | log → baseline standard scaling |
| `scale_numeric` (LR) | μόνο το scaling του plan (το baseline scaling παραλείπεται) |

Οι στήλες τοποθετούνται με τη σειρά του conventional transformer, ώστε και τα δύο
pipelines να βλέπουν την ίδια διάταξη χαρακτηριστικών. Έτσι κάθε διαφορά μεταξύ
conventional και Agentic οφείλεται μόνο στις ενέργειες του plan. Πριν από τις
17/10/2026 ο builder έχτιζε το pipeline μόνο από το plan, οπότε ένα plan χωρίς
`scale_numeric` αφαιρούσε σιωπηρά το scaling της Logistic Regression.

## Shared cross-validation και ML evaluation

Η conventional και η Agentic αξιολόγηση χρησιμοποιούν την ίδια κλάση
`CrossValidationFoldProvider`. Ο provider δημιουργεί μία φορά immutable
`CrossValidationFoldSet` και συνδέει τα indices με SHA-256 fingerprint της ακριβούς
target σειράς. Έτσι, ένα σύνολο folds δεν μπορεί να χρησιμοποιηθεί κατά λάθος σε
διαφορετικό sample ή σε διαφορετική σειρά εγγραφών.

Ο `MachineLearningEvaluator`:

- λαμβάνει injected fold set και configuration,
- κλωνοποιεί το pipeline πριν από κάθε fit,
- δεν μεταβάλλει το αρχικό reusable pipeline,
- παράγει typed fold results και metric aggregates,
- κρατά τις αναλυτικές OOF predictions εκτός του compact Agent state,
- καταγράφει fold errors ως typed feedback,
- διαχωρίζει `success`, `partial_failure` και `error`.

Με αυτή τη διάταξη, conventional και Agentic pipelines αξιολογούνται με κοινή
υποδομή folds και metrics. Η μόνη πειραματική διαφορά παραμένει η στρατηγική
preprocessing.

## Feedback, state και persistence

Το `FeedbackPolicy` είναι deterministic service. Το LLM δεν αποφασίζει αν μία
στρατηγική πέτυχε και δεν ελέγχει τον τερματισμό του loop. Η policy συγκρίνει
per-model primary metric, Recall και Precision με το conventional baseline και
συνυπολογίζει Data Quality Score, χρόνο εκτέλεσης, validation/execution failures
και ιστορικό προηγούμενων προσπαθειών.

Οι thresholds παρέχονται μέσω immutable `FeedbackPolicyConfig`. Οι δυνατές
αποφάσεις είναι:

```text
ACCEPT
RETRY
STOP_NO_IMPROVEMENT
STOP_MAX_ITERATIONS
STOP_INVALID_STRATEGIES
STOP_EXECUTION_ERROR
```

Η policy εφαρμόζει τους ελέγχους με σταθερή σειρά και σταματά στον πρώτο που
αποτυγχάνει. Οι τιμές είναι αυτές του `configs/agent.yaml`:

| Σειρά | Έλεγχος | Όριο | Αν αποτύχει |
|---|---|---|---|
| 1 | Valid plan (Validator) | — | `RETRY`· `STOP_INVALID_STRATEGIES` μετά από 2 invalid |
| 2 | Επιτυχής execution | — | `RETRY`· `STOP_EXECUTION_ERROR` μετά από 2 errors ή fatal error |
| 3 | Όλα τα μοντέλα αξιολογήθηκαν επιτυχώς | `require_all_models_successful` | `RETRY` |
| 4 | Πτώση Recall ή Precision ανά μοντέλο έναντι baseline | ≤ 0,02 | `RETRY` (`metric_guardrail_violation`) |
| 5 | Runtime multiplier | ≤ 3,0 | `RETRY` (`excessive_runtime_cost`) |
| 6 | Όφελος: Δ mean PR-AUC ≥ 0,005 **ή** (Δ quality ≥ 0,0001 **και** Δ PR-AUC ≥ 0) | — | `ACCEPT` αν ισχύει, αλλιώς `RETRY` (`insufficient_improvement`) |

Κάθε `RETRY` μετατρέπεται σε `STOP_MAX_ITERATIONS` όταν εξαντληθούν τα 3
iterations, ή σε `STOP_NO_IMPROVEMENT` όταν 3 διαδοχικές προσπάθειες δεν βελτιώνουν
την PR-AUC (`no_improvement_patience`).

Ο κανόνας 6 σημαίνει ότι ένα plan μπορεί να γίνει `ACCEPT` μόνο λόγω βελτίωσης
ποιότητας, με αμετάβλητη PR-AUC. Αυτό συνέβη στο run της 20/10/2026 (plan-3).

Το `AgentState` είναι immutable και πλήρως serializable. Δεν περιέχει fitted
pipelines, model clients ή άλλες μη αναπαραγώγιμες Python αναφορές. Αποθηκεύει
typed summaries, plans, validation, feedback και artifact references. Μόνο ο
`AgentStateManager` επιτρέπεται να δημιουργεί νέα state transitions και ελέγχει
iteration sequence, dataset identity, plan/feedback coherence και terminal state.

Η persistence εξαρτάται από το `CheckpointStoreProtocol`. Η αρχική υλοποίηση
`JsonCheckpointStore` γράφει atomic UTF-8 JSON checkpoints, επικυρώνει ξανά το
state κατά το load και προστατεύει από path traversal. Μελλοντικός LangGraph
checkpointer μπορεί να αντικαταστήσει το store χωρίς αλλαγή της domain λογικής.

## Strategy Generator και τοπικό LLM

Το LLM integration ακολουθεί dependency inversion. Ο `StrategyGenerator` γνωρίζει
μόνο το `ModelClientProtocol` και συνεπώς δεν εξαρτάται από Groq, Ollama, LangChain ή
συγκεκριμένο model provider. Ο `GroqModelClient` υλοποιεί το κύριο interface για το
`openai/gpt-oss-20b` στο Free Tier, ο `OllamaModelClient` παραμένει προαιρετικό local
fallback και ο `FakeModelClient` παρέχει deterministic responses
στα tests χωρίς server, δίκτυο ή inference κόστος.

```text
StrategyPromptContext
        ↓
StrategyPromptProvider
        ↓
ModelClientProtocol → GroqModelClient → openai/gpt-oss-20b
                    ↘ OllamaModelClient (optional fallback)
        ↓
TransformationPlanParser
        ↓
TransformationPlan → Validator → Executor
```

Το prompt περιλαμβάνει μόνο compact metadata και metrics. Δεν περιλαμβάνει raw
PaySim rows και δεν επιτρέπεται πρόσβαση στο test dataset. Συγκεκριμένα, το
`StrategyPromptContext` περιέχει:

- dataset schema (στήλες, τύποι, protected target)·
- quality metrics και missingness ανά στήλη·
- baseline metrics ανά μοντέλο και το baseline pipeline contract, μαζί με τη ρητή
  δήλωση ότι το plan εφαρμόζεται πάνω στο baseline·
- allowlisted transformations και τα parameter contracts τους·
- `previous_iterations`: για κάθε προηγούμενο iteration, το plan, τα validation
  issues, το execution status/errors, την απόφαση με `reason_code`, τα mean
  primary/quality deltas, τον runtime multiplier και τα per-model deltas PR-AUC,
  Recall και Precision έναντι του baseline.

Τα `previous_iterations` παράγονται από τον `IterationHistorySummarizer` στο
`graph.py`. Το system prompt ζητά ρητά να μην ξαναυποβληθεί plan που έχει ήδη
αξιολογηθεί και να διορθωθεί κάθε validation issue. Πριν από τις 17/10/2026 το
LLM λάμβανε σε retry μόνο τα γενικά `recommendations` της policy. Το endpoint λαμβάνει
το JSON Schema του `TransformationPlan`, αλλά η απάντηση θεωρείται πάντοτε μη
έμπιστη: περνά ξανά από strict Pydantic validation (`extra="forbid"`) και από
ελέγχους dataset/iteration identity. Το LLM προτείνει μόνο allowlisted declarative
actions· δεν παράγει ούτε εκτελεί Python ή shell code.

## LangGraph orchestration

Το `AgentWorkflow` είναι το public facade του graph. Το `AgentGraphBuilder` συναρμολογεί
αντικαταστάσιμα class-based nodes και το `AgentGraphDependencies` συγκεντρώνει όλες
τις injected υπηρεσίες. Κανένα node δεν κατασκευάζει μόνο του model client,
validator, evaluator, policy ή checkpoint store.

Nodes:

1. `PrepareIterationNode`: ελέγχει lifecycle, dataset identity, iteration budget και
   επιτρέπει αποκλειστικά training context.
2. `GenerateStrategyNode`: δημιουργεί compact prompt context και καλεί τον
   provider-neutral Strategy Generator.
3. `ValidateStrategyNode`: εφαρμόζει allowlist, schema, parameter και leakage rules.
4. `InvalidStrategyAssessmentNode`: μετατρέπει invalid plan σε typed assessment,
   χωρίς execution.
5. `EvaluateCandidateNode`: καλεί το injected `CandidateEvaluatorProtocol` και
   μετατρέπει μόνο αναμενόμενα execution failures σε typed feedback.
6. `DecideFeedbackNode`: εφαρμόζει αποκλειστικά deterministic `FeedbackPolicy`.
7. `RecordIterationNode`: δημιουργεί immutable audit record και atomic checkpoint.

Το routing βασίζεται μόνο σε typed validation/status values. Δεν ζητείται από το LLM
να αποφασίσει routing ή τερματισμό. Το recursion limit παράγεται από το domain
`max_iterations`, προσφέροντας πρόσθετη προστασία από ακούσιο infinite loop.

## Candidate evaluation boundary

Ο `AgentCandidateEvaluator` κρατά εκτός graph state:

- training feature matrix και target,
- immutable shared fold set,
- estimators και fitted pipeline clones,
- detailed fold predictions.

Για κάθε valid plan καλεί τον `TransformationExecutor`, τον
`MachineLearningEvaluator` και ένα injected `PlanQualityEvaluatorProtocol`, και
επιστρέφει μόνο `CandidateAssessment`. Η concrete quality υλοποίηση είναι ο
`TransformationAwarePlanQualityEvaluator`, οπότε το Data Quality Score βασίζεται σε
μετρημένη post-transformation ποιότητα και όχι σε εκτίμηση του LLM. Στο sampled
πείραμα το report μετρά μόνο completeness και duplicates· τα validity και
consistency checks είναι κενά και δίνουν πάντα 1.

## Controlled degradation και experiment lifecycle

Το `ControlledDataDegrader` εφαρμόζει deterministic MCAR missingness μόνο σε deep
copy του reproducible training sample. Target, source Parquet και test
dataset προστατεύονται. Τα `is_transfer` και `is_cash_out` υπολογίζονται από τη
`type` πριν από την αλλοίωση, γι' αυτό δεν είναι model features: αλλιώς θα
αποκάλυπταν στο μοντέλο την τιμή της `type` που σβήστηκε. Ο `TransformationAwarePlanQualityEvaluator` εφαρμόζει σε
αντίγραφο μόνο quality-relevant actions (numeric/categorical imputation) και
επανυπολογίζει τον Data Quality Score· scaling, encoding και sampling δεν
βαφτίζονται λανθασμένα ως quality improvements.

Ο `AgenticExperimentRunner` επιβάλλει την ακολουθία:

```text
shared sample → controlled degradation → conventional shared-fold baseline
→ LangGraph search → locked accepted plan → Agentic shared-fold benchmark
→ paired Wilcoxon + Nadeau–Bengio tests (Holm) → one-time test dataset evaluation
```

Το Wilcoxon αντιμετωπίζει τις 25 fold διαφορές ως ανεξάρτητες. Τα training sets του
repeated k-fold όμως επικαλύπτονται, οπότε αναφέρεται δίπλα και το corrected
resampled t-test των Nadeau–Bengio. Αυτό αυξάνει τη διασπορά κατά τον λόγο
`n_validation / n_train` (≈ 0,25 με 5 folds) και είναι το πιο συντηρητικό από τα δύο.

Το test Parquet δεν διαβάζεται πριν ολοκληρωθεί το graph και επιλεγεί το plan.
Τα μεγάλα OOF/test predictions και checkpoints παραμένουν εκτός Git, ενώ τα
compact aggregate metrics, statistical tests, final plan και run summary είναι
trackable για αναπαραγωγιμότητα.

## Μεθοδολογικές διορθώσεις (commit `220fbb3`, 17/10/2026)

Μετά τον μεθοδολογικό έλεγχο (`docs/METHODOLOGICAL_REVIEW_GR.docx`) διορθώθηκαν
τέσσερα σημεία που επηρέαζαν την εγκυρότητα της σύγκρισης:

| # | Πριν | Μετά | Εύρημα review |
|---|---|---|---|
| 1 | Το pipeline χτιζόταν μόνο από το plan· η Agentic LR έχανε το scaling | Το plan εφαρμόζεται πάνω στο `BaselinePreprocessingContract` | 4.2 |
| 2 | `is_transfer`/`is_cash_out` ήταν model features και διέρρεαν τη σβησμένη `type` | Αφαιρέθηκαν από τα model features | 4.1, 5.1 |
| 3 | Σε retry το LLM έβλεπε μόνο γενικά recommendations | Βλέπει plans, issues, αποφάσεις και per-model deltas | 5.6 |
| 4 | Μόνο Wilcoxon | Wilcoxon + Nadeau–Bengio, και τα δύο με Holm | 5.3 |

Τα αποτελέσματα πριν από αυτό το commit (run `agentic_20260805T090249Z`) δεν είναι
συγκρίσιμα με τα μεταγενέστερα και δεν πρέπει να αναμιγνύονται.

## Definition of Done για Agentic components

Ένα component θεωρείται ολοκληρωμένο μόνο όταν:

- έχει σαφές public interface,
- χρησιμοποιεί type hints και περιγραφικό docstring,
- έχει περιορισμένη και σαφή ευθύνη,
- δεν βασίζεται σε κρυφό global state,
- υποστηρίζει dependency injection,
- διαθέτει unit tests για επιτυχία και αποτυχία,
- παράγει κατανοητά errors,
- μπορεί να αντικατασταθεί χωρίς αλλαγή του υπόλοιπου graph,
- τεκμηριώνεται στο README ή στο παρόν αρχείο.
