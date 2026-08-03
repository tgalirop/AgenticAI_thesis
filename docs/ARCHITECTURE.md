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
- Το temporal test set δεν είναι διαθέσιμο στους Agent components.
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
έλεγχο και αρνείται execution σε validation-fold ή temporal-test context, ακόμη
και αν του δοθεί χειροκίνητα ένα φαινομενικά έγκυρο result.

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
