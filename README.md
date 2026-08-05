# Agentic AI για Αυτοματοποιημένη Διαχείριση Ποιότητας Δεδομένων σε FinTech

## Ανάπτυξη και αξιολόγηση συστήματος ανίχνευσης απάτης

Το repository αποτελεί το πρακτικό και πειραματικό μέρος της διπλωματικής εργασίας με αντικείμενο τη σχεδίαση, ανάπτυξη και αξιολόγηση ενός συστήματος **Agentic AI** για την αυτοματοποιημένη διαχείριση ποιότητας δεδομένων σε περιβάλλον μεγάλης κλίμακας FinTech.

Η αναλυτική καταγραφή των ολοκληρωμένων εργασιών και των επόμενων βημάτων
διατηρείται στο [`PROGRESS.md`](PROGRESS.md).

Οι δεσμευτικές αρχές αντικειμενοστρεφούς και επεκτάσιμου σχεδιασμού του Agentic
AI συστήματος περιγράφονται στο [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Η κεντρική ερευνητική ιδέα είναι να εξεταστεί εάν ένα αυτόνομο επίπεδο Τεχνητής Νοημοσύνης στη διαδικασία προετοιμασίας δεδομένων μπορεί να βελτιώσει:

- την ποιότητα των δεδομένων,
- την απόδοση συμβατικών μοντέλων Μηχανικής Μάθησης,
- την αναπαραγωγιμότητα και την αιτιολόγηση των αποφάσεων preprocessing.

Το Agentic AI δεν αντικαθιστά τα μοντέλα πρόβλεψης και δεν αποτελεί έναν επιπλέον classifier. Λειτουργεί ως **Agentic AI orchestration layer** πριν και γύρω από την εκπαίδευση, αναλαμβάνοντας τη διάγνωση προβλημάτων ποιότητας, την επιλογή και εφαρμογή μετασχηματισμών, την αξιολόγηση των αποτελεσμάτων και την αναθεώρηση της στρατηγικής preprocessing.

Τα τελικά μοντέλα ανίχνευσης απάτης είναι:

- Logistic Regression,
- Decision Tree,
- Random Forest.

Ο Data Profiler και η παραγωγή προτάσεων μπορούν να λειτουργούν χωρίς labels. Ωστόσο, ο ML Evaluator χρησιμοποιεί την πραγματική κλάση `isFraud`, επομένως ο συνολικός μηχανισμός δεν είναι αυστηρά μη επιβλεπόμενος.

## 1. Ερευνητικά ερωτήματα και υποθέσεις

Η μελέτη οργανώνεται γύρω από τα δύο επίσημα ερευνητικά ερωτήματα που
διατυπώνονται στο κείμενο της διπλωματικής:

- **RQ1:** Παρουσιάζουν τα μοντέλα που εκπαιδεύονται μετά από preprocessing με Agentic AI καλύτερη προβλεπτική απόδοση από τα ίδια μοντέλα όταν χρησιμοποιείται ένα συμβατικό και προκαθορισμένο preprocessing pipeline, και είναι οι παρατηρούμενες διαφορές στατιστικά σημαντικές και πρακτικά ουσιώδεις;
- **RQ2:** Ποιο είναι το πρόσθετο υπολογιστικό κόστος του Agentic AI και δικαιολογείται από τις βελτιώσεις στην ποιότητα των δεδομένων και στην προβλεπτική απόδοση των μοντέλων;

Για το RQ1, η βασική ερευνητική υπόθεση είναι:

```text
H₁: μΔ > 0
```

όπου `Δ` είναι η διαφορά της προκαθορισμένης κύριας μετρικής απόδοσης μεταξύ
Agentic και conventional pipeline. Η μηδενική υπόθεση είναι:

```text
H₀: μΔ = 0
```

Για το RQ2 δεν διατυπώνεται ξεχωριστή στατιστική υπόθεση. Αξιολογείται το ισοζύγιο
οφέλους και κόστους με βάση την ποιότητα δεδομένων, την προβλεπτική απόδοση, τον
χρόνο εκτέλεσης, τη χρήση μνήμης, τις κλήσεις προς το LLM και τον αριθμό των
Agent iterations. Η μη επιβεβαίωση της θετικής υπόθεσης του RQ1 ή η διαπίστωση
δυσανάλογου κόστους στο RQ2 αποτελούν επίσης έγκυρα ερευνητικά αποτελέσματα.

## 2. Σύνολο δεδομένων

Για την πειραματική αξιολόγηση χρησιμοποιείται το Kaggle dataset **Synthetic Financial Datasets for Fraud Detection**, το οποίο δημιουργήθηκε με τον προσομοιωτή PaySim.

- 6.362.620 συναλλαγές,
- 8.213 fraudulent συναλλαγές,
- ποσοστό απάτης περίπου 0,129%,
- 744 ωριαία βήματα,
- περίοδος προσομοίωσης 30 ημερών.

Ο μεγάλος όγκος και η ακραία ανισορροπία καθιστούν το PaySim κατάλληλο για τη μελέτη ενός FinTech pipeline μεγάλης κλίμακας. Θα χρησιμοποιηθεί το πλήρες dataset, columnar αποθήκευση και καταγραφή χρόνου, μνήμης και κλιμάκωσης των επιμέρους σταδίων.

### Βασικές μεταβλητές

- `step`: ώρα της προσομοίωσης,
- `type`: είδος συναλλαγής,
- `amount`: ποσό συναλλαγής,
- `nameOrig`: λογαριασμός προέλευσης,
- `nameDest`: λογαριασμός προορισμού,
- μεταβλητές υπολοίπων πριν και μετά τη συναλλαγή,
- `isFraud`: πραγματική κλάση απάτης,
- `isFlaggedFraud`: αποτέλεσμα υπάρχοντος κανόνα επισήμανσης.

Οι τύποι συναλλαγών είναι `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT` και `TRANSFER`.

Για την αποφυγή data leakage, το κύριο πείραμα πραγματοποιείται:

- χωρίς το `isFlaggedFraud`,
- χωρίς τις τέσσερις balance μεταβλητές,
- χωρίς άμεση κωδικοποίηση των `nameOrig` και `nameDest`.

Επιτρέπονται ασφαλή παράγωγα χαρακτηριστικά, όπως η ένδειξη ότι ο παραλήπτης είναι merchant. Μπορεί να γίνει ξεχωριστό sensitivity experiment με τις balance μεταβλητές, χωρίς να αποτελεί το κύριο συμπέρασμα.

> Το PaySim dataset δεν αποθηκεύεται στο GitHub και πρέπει να παραμένει τοπικά στο `data/raw/`.

## 3. Πειραματικός σχεδιασμός

### Προσέγγιση Α: Conventional preprocessing

Εφαρμόζεται σταθερό, προκαθορισμένο pipeline που μπορεί να περιλαμβάνει:

- διαχείριση missing values,
- κωδικοποίηση του `type`,
- λογαριθμικό μετασχηματισμό του `amount`,
- robust ή standard scaling όπου απαιτείται,
- χρονικά χαρακτηριστικά από το `step`,
- αφαίρεση αναγνωριστικών,
- class weighting ή προκαθορισμένο undersampling,
- προκαθορισμένο feature engineering.

Η συνταγή παραμένει σταθερή σε όλα τα μοντέλα και τις επαναλήψεις.

### Προσέγγιση Β: Agentic AI preprocessing

Ένας Agent, υλοποιημένος με LangGraph, εξετάζει το dataset, δημιουργεί transformation plan, εφαρμόζει εγκεκριμένες ενέργειες, εκπαιδεύει τα μοντέλα, αξιολογεί τα αποτελέσματα και αποφασίζει εάν θα διατηρήσει ή θα αναθεωρήσει τη στρατηγική.

Η σύγκριση γίνεται ανάμεσα σε δύο ολοκληρωμένα pipelines:

```text
Conventional preprocessing  vs  Agentic AI preprocessing
```

Και στις δύο περιπτώσεις χρησιμοποιούνται:

- τα ίδια δεδομένα,
- τα ίδια training και validation folds,
- το ίδιο test set,
- τα ίδια μοντέλα και random seeds,
- αντίστοιχο hyperparameter budget,
- οι ίδιες μετρικές αξιολόγησης.

## 4. Θέση του Agentic AI στο pipeline

```text
PaySim
  ↓
Temporal split
  ↓
Development set ──→ Data Profiler ──→ Strategy Generator
                                            ↓
                                      Plan Validator
                                            ↓
                                  Transformation Executor
                                            ↓
                              Data Quality & ML Evaluators
                                            ↓
                                  Feedback / Refinement
                                            ↺
                                  Best approved pipeline
                                            ↓
                              Untouched temporal test set
```

Ο Agent διατηρεί κατάσταση και ιστορικό, εκτελεί συγκεκριμένες ενέργειες, παρατηρεί και συγκρίνει αποτελέσματα, αλλάζει στρατηγική όταν δεν υπάρχει βελτίωση και τερματίζει βάσει προκαθορισμένων κριτηρίων.

Το LangGraph οργανώνει την εφαρμογή ως `StateGraph`, όπου οι κόμβοι διαβάζουν και ενημερώνουν κοινό state, με conditional edges, loops και ελεγχόμενο τερματισμό.

## 5. Αρχιτεκτονική του LangGraph Agent

### 5.1 Data Profiler

Εξετάζει αποκλειστικά τα training δεδομένα και παράγει δομημένο Data Quality Report με:

- αριθμό γραμμών και στηλών,
- schema και τύπους δεδομένων,
- missing-value rates και duplicates,
- invalid values και outlier indicators,
- skewness αριθμητικών μεταβλητών,
- categorical cardinality,
- σταθερές ή σχεδόν σταθερές μεταβλητές,
- class distribution,
- πιθανές ενδείξεις leakage,
- χρήση μνήμης και χρόνο επεξεργασίας.

Το class imbalance καταγράφεται ως modeling risk και όχι αυτόματα ως σφάλμα ποιότητας.

Το LLM δεν λαμβάνει ολόκληρο το dataset. Λαμβάνει το report, το schema, το data dictionary, συγκεντρωτικά στατιστικά, μικρό ανωνυμοποιημένο δείγμα και τα αποτελέσματα προηγούμενων iterations.

### 5.2 Strategy Generator

Το LLM αποφασίζει δυναμικά:

- αν απαιτείται preprocessing,
- ποια προβλήματα θα αντιμετωπιστούν,
- ποιες τεχνικές και παράμετροι θα χρησιμοποιηθούν,
- σε ποιες μεταβλητές και με ποια σειρά,
- αν θα δημιουργηθούν νέα χαρακτηριστικά,
- πώς θα αντιμετωπιστεί η ανισορροπία των κλάσεων,
- αν χρειάζεται διαφορετικό pipeline ανά μοντέλο.

Ο ερευνητής παρέχει τα επιτρεπόμενα εργαλεία. Το LLM αποφασίζει ποια θα χρησιμοποιηθούν, πού, πώς και με ποια σειρά.

Ο Agent επιστρέφει structured JSON και όχι αυθαίρετο εκτελέσιμο Python code. Ενδεικτικά, κάθε ενέργεια περιγράφει τον τύπο της, τις στήλες, τις παραμέτρους και την αιτιολόγησή της.

### 5.3 Validator και Transformation Executor

Ο Validator απορρίπτει plans που:

- αναφέρονται σε ανύπαρκτη μεταβλητή,
- χρησιμοποιούν το `isFraud` ως predictor,
- χρησιμοποιούν πληροφορίες του test set,
- περιλαμβάνουν μη επιτρεπόμενη ενέργεια,
- αφαιρούν υπερβολικά μεγάλο μέρος των δεδομένων,
- δεν είναι έγκυρα ή αναπαραγώγιμα,
- επιχειρούν filesystem, network ή system commands,
- εφαρμόζουν oversampling πριν από τον διαχωρισμό των folds.

Ο Executor εφαρμόζει μόνο εγκεκριμένες ενέργειες. Κάθε iteration ξεκινά από το ίδιο αρχικό training dataset:

```text
Raw training data + Strategy 1 → Dataset 1
Raw training data + Strategy 2 → Dataset 2
Raw training data + Strategy 3 → Dataset 3
```

Οι μετασχηματισμοί διαφορετικών iterations δεν συσσωρεύονται ανεξέλεγκτα.

### 5.4 Data Quality Evaluator

Υπολογίζει άμεσες διαστάσεις ποιότητας:

- completeness,
- validity,
- consistency,
- uniqueness,
- ποσοστό invalid values,
- ποσοστό δεδομένων που αφαιρέθηκε,
- αποτυχημένα transformations,
- χρόνο preprocessing,
- μεταβολή του αριθμού χαρακτηριστικών.

Το προτεινόμενο σύνθετο score είναι:

```text
DQScore = 0.30·Completeness + 0.25·Validity
        + 0.25·Consistency  + 0.20·Uniqueness
```

Τα βάρη ορίζονται πριν από τα τελικά πειράματα και παραμένουν σταθερά.

### 5.5 Machine Learning Evaluator

Σε κάθε iteration εκπαιδεύει Logistic Regression, Decision Tree και Random Forest και υπολογίζει:

- Accuracy,
- Recall / TPR,
- Specificity / TNR,
- Precision,
- F1-score,
- ROC-AUC,
- PR-AUC,
- Balanced Accuracy,
- confusion matrix,
- χρόνο εκπαίδευσης και πρόβλεψης.

Λόγω της ακραίας ανισορροπίας, η Accuracy δεν αποτελεί κύρια μετρική. Η αξιολόγηση βασίζεται κυρίως σε PR-AUC, Recall και F1-score, με παράλληλη παρακολούθηση του Precision.

### 5.6 Feedback and Refinement

Ο Feedback node λαμβάνει τα quality reports, τη στρατηγική, τις μετρικές, errors και warnings, το ιστορικό και το καλύτερο pipeline. Επιστρέφει μία από τις αποφάσεις:

- `ACCEPT`,
- `RETRY`,
- `STOP_NO_IMPROVEMENT`,
- `STOP_MAX_ITERATIONS`,
- `STOP_INVALID_STRATEGIES`.

Για την κύρια μελέτη προβλέπονται έως τρία iterations. Ο Agent τερματίζει νωρίτερα όταν επιτευχθεί η απαιτούμενη βελτίωση, δεν υπάρχει βελτίωση σε δύο συνεχόμενα iterations, απορριφθούν όλες οι νέες προτάσεις ή παρουσιαστεί μη ανακτήσιμο σφάλμα.

Το state αποθηκεύει iteration, reports, plans, validation errors, metrics, καλύτερο score και pipeline, λόγο τερματισμού και πλήρες ιστορικό αποφάσεων. Checkpoints επιτρέπουν ανάκτηση από αποτυχίες και επιθεώρηση προηγούμενων καταστάσεων.

## 6. Controlled Data Quality Scenarios

### Σενάριο 1: Original PaySim

Ο Agent αντιμετωπίζει τις πραγματικές ιδιότητες του dataset:

- ακραίο class imbalance,
- skewed ποσά συναλλαγών,
- high-cardinality identifiers,
- πιθανές ακραίες παρατηρήσεις,
- categorical encoding,
- feature generation.

### Σενάριο 2: Controlled Quality Degradation

Σε αντίγραφο των δεδομένων εισάγονται ελεγχόμενα:

- missing values,
- duplicates,
- ακραίες αριθμητικές τιμές,
- invalid categorical values,
- type inconsistencies.

Οι αλλοιώσεις δημιουργούνται με σταθερό random seed και αποθηκεύονται σε corruption log. Εφόσον είναι γνωστές οι θέσεις και οι πραγματικές τιμές, αξιολογείται άμεσα η ικανότητα εντοπισμού και διόρθωσης προβλημάτων. Η υποβάθμιση γίνεται μετά τον αρχικό διαχωρισμό, ώστε να διατηρείται η ανεξαρτησία του test set.

## 7. Διαχωρισμός δεδομένων και αποφυγή leakage

Χρησιμοποιείται αρχικός χρονικός διαχωρισμός:

- περίπου τα πρώτα 80% των χρονικών βημάτων ως development set,
- περίπου τα τελευταία 20% ως τελικό temporal test set.

Το test set δεν χρησιμοποιείται από τον Profiler ή το LLM, ούτε για transformations, feature selection, threshold tuning ή feedback. Το καλύτερο pipeline εφαρμόζεται σε αυτό μόνο μία φορά μετά την ολοκλήρωση του Agentic loop.

Μέσα στο development set χρησιμοποιείται `RepeatedStratifiedKFold`, ενδεικτικά με 5 folds και 5 επαναλήψεις. Όλα τα preprocessing στάδια προσαρμόζονται μόνο στο training μέρος κάθε fold. Το SMOTE, εφόσον επιλεγεί, εφαρμόζεται μόνο μέσα στο training fold.

## 8. Ενσωμάτωση του κώδικα benchmarking

Το αρχικό `mlr.Rmd` του καθηγητή αποτελεί τη βάση της λογικής benchmarking και μεταφέρεται σε Python. Η υλοποίηση καλύπτει:

- train/test split και ορισμό positive class,
- προβλέψεις, confusion matrix και μετρικές,
- ROC και PR curves,
- repeated cross-validation,
- benchmarking διαφορετικών learners,
- paired statistical tests σε fold-level αποτελέσματα.

Ο ίδιος ανεξάρτητος benchmarking μηχανισμός χρησιμοποιείται τόσο στο conventional pipeline όσο και στον Agentic Evaluator, ώστε οι διαφορές να προκύπτουν από το preprocessing και όχι από διαφορετική διαδικασία αξιολόγησης.

## 9. Τεχνολογική στοίβα

- Python,
- LangGraph για orchestration και Agent state,
- LangChain model interfaces ή αντίστοιχο LLM API,
- Polars ή PyArrow για αποδοτική επεξεργασία,
- Parquet για columnar αποθήκευση,
- pandas και NumPy όπου απαιτούνται,
- scikit-learn για μοντέλα και μετρικές,
- imbalanced-learn για sampling pipelines,
- SciPy ή statsmodels για στατιστικούς ελέγχους,
- Matplotlib για διαγράμματα,
- JSON/JSONL για plans και Agent logs,
- MLflow ή structured experiment logs.

Για τον περιορισμό του κόστους, οι υποψήφιες στρατηγικές μπορούν να αξιολογούνται σε σταθερό development subset που περιλαμβάνει όλες τις fraud περιπτώσεις και αντιπροσωπευτικό δείγμα normal συναλλαγών. Η ίδια δειγματοληψία εφαρμόζεται και στο conventional pipeline.

## 10. Στατιστική σύγκριση

Για κάθε μοντέλο, fold και repeat υπολογίζεται:

```text
Δₘ = MetricAgentic,ₘ − MetricConventional,ₘ
```

Παρουσιάζονται:

- μέση και διάμεση διαφορά,
- τυπική απόκλιση,
- 95% confidence interval,
- paired Wilcoxon ή κατάλληλο corrected paired test,
- effect size,
- Holm correction για πολλαπλές συγκρίσεις.

Οι paired t-tests του αρχικού R script διατηρούνται ως αναφορά, αλλά η ανάλυση δεν βασίζεται αποκλειστικά σε αυτούς, επειδή τα cross-validation folds δεν είναι πλήρως ανεξάρτητα. Αξιολογείται τόσο η στατιστική όσο και η πρακτική σημασία, μαζί με τον χρόνο εκτέλεσης και τις μεταβολές σε Precision και Recall.

## 11. Προτεινόμενη δομή repository

```text
AgenticAI_thesis/
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   └── agent.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/agenticai_thesis/
│   ├── data/
│   ├── quality/
│   ├── modeling/
│   ├── agentic/
│   └── utils/
├── experiments/
├── reports/
├── figures/
├── logs/
├── notebooks/
└── tests/
```

Η ανάπτυξη οργανώνεται σε τρεις φάσεις:

1. **Conventional pipeline:** PaySim → Parquet → temporal split → profiler → ML pipelines → benchmarking.
2. **Agentic AI:** profile → strategy generator → validator → executor → evaluators → feedback.
3. **Τελικά πειράματα:** conventional vs Agentic και original vs degraded dataset → statistical tests → tables → figures → thesis results.

### Τρέχουσα υλοποίηση: dataset pipeline

Το πρώτο λειτουργικό επίπεδο μετατρέπει το πλήρες PaySim CSV σε συμπιεσμένο
Parquet, δημιουργεί leakage-safe χαρακτηριστικά και πραγματοποιεί χρονικό
διαχωρισμό σε development και ανέγγιχτο temporal test set.

Από PowerShell, στη ρίζα του repository:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_data_pipeline
```

Τα παραγόμενα αρχεία είναι:

```text
data/processed/paysim.parquet
data/splits/development.parquet
data/splits/temporal_test.parquet
```

Για την εκτέλεση των αυτοματοποιημένων ελέγχων:

```powershell
python -m pytest -q
```

Τα datasets και τα παραγόμενα Parquet αρχεία εξαιρούνται από το Git. Μόνο τα
configuration files, ο κώδικας, τα tests και τα κενά directory placeholders
αποθηκεύονται στο repository.

### Data Quality Profiler

Ο αυτόνομος profiler αναλύει αποκλειστικά το development set και δεν χρησιμοποιεί
LLM ή το temporal test set:

```powershell
$env:PYTHONPATH = "src"
python -m agenticai_thesis.quality.profiler
```

Το report αποθηκεύεται στο:

```text
reports/profiles/development_profile.json
```

Περιλαμβάνει dimensions, schema, missingness, cardinality, αριθμητικά στατιστικά,
skewness, categorical και class distributions, domain-invalid values, duplicates,
χρονικό εύρος, modeling risks, χρόνο εκτέλεσης και χρήση μνήμης. Τα duplicates
αφορούν ισότητα σε όλες τις στήλες του τελικού feature dataset. Επειδή έχουν ήδη
αφαιρεθεί identifiers και balances, δεν ερμηνεύονται αυτομάτως ως διπλές αρχικές
συναλλαγές.

### Conventional baseline

Το conventional benchmark χρησιμοποιεί σταθερό preprocessing και τα ίδια
materialized repeated-stratified folds για Logistic Regression, Decision Tree και
Random Forest. Εκτελείται με:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_conventional
```

Για την εσωτερική επιλογή στρατηγικής χρησιμοποιούνται όλες οι fraud εγγραφές του
development set και αναπαραγώγιμο normal sample με αναλογία 10:1. Το sample και τα
5×5 folds είναι κοινά για όλα τα μοντέλα. Παράγονται:

```text
reports/metrics/conventional_results.csv
reports/metrics/conventional_fold_results.csv
reports/metrics/conventional_oof_predictions.parquet
figures/confusion_matrices/
figures/pr_curves/
figures/roc_curves/
logs/conventional/run_summary.json
```

Οι μετρικές αυτές αφορούν αποκλειστικά εσωτερικό cross-validation σε dataset με
τεχνητά αυξημένο fraud prevalence. Δεν αποτελούν τελική εκτίμηση της επίδοσης στον
πραγματικό πληθυσμό PaySim. Η τελική σύγκριση θα γίνει στο πλήρες, ανέγγιχτο
temporal holdout, όπου διατηρείται η φυσική κατανομή των κλάσεων.

## 12. Τελικά παραδοτέα

- Πλήρης περιγραφή του PaySim και των περιορισμών του.
- Python μεταφορά και παραμετροποίηση του κώδικα του καθηγητή.
- Conventional preprocessing pipeline.
- Λειτουργικό LangGraph Agent.
- Data Profiler και Data Quality Report.
- Strategy Generator με structured output.
- Validator και ασφαλής Transformation Executor.
- Data Quality και ML Evaluator.
- Feedback loop με έως τρία iterations.
- Αποθήκευση του καλύτερου pipeline.
- Logistic Regression, Decision Tree και Random Forest.
- Repeated cross-validation και τελικό temporal holdout.
- Confusion matrices και όλες οι απαιτούμενες μετρικές.
- Άμεσες μετρικές ποιότητας δεδομένων.
- Στατιστική σύγκριση conventional και Agentic pipeline.
- Καταγραφή prompts, LLM responses, errors και αποφάσεων.
- Μέτρηση χρόνου εκτέλεσης και χρήσης πόρων.
- Πλήρως αναπαραγώγιμο repository με seeds και package versions.

## 13. Open-weight LLM μέσω Groq Free Tier

Ο Strategy Generator χρησιμοποιεί το `openai/gpt-oss-20b` μέσω του Groq Free Tier. Το πλήρες
dataset δεν αποστέλλεται στο LLM. Το μοντέλο λαμβάνει αποκλειστικά compact schema,
δείκτες ποιότητας, baseline metrics και feedback προηγούμενων iterations.

Η σύνδεση παραμένει αντικαταστάσιμη:

```text
StrategyGenerator
      │
ModelClientProtocol
      ├── GroqModelClient   → openai/gpt-oss-20b (κύριο)
      ├── OllamaModelClient → προαιρετικό local fallback
      └── FakeModelClient   → deterministic unit tests
```

Το API key δημιουργείται στο Groq Console και ορίζεται μόνο στο environment:

```powershell
$env:GROQ_API_KEY = "your-free-tier-key"
```

Το πραγματικό key δεν γράφεται σε YAML, source code, logs ή Agent state και τα `.env`
αρχεία αγνοούνται από το Git. Οι παράμετροι βρίσκονται στο `configs/agent.yaml`.
Το Groq καλείται μέσω HTTPS, με temperature `0`, JSON Schema structured output και
χωρίς δυνατότητα παραγωγής/εκτέλεσης αυθαίρετου κώδικα. Η έξοδος επικυρώνεται ξανά
ως immutable `TransformationPlan` πριν φτάσει στον Validator και στον Executor.
Σε HTTP 429 ο client κάνει μόνο περιορισμένα retries και τελικά σταματά μέχρι το
Free Tier quota reset· δεν υπάρχει αυτόματη μετάβαση σε πληρωμένο tier.

### LangGraph Agent workflow

Το LangGraph χρησιμοποιείται αποκλειστικά για orchestration. Κάθε node είναι μικρή
κλάση με μία ευθύνη και λαμβάνει τις υπηρεσίες του μέσω dependency injection:

```text
START
  ↓
prepare_iteration → generate_strategy → validate_strategy
                                         ├── valid → evaluate_candidate
                                         └── invalid → assess_invalid_strategy
                                                           ↓
                         decide_feedback → record_iteration/checkpoint
                                  ↑                 ├── RETRY
                                  └─────────────────┘
                                                    └── ACCEPT/STOP → END
```

Το graph state περιέχει μόνο immutable/serializable domain objects και compact
metrics. Raw rows, fitted pipelines, estimators και fold predictions παραμένουν
στον injected `AgentCandidateEvaluator`. Ο evaluator συνθέτει τον ασφαλή Executor,
τον quality evaluator και τον ML evaluator πάνω στα κοινά cross-validation folds.

Η πρόσβαση σε temporal-test context απορρίπτεται πριν κληθεί το LLM. Invalid plans
δεν φτάνουν ποτέ στον Executor. Κάθε ολοκληρωμένο iteration αποθηκεύεται atomically
με plan, validation issues, assessment, feedback, warnings και execution errors.

### Πλήρης εκτέλεση Agentic πειράματος

Το ενιαίο, fail-fast production workflow εκτελεί διαδοχικά dataset preparation,
conventional benchmark και LangGraph Agentic benchmark:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_full_experiment
```

Όταν τα immutable temporal partitions υπάρχουν ήδη, μπορούν να εκτελεστούν μόνο
τα υπολογιστικά πειραματικά στάδια χωρίς νέα μετατροπή του raw CSV:

```powershell
python -m experiments.run_full_experiment --stages conventional agentic
```

Για μεμονωμένη επανάληψη μόνο του Agentic πειράματος:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_agentic
```

Ο `FullExperimentRunner` είναι class-based coordinator. Ελέγχει configurations
και dataset prerequisites πριν την εκτέλεση, καλεί κάθε εξειδικευμένο runner σε
ξεχωριστό process με τον ενεργό Python interpreter και σταματά αμέσως αν κάποιο
στάδιο αποτύχει, ώστε να μη χρησιμοποιηθούν ελλιπή ή παλιά artifacts.

Η τελική πραγματική εκτέλεση χρησιμοποίησε 72.149 development observations, 5×5
shared repeated-stratified folds και controlled MCAR missingness σε 3.607 τιμές της
στήλης `type`. Ο Agent έκανε `ACCEPT` στο πρώτο iteration με plan:

- categorical most-frequent imputation στο `type`,
- one-hot encoding με ασφαλή unknown-category handling.

Το Data Quality Score αυξήθηκε από `0,998062` σε `0,999723` (`+0,001661`). Η μέση
primary-metric μεταβολή έναντι του degraded conventional baseline ήταν `+0,002429`,
με runtime multiplier `1,667`, χωρίς παραβίαση Recall/Precision guardrails.

Στο untouched temporal holdout τα PR-AUC αποτελέσματα ήταν:

| Model | Conventional | Agentic | Διαφορά |
|---|---:|---:|---:|
| Logistic Regression | 0,205972 | 0,213225 | +0,007253 |
| Decision Tree | 0,080481 | 0,075760 | -0,004721 |
| Random Forest | 0,353971 | 0,355852 | +0,001881 |

Τα αποτελέσματα δείχνουν ότι η βελτίωση ποιότητας δεν συνεπάγεται ομοιόμορφη
βελτίωση κάθε classifier. Η επίδραση ήταν θετική στη Logistic Regression και στο
Random Forest PR-AUC, αλλά αρνητική στο Decision Tree. Τα
πλήρη fold metrics, temporal metrics και Holm-corrected Wilcoxon tests βρίσκονται
στο `reports/`.

### Χρήση μνήμης και LLM

Η χρήση μνήμης μετράται με δειγματοληψία του RSS του κύριου process και των child
processes. Επειδή τα δύο στάδια δεν ξεκινούν από το ίδιο RSS, η κύρια συγκρίσιμη
μέτρηση είναι η αύξηση από την αρχική έως τη μέγιστη τιμή και όχι τα απόλυτα peaks:

| Pipeline | Αρχικό RSS | Peak RSS | Αύξηση peak |
|---|---:|---:|---:|
| Degraded conventional benchmark | 607,289 MiB | 627,738 MiB | 20,449 MiB |
| Selected Agentic candidate | 258,555 MiB | 287,375 MiB | 28,820 MiB |

Το Agentic candidate χρησιμοποίησε κατά το συγκεκριμένο run `8,371 MiB` μεγαλύτερη
αύξηση peak RSS, δηλαδή περίπου `40,9%` υψηλότερη από το conventional benchmark.
Η κλήση του `openai/gpt-oss-20b` μέσω Groq ολοκληρώθηκε με ένα επιτυχές request,
χωρίς failures ή retries: `1.442` prompt tokens, `1.502` completion tokens και
`2.944` tokens συνολικά, με provider-reported χρόνο `1,728 s`. Το παρατηρούμενο
κόστος ήταν `$0,00` με τη διαμορφωμένη χρήση του Groq Free Tier. Δεν αποθηκεύονται
prompts, responses ή API secrets στο telemetry.

Οι παραπάνω τιμές αφορούν μία πλήρη εκτέλεση. Το LLM μπορεί να παράγει διαφορετικό
έγκυρο plan ακόμη και με μηδενικό temperature, επομένως το αποθηκευμένο run ID
`agentic_20260805T090249Z` αποτελεί το ενιαίο επίσημο run για αυτά τα αποτελέσματα.

### Τελικά figures για τη συγγραφή

Τα publication-ready συγκριτικά figures δημιουργούνται αποκλειστικά από τα
τελικά temporal metrics και το Agentic run summary:

```powershell
$env:PYTHONPATH = "src"
python -m experiments.generate_final_figures
```

Ο φάκελος `figures/final_comparisons/` περιέχει:

- temporal PR-AUC σύγκριση Conventional–Agentic ανά classifier,
- temporal Precision/Recall σύγκριση,
- Data Quality Score πριν και μετά το επιλεγμένο plan,
- runtime multiplier έναντι του conventional baseline,
- το πραγματικά υλοποιημένο LangGraph workflow με validation και retry routing.
- high-level αρχιτεκτονική ολόκληρου του experimental workflow, από το raw PaySim
  έως το temporal holdout και τα τελικά thesis outputs.

## 14. Μεθοδολογικό συμπέρασμα

Η εργασία δεν εξετάζει απλώς εάν ένα LLM μπορεί να προτείνει τεχνικές preprocessing. Αναπτύσσει έναν ολοκληρωμένο Agentic AI μηχανισμό που λειτουργεί ως αυτόνομος Data Scientist μέσα σε σαφώς καθορισμένα όρια.

Ο Agent παρατηρεί τα δεδομένα, εντοπίζει προβλήματα, δημιουργεί στρατηγική, εφαρμόζει ασφαλείς μετασχηματισμούς, εκπαιδεύει και αξιολογεί μοντέλα, λαμβάνει feedback, αναθεωρεί τις αποφάσεις του και τερματίζει επιλέγοντας το καλύτερο pipeline.

Η βασική πειραματική συνεισφορά είναι η άμεση και δίκαιη σύγκριση ενός συμβατικού preprocessing pipeline με ένα επαναληπτικό LangGraph Agentic AI pipeline, χρησιμοποιώντας τα ίδια μοντέλα, folds, δεδομένα και κριτήρια αξιολόγησης.
