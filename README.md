# Agentic AI για Αυτοματοποιημένη Διαχείριση Ποιότητας Δεδομένων σε FinTech

## Ανάπτυξη και αξιολόγηση συστήματος ανίχνευσης απάτης

Το repository αποτελεί το πρακτικό και πειραματικό μέρος της διπλωματικής εργασίας με αντικείμενο τη σχεδίαση, ανάπτυξη και αξιολόγηση ενός συστήματος **Agentic AI** για την αυτοματοποιημένη διαχείριση ποιότητας δεδομένων σε περιβάλλον μεγάλης κλίμακας FinTech.

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

## 13. Μεθοδολογικό συμπέρασμα

Η εργασία δεν εξετάζει απλώς εάν ένα LLM μπορεί να προτείνει τεχνικές preprocessing. Αναπτύσσει έναν ολοκληρωμένο Agentic AI μηχανισμό που λειτουργεί ως αυτόνομος Data Scientist μέσα σε σαφώς καθορισμένα όρια.

Ο Agent παρατηρεί τα δεδομένα, εντοπίζει προβλήματα, δημιουργεί στρατηγική, εφαρμόζει ασφαλείς μετασχηματισμούς, εκπαιδεύει και αξιολογεί μοντέλα, λαμβάνει feedback, αναθεωρεί τις αποφάσεις του και τερματίζει επιλέγοντας το καλύτερο pipeline.

Η βασική πειραματική συνεισφορά είναι η άμεση και δίκαιη σύγκριση ενός συμβατικού preprocessing pipeline με ένα επαναληπτικό LangGraph Agentic AI pipeline, χρησιμοποιώντας τα ίδια μοντέλα, folds, δεδομένα και κριτήρια αξιολόγησης.
