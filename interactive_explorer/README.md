# Agentic AI Thesis Interactive Explorer

Αυτός ο φάκελος είναι ένα **απομονωμένο, read-only web υποσύστημα** που εξηγεί
διαδραστικά το πείραμα της διπλωματικής. Δεν αλλάζει τον κώδικα του πειράματος,
δεν επεμβαίνει στα datasets και δεν εκτελεί νέο LLM request.

## Τι περιλαμβάνει

- `frontend/`: React + TypeScript εφαρμογή με το διαδραστικό workflow και τα
  γραφήματα.
- `backend/`: FastAPI υπηρεσία που επιστρέφει ένα ελεγμένο JSON snapshot.
- `tools/build_snapshot.py`: δημιουργεί το snapshot από τα πραγματικά CSV,
  JSON και prediction Parquet artifacts του repository.
- `compose.yaml`: ξεκινά frontend και backend ως ανεξάρτητα containers.

## Γρήγορη εκκίνηση

Από τον φάκελο `interactive_explorer`:

```powershell
docker compose up --build
```

Έπειτα άνοιξε:

- εφαρμογή: <http://localhost:3000>
- API documentation: <http://localhost:8000/docs>

Για τερματισμό:

```powershell
docker compose down
```

## Ανανέωση των δεδομένων της εφαρμογής

Το versioned αρχείο `backend/data/thesis_snapshot.json` επιτρέπει στο app να
λειτουργεί σε νέο clone χωρίς το PaySim dataset ή τα μεγάλα prediction files.
Μετά από νέα επίσημη εκτέλεση του πειράματος, ανανέωσέ το από τη ρίζα του repo:

```powershell
python interactive_explorer/tools/build_snapshot.py
```

Ο generator:

1. διαβάζει τα επίσημα aggregate artifacts,
2. υπολογίζει PR/ROC curve points από τις αποθηκευμένες προβλέψεις,
3. επαληθεύει ότι τα AUC των καμπυλών συμφωνούν με τα επίσημα CSV,
4. γράφει μόνο ένα μικρό JSON snapshot μέσα στο web υποσύστημα.

Τα prediction Parquet αρχεία παραμένουν εκτός GitHub, όπως ορίζει το υπάρχον
`.gitignore`. Το snapshot περιέχει μόνο συγκεντρωτικά αποτελέσματα και
downsampled σημεία καμπυλών — όχι raw συναλλαγές ή προσωπικά δεδομένα.

## Επιστημονική χρήση των οθονών

- **Workflow:** εξηγεί βήμα-βήμα τη μεθοδολογία και συνδέει κάθε βήμα με τα
  πραγματικά αρχεία του repository.
- **Data journey:** δείχνει chronological partitions, class imbalance, sampling και
  controlled MCAR degradation.
- **Model evaluation:** παρουσιάζει PR curves, ROC curves, confusion matrices,
  final test comparisons και paired-fold διαφορές.
- **Statistics:** ξεχωρίζει τα paired tests των 25 folds από τη μία περιγραφική
  αξιολόγηση του test dataset.
- **Resources:** παρουσιάζει runtime, memory και LLM/token telemetry.

Οι καμπύλες του test dataset και των out-of-fold predictions προέρχονται από
πραγματικά `y_true`/`y_score`. Δεν είναι σχεδιασμένα παραδείγματα.
