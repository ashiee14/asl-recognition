"""
analyze_confusion.py

Diagnostic tool -- shows a confusion matrix so you can see exactly which
letter is being mistaken for which, instead of guessing from the
precision/recall report alone.

HOW TO USE:
  Run after training (needs data/data_normalized.csv to exist):
    python analyze_confusion.py

  Read it as: rows = the TRUE letter, columns = what the model PREDICTED.
  The diagonal is correct predictions. Any large off-diagonal number tells
  you "true letter X is often predicted as Y" -- that's your target for
  more/cleaner data, or for a closer look at how you're forming those two
  handshapes (they may be more similar than expected).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

DATA_FILE = "data/data_normalized.csv"

df = pd.read_csv(DATA_FILE, header=None)

# Only static ASL letters
VALID_LETTERS = list("ABCDEFGHI KLMNOPQRSTUVWXY".replace(" ", ""))

df = df[df.iloc[:, -1].isin(VALID_LETTERS)]

print("Letters being evaluated:", sorted(df.iloc[:, -1].unique()))

X, y = df.iloc[:, :-1], df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

labels = sorted(y.unique())
preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("Confusion matrix (rows = true letter, columns = predicted):\n")
print(cm_df.to_string())

print("\nTop confusions (true -> predicted : count), excluding correct predictions:")
confusions = []
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        if i != j and cm[i][j] > 0:
            confusions.append((cm[i][j], true_label, pred_label))

confusions.sort(reverse=True)
for count, true_label, pred_label in confusions[:15]:
    print(f"  {true_label} -> {pred_label} : {count} times")