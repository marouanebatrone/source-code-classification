import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.logger import get_logger

log = get_logger("train")

FEATURES = ["perplexity", "comment_ratio", "id_length", "ast_nodes", "ast_depth"]
MODELS   = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "Gradient Boosting"   : GradientBoostingClassifier(random_state=42),
}

def run():
    df = pd.read_csv("extracted_features.csv")
    log.info(f"Loaded {len(df)} samples. Label distribution:\n{df['label'].value_counts().to_string()}")

    X = df[FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("\n" + "=" * 45)
    print("           MODEL RESULTS")
    print("=" * 45)

    best_model, best_acc = None, 0
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name:25}: {acc:.2%}")
        if acc > best_acc:
            best_acc, best_model = acc, (name, model)

    name, model = best_model
    print(f"\n{'=' * 45}")
    print(f"  BEST MODEL: {name} ({best_acc:.2%})")
    print(f"{'=' * 45}")
    print(classification_report(y_test, model.predict(X_test), target_names=["Human", "AI"]))

    rf = MODELS["Random Forest"]
    print("FEATURE IMPORTANCE (Random Forest):")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:20}: {imp:.4f}")

if __name__ == "__main__":
    run()