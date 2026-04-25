import os
from preprocess import load_and_clean_data
from model_utils import get_xgb_model, get_dl_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import shap

def run_project():
    # Puisque le fichier est au même endroit que main.py
    path = 'creditcard.csv'
    
    if not os.path.exists(path):
        print(f"❌ Erreur : Le fichier '{path}' est introuvable dans le dossier actuel.")
        print("Vérifiez que le fichier n'est pas caché dans un sous-dossier.")
        return

    # 1. Chargement
    print("--- ⏳ Étape 1 : Chargement et Preprocessing (SMOTE) ---")
    try:
        X_train, X_test, y_train, y_test = load_and_clean_data(path)
        print(f"✅ Données prêtes ! Taille du set d'entraînement : {X_train.shape}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return

    # 2. XGBoost
    print("\n--- 🤖 Étape 2 : Entraînement XGBoost ---")
    xgb_mod = get_xgb_model()
    xgb_mod.fit(X_train, y_train)
    y_pred_xgb = xgb_mod.predict(X_test)
    print("\n📊 Rapport de performance XGBoost :")
    print(classification_report(y_test, y_pred_xgb))

    # 3. Réseau de Neurones
    print("\n--- 🧠 Étape 3 : Entraînement Réseau de Neurones (MLP) ---")
    dl_mod = get_dl_model()
    dl_mod.fit(X_train, y_train)
    y_pred_dl = dl_mod.predict(X_test)
    print("\n📊 Rapport de performance Deep Learning :")
    print(classification_report(y_test, y_pred_dl))

    # 4. Explicabilité (SHAP)
    print("\n--- 🔍 Étape 4 : Génération de l'explicabilité (SHAP) ---")
    explainer = shap.TreeExplainer(xgb_mod)
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], show=False)
    plt.savefig('resultat_shap.png')
    print("✅ Graphique SHAP sauvegardé sous 'resultat_shap.png'")

if __name__ == "__main__":
    run_project()