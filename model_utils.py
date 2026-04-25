import xgboost as xgb
from sklearn.neural_network import MLPClassifier

def get_xgb_model():
    # Modèle XGBoost performant pour les données tabulaires
    return xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42,
        eval_metric='logloss'
    )

def get_dl_model():
    # Réseau de neurones (MLP) compatible avec ta version de Python
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        max_iter=20, 
        random_state=42,
        verbose=True
    )