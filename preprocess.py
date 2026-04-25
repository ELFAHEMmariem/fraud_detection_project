import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# Vérifie bien que le nom ici est EXACTEMENT load_and_clean_data
def load_and_clean_data(filepath):
    # Lecture du fichier
    df = pd.read_csv(filepath)
    
    # Scaling des colonnes Time et Amount
    scaler = RobustScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    # Séparation Features/Target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split Train/Test (stratify pour garder le ratio de fraude)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE pour équilibrer les classes
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test