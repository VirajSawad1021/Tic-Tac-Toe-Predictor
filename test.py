import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

columns = ["top-left", "top-middle", "top-right", 
           "middle-left", "middle-middle", "middle-right", 
           "bottom-left", "bottom-middle", "bottom-right", "target"]
df = pd.read_csv("tic-tac-toe.csv",header=None,names=columns)

encoders = {}
df_encoded = df.copy()
for col in df_encoded.columns[:-1]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le 

le_target = LabelEncoder()
df_encoded["target"] = le_target.fit_transform(df_encoded["target"])

X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

joblib.dump(clf, 'tic_tac_toe_random_forest_model.pkl')
joblib.dump(encoders, 'feature_encoders.pkl')
joblib.dump(le_target, 'target_encoder.pkl')