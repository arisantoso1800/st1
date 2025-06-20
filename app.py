import pandas as pd
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # untuk load model

url = 'https://docs.google.com/spreadsheets/d/14eZEx-7Gi7txnTR8IBz6qL4Ydll2ZoLj7dpnalF2574/edit?gid=0#gid=0'
df1 = url.replace('/edit?gid=', '/export?format=csv&gid=').replace('#gid=', '&gid=')

df = pd.read_csv(df1)

df = df[['KELAS_RAWAT','DIAGLIST','INACBG']]

# Encode fitur gabungan dan target
le_diag = LabelEncoder()
le_inacbg = LabelEncoder()

X_full = le_diag.fit_transform(df['DIAGLIST']).reshape(-1, 1)
y_full = le_inacbg.fit_transform(df['INACBG'])

# Split data
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Train RandomForest
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_full, y_train_full)

# Save the trained model before loading it
joblib.dump(rf_full, 'rf.pkl')                  # Model machine learning       # TF-IDF atau CountVectorizer

st.title("Prediksi Kode INACBG")

# Form input
kelas = st.selectbox("Kelas Rawat", [1, 2])
diagnosa = st.text_input("Diagnosa (contoh: N18.5;J81;J80)")

if st.button("Prediksi"):
    if diagnosa:

        # Gabungkan fitur kelas rawat (numerik) + diagnosis (vektor teks)
        # input_data = pd.DataFrame([[kelas,diagnosa]], columns=['KELAS_RAWAT','DIAGLIST'])
        # sample_input = 'N18.5;J81;J80'
        sample_input_encoded = le_diag.transform([diagnosa]).reshape(-1, 1)

        predicted_inacbg = le_inacbg.inverse_transform(rf_full.predict(sample_input_encoded))

        # Prediksi
        # pred = rf_full.predict(input_data)[0]
        st.success(f"💡 Kode INACBG: {predicted_inacbg[0]}")
        # st.success(f"💡 Kode INACBG: {diagnosa}")
    else:
        st.warning("Masukkan teks diagnosis terlebih dahulu.")
