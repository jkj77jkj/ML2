import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

clf = load_model()

st.title("행동 분류 웹 앱")
st.write("걷기 · 뛰기 · 정지 · 계단오르기 상태 분류")

uploaded = st.file_uploader("CSV 파일 업로드", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### 입력 데이터 미리보기", df.head(5))

    def extract_features(df, window_size=50, step_size=25):
        X, rows = [], []
        for start in range(0, len(df) - window_size, step_size):
            w = df.iloc[start:start+window_size]
            mag = np.sqrt(w['acc_x']**2 + w['acc_y']**2 + w['acc_z']**2)
            feats = [
                w['acc_x'].mean(), w['acc_y'].mean(), w['acc_z'].mean(),
                w['acc_x'].std(),  w['acc_y'].std(),  w['acc_z'].std(),
                mag.mean(),       mag.std()
            ]
            X.append(feats)
            rows.append(start)
        return np.array(X), rows

    X_feats, indices = extract_features(df)
    preds = clf.predict(X_feats)

    result_df = pd.DataFrame({
        "start_index": indices,
        "prediction": preds
    })

    st.write("### 예측 결과", result_df)
    st.bar_chart(result_df["prediction"].value_counts())
