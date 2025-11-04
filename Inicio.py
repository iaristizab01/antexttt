import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ----------------------------
# CONFIGURACIÓN VISUAL
# ----------------------------
st.set_page_config(page_title="La máquina que busca sentido", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: white;
        color: black;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stApp {
        background-color: white;
        color: black;
    }
    h1 {
        text-align: center;
        font-size: 2.2em;
        margin-bottom: 0.2em;
        color: #111;
    }
    .stMarkdown, .stTextInput, .stDataFrame {
        color: #111 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# NARRATIVA
# ----------------------------
st.title("🧠 La máquina que busca sentido")
st.caption("Un experimento sobre cómo los algoritmos descubren afinidades entre frases humanas.")

st.write("""
Cada línea es una historia.  
Cada palabra, una frecuencia.  
La máquina no entiende emociones, pero mide **distancias entre significados**.  
""")

# ----------------------------
# INTERFAZ
# ----------------------------
text_input = st.text_area(
    "Escribe tus frases o documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("Formula una pregunta o frase para comparar:", "Who is playing?")

# ----------------------------
# PROCESAMIENTO TF-IDF
# ----------------------------
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🔍 Buscar sentido"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos una frase o documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.markdown("### 🧩 Mapa de frecuencias (TF-IDF)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 🗣️ Resultado del experimento")
        st.write(f"**Tu frase:** {question}")
        st.write(f"**Frase más afín (Doc {best_idx+1}):** {best_doc}")
        st.write(f"**Grado de afinidad:** {best_score:.3f}")

        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.markdown("### 🌐 Todas las afinidades")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))

        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

        st.markdown("### ✳️ Fragmentos en común")
        if matched:
            st.write("Palabras raíz presentes en ambas frases:", matched)
        else:
            st.write("_Ninguna coincidencia directa. A veces el sentido se pierde en la traducción._")

st.caption("“La máquina no comprende, pero se aproxima.”")
