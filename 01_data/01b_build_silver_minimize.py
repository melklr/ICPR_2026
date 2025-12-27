import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CLASSE DE DÉTECTION TRL (règle MIN)
# ==========================================
class TRLDetectorInferrerMin:
    def __init__(self):
        # --- Definitions TRL (Sémantique) ---
        self.TRL_LONG = {
            1: "basic principles observed fundamental research theoretical studies lowest maturity scientific discovery idea generation",
            2: "technology concept formulated hypothesis speculative application potential use cases theoretical framework no validation",
            3: "experimental proof of concept critical function validation feasibility study lab experiment analytical studies early prototype",
            4: "technology validation in lab component testing prototype in controlled environment laboratory verification low-fidelity",
            5: "technology validation in relevant environment high-fidelity prototype simulation industrial validation risk reduction",
            6: "system demonstration in relevant environment pilot functional prototype near-operational performance testing",
            7: "system demonstration in operational environment field testing real world conditions deployment pre-commercial",
            8: "system complete and qualified certification standards production readiness commercial ready final system",
            9: "system proven in operational environment market deployment full scale commercialized sales mission ready"
        }
        self.trl_levels = np.arange(1, 10, dtype=int)
        self.ref_texts = [self.TRL_LONG[i] for i in self.trl_levels]

        # --- TF-IDF ---
        self.VEC = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            #stop_words="english",
            sublinear_tf=True,
        )
        self.VEC.fit(self.ref_texts)
        self.ref_vectors = self.VEC.transform(self.ref_texts)

        # --- Regex (Symbolique) ---
        self.REGEX_TRL = re.compile(
            r"\bTRL\s*[:=]?\s*([1-9])(?:\s*[-to]\s*([1-9]))?",
            re.IGNORECASE,
        )

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+", "", text)  # Pas d'URL
        return re.sub(r"\s+", " ", text).strip()

    def get_sentence_context(self, text, start_idx, end_idx):
        """Récupère la phrase entourant les indices (délimité par des points)"""
        s_bound = text.rfind(".", 0, start_idx) + 1
        e_bound = text.find(".", end_idx)
        if e_bound == -1:
            e_bound = len(text)

        snippet = text[s_bound:e_bound].strip()
        if len(snippet) > 300:
            snippet = (
                "..."
                + text[max(0, start_idx - 50) : min(len(text), end_idx + 50)]
                + "..."
            )
        return snippet

    def detect_smart(self, text):
        """
        Retourne: (Label, Source, Confiance, Justification)
        RÈGLE MINIMIZE :
        - si plusieurs TRL sont trouvés dans le texte (ex: 'TRL 3', 'TRL 7', 'TRL 4-6'),
          on retient la PLUS PETITE valeur possible.
        - pour un range '4-6', on prend 4 (min du range).
        """
        text_clean = self.clean_text(text)
        if len(text_clean) < 10:
            return None, None, 0.0, ""

        # 1. REGEX (L'Expert) - RÈGLE MIN
        matches = list(self.REGEX_TRL.finditer(text_clean))

        if matches:
            best_trl = 10  # plus grand que 9 pour pouvoir minimiser
            best_justif = ""

            for match in matches:
                try:
                    val1 = int(match.group(1))
                    val2 = int(match.group(2)) if match.group(2) else val1
                    # Pour un range (ex: "4-6"), on prend le MINIMUM
                    current_trl = min(val1, val2)

                    # On garde la valeur la plus FAIBLE trouvée dans tout le texte
                    if current_trl < best_trl:
                        best_trl = current_trl
                        start_idx, end_idx = match.span()
                        best_justif = self.get_sentence_context(
                            text_clean, start_idx, end_idx
                        )
                except ValueError:
                    continue

            if 1 <= best_trl <= 9:
                return best_trl, "regex_min", 1.0, best_justif

        # 2. Fallback INFÉRENCE (Le Devin) - identique
        try:
            sentences = re.split(r"(?<=[.!?])\s+", text_clean)
            valid_sentences = [s for s in sentences if len(s) > 20]
            if not valid_sentences:
                valid_sentences = [text_clean]

            vec_sentences = self.VEC.transform(valid_sentences)
            sims = cosine_similarity(vec_sentences, self.ref_vectors)

            max_idx_flat = np.argmax(sims)
            sent_idx, trl_idx = np.unravel_index(max_idx_flat, sims.shape)
            best_score = sims[sent_idx, trl_idx]

            if best_score > 0.15:
                found_trl = self.trl_levels[trl_idx]
                justif = valid_sentences[sent_idx].strip()
                return found_trl, "inference", float(best_score), justif

        except Exception:
            pass

        return None, None, 0.0, ""

# ==========================================
# 2. PIPELINE PRINCIPAL
# ==========================================
def main():
    PROJECTS_FILE = r"C:\Users\Melusine\.venv\project.csv"

    print(f"--- Chargement massif : {PROJECTS_FILE} ---")
    try:
        df = pd.read_csv(PROJECTS_FILE, sep=";", on_bad_lines="skip", low_memory=False)
        if len(df.columns) < 3:
            df = pd.read_csv(PROJECTS_FILE, sep=",", on_bad_lines="skip", low_memory=False)
    except Exception as e:
        print(f"Erreur lecture : {e}")
        return

    print(f"Total Projets bruts : {len(df)}")

    # Standardisation colonnes
    df.columns = [c.lower() for c in df.columns]

    # Préparation Texte (Titre + Objectif)
    df["text"] = df["title"].fillna("") + ". " + df["objective"].fillna("")

    # Nettoyage : on vire les textes trop courts (< 50 chars)
    df = df[df["text"].str.len() > 50].copy()
    print(f"Projets avec texte valide : {len(df)}")

    # --- DÉTECTION ---
    print("Lancement de l'analyse Smart (Regex MIN + Inference)...")
    detector = TRLDetectorInferrerMin()

    results = df["text"].apply(lambda x: detector.detect_smart(x))

    df["label"] = [r[0] for r in results]
    df["source"] = [r[1] for r in results]
    df["conf"] = [r[2] for r in results]
    df["justification"] = [r[3] for r in results]

    # --- FILTRAGE SILVER ---
    df_silver = df.dropna(subset=["label"]).copy()
    df_silver["label"] = df_silver["label"].astype(int)

    # SAUVEGARDE AVEC NOM SPÉCIFIQUE
    output_file = "minimized_silver_dataset.csv"
    cols_to_save = ["id", "rcn", "text", "label", "source", "conf", "justification"]
    final_cols = [c for c in cols_to_save if c in df_silver.columns]

    df_silver[final_cols].to_csv(output_file, index=False)

    print("-" * 30)
    print(f"✅ SUCCÈS : Silver Dataset (règle MIN) généré : {len(df_silver)} lignes")
    print(f"Sauvegardé sous : {output_file}")

    print("\nDistribution des Sources :")
    print(df_silver["source"].value_counts())

    mean_conf_inf = df_silver[df_silver["source"] == "inference"]["conf"].mean()
    print(f"\n🔍 Moyenne de confiance (Inférence uniquement) : {mean_conf_inf:.4f}")

    print("\nDistribution des TRL :")
    print(df_silver["label"].value_counts().sort_index())
    print("-" * 30)


if __name__ == "__main__":
    main()