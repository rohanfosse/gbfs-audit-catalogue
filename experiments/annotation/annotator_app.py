"""GBFS Audit Catalogue -- Outil d'annotation humaine.

Validation ground-truth sur un echantillon stratifie de 175 stations.
Deux annotateurs independants evaluent chaque station ; la fiabilite
inter-annotateurs est calculee ensuite via compute_reliability.py.

Usage :
    python -m streamlit run experiments/annotation/annotator_app.py
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PATH = Path(__file__).resolve().parent / "sample_200.csv"
LABELS_DIR = Path(__file__).resolve().parent
CATALOGUE_PATH = REPO_ROOT / "catalogue" / "stations_gold_standard_final.parquet"

STRATUM_COLORS = {
    "clean_docked": "#27ae60",
    "A1_carsharing": "#8e44ad",
    "A2_placeholder": "#d35400",
    "A3_freefloating": "#2980b9",
    "A4_agree_flag": "#c0392b",
    "A4_discordant_legacy": "#e74c3c",
    "A4_discordant_composite": "#2471a3",
    "A6_zero_capacity": "#16a085",
    "A7_null_capacity": "#f39c12",
    "A3_boundary": "#7f8c8d",
}

STRATUM_GUIDELINES = {
    "clean_docked": (
        "Cette station n'a declenche aucun flag et a ete classee en haute "
        "confiance. Votre role est de verifier qu'il ne s'agit pas d'un "
        "faux negatif : la station est-elle reellement un dock velo "
        "fonctionnel aux coordonnees indiquees ?"
    ),
    "A1_carsharing": (
        "Le pipeline a detecte un systeme d'autopartage (voitures) "
        "publie sous le schema GBFS velo. Verifiez sur la carte et "
        "Street View : voyez-vous des bornes de velos ou des places "
        "de stationnement automobile ?"
    ),
    "A2_placeholder": (
        "Toutes les stations de ce systeme declarent exactement la "
        "meme capacite (ex : 100 partout). Verifiez : est-ce un vrai "
        "nombre de bornes physiques, ou un placeholder insere par "
        "l'operateur ?"
    ),
    "A3_freefloating": (
        "Cette station est identifiee comme un ancrage virtuel de "
        "flotte free-floating. Verifiez sur la carte : y a-t-il des "
        "bornes physiques visibles, ou s'agit-il simplement d'un "
        "point GPS ou un vehicule a ete gare ?"
    ),
    "A4_agree_flag": (
        "Les DEUX detecteurs (centroide legacy et composite topologique) "
        "considerent cette station comme un outlier geospatial. "
        "Verifiez : la station est-elle reellement isolee du reste "
        "du reseau, ou appartient-elle a une extension legitime "
        "(gare, campus, zone d'activite) ?"
    ),
    "A4_discordant_legacy": (
        "C'est la strate la plus importante de l'annotation. Le "
        "centroide legacy flag cette station comme outlier, mais le "
        "detecteur composite (HDBSCAN + spectral) la considere comme "
        "normale. Votre verdict determine directement si les 8 005 "
        "stations discordantes sont de vrais faux positifs du legacy."
    ),
    "A4_discordant_composite": (
        "Le detecteur composite flag cette station, mais le centroide "
        "legacy ne la detecte pas. C'est une nouvelle detection. "
        "Verifiez : la station est-elle reellement problematique ?"
    ),
    "A6_zero_capacity": (
        "La station declare une capacite de zero bornes. Verifiez sur "
        "la carte : la station existe-t-elle physiquement ? A-t-elle "
        "ete desinstallee ou est-elle en travaux ?"
    ),
    "A7_null_capacity": (
        "La station declare capacity = NaN (champ vide). C'est le "
        "pattern typique de Dott et Bird. Verifiez simplement que "
        "la station existe bien a ces coordonnees."
    ),
    "A3_boundary": (
        "Le ratio de capacite de ce systeme est entre 2 et 5 -- la zone "
        "grise du seuil A3. Verifiez : s'agit-il d'un systeme de "
        "bornes physiques avec des capacites heterogenes, ou d'un "
        "systeme free-floating avec un estimateur de profil ?"
    ),
}

FLAG_LABELS = {
    "A1": "Inclusion hors domaine (autopartage)",
    "A2": "Capacite placeholder (constante sur tout le systeme)",
    "A3": "Sur-capacite structurelle (ancrage free-floating)",
    "A4": "Outlier geospatial (detecteur topologique composite)",
    "A5": "Hors perimetre (surface > 50 000 km2)",
    "A6": "Dock a zero capacite (avertissement semantique)",
    "A7": "Champ capacite nul / NaN (avertissement semantique)",
}


# =====================================================================
# Configuration de la page
# =====================================================================

st.set_page_config(
    page_title="Annotation GBFS -- Ground Truth",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 0.6rem; max-width: 1400px; }
    section[data-testid="stSidebar"] > div { padding-top: 0.8rem; }
    .stratum-badge {
        display: inline-block; padding: 0.12rem 0.5rem;
        border-radius: 3px; font-size: 0.73rem; font-weight: 700;
        color: white; letter-spacing: 0.02em;
    }
    .guideline-box {
        background: #f0f4f8; border-left: 3px solid #1A6FBF;
        padding: 0.55rem 0.85rem; border-radius: 0 4px 4px 0;
        font-size: 0.84rem; margin-bottom: 0.5rem; line-height: 1.45;
    }
    .links-bar {
        display: flex; gap: 0.6rem; flex-wrap: wrap;
        margin-top: 0.3rem; margin-bottom: 0.2rem;
    }
    .links-bar a {
        display: inline-block; padding: 0.25rem 0.65rem;
        border-radius: 4px; font-size: 0.78rem; font-weight: 600;
        text-decoration: none; border: 1px solid #ccc; color: #333;
    }
    .links-bar a:hover { background: #f0f0f0; }
    .links-bar a.primary { background: #1A6FBF; color: white; border-color: #1A6FBF; }
    .links-bar a.primary:hover { background: #155a8a; }
    .flag-row { font-size: 0.82rem; margin-bottom: 0.15rem; }
    .flag-on { color: #c0392b; font-weight: 700; }
    .flag-off { color: #bdc3c7; }
    .meta-label { font-size: 0.76rem; color: #777; margin-top: 0.25rem; }
    .meta-value { font-size: 0.88rem; font-weight: 600; color: #222; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Session state
# =====================================================================

if "start_times" not in st.session_state:
    st.session_state.start_times = {}


# =====================================================================
# Sidebar : identite, protocole, progression
# =====================================================================

with st.sidebar:
    st.markdown("## Session d'annotation")

    annotator_name = st.text_input(
        "Votre nom",
        value="",
        placeholder="ex : Rohan",
        key="annotator_name",
    )
    if not annotator_name.strip():
        st.warning("Entrez votre nom pour commencer.")
        st.stop()

    annotator_id = annotator_name.strip().lower().replace(" ", "_")
    labels_path = LABELS_DIR / f"labels_{annotator_id}.csv"

    st.markdown("---")
    st.markdown("### Protocole")
    st.markdown(
        "Pour chaque station, vous disposez de :\n\n"
        "- Une **carte OpenStreetMap** centree sur la station "
        "(couche CycleMap, pistes cyclables visibles)\n"
        "- Des **liens directs** vers Street View, CyclOSM et "
        "la page OSM de la zone\n"
        "- Les **metadonnees GBFS** et le **verdict du pipeline**\n"
        "- Une **consigne specifique** a la strate de cette station\n\n"
        "Repondez aux 5 questions, puis cliquez sur Enregistrer."
    )
    st.markdown(
        "<div class='guideline-box'>"
        "<b>Principe fondamental :</b> vous etes la verite terrain. "
        "Si le pipeline signale une anomalie mais que vous constatez "
        "une station velo legitime sur Street View ou sur la carte, "
        "repondez <b>faux positif du pipeline</b>. Inversement, si "
        "le pipeline ne signale rien mais que la station vous semble "
        "suspecte, notez-le dans les remarques."
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# Chargement des donnees
# =====================================================================

if not SAMPLE_PATH.exists():
    st.error(f"Fichier echantillon introuvable : {SAMPLE_PATH}")
    st.stop()

sample = pd.read_csv(SAMPLE_PATH)

if labels_path.exists():
    labels_df = pd.read_csv(labels_path)
    done_keys = set(
        labels_df.apply(lambda r: f"{r['system_id']}|{r['station_id']}", axis=1)
    )
else:
    labels_df = pd.DataFrame()
    done_keys = set()

sample["_key"] = sample.apply(
    lambda r: f"{r['system_id']}|{r['station_id']}", axis=1
)
remaining = sample[~sample["_key"].isin(done_keys)]
n_total = len(sample)
n_done = n_total - len(remaining)

if CATALOGUE_PATH.exists():
    @st.cache_data
    def _load_catalogue():
        return pd.read_parquet(CATALOGUE_PATH)
    full_cat = _load_catalogue()
else:
    full_cat = None


# =====================================================================
# Sidebar : progression
# =====================================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("### Progression")
    st.progress(n_done / n_total if n_total > 0 else 0.0)
    st.markdown(f"**{n_done}** / **{n_total}** stations annotees")

    if n_done > 0 and labels_path.exists():
        tmp = pd.read_csv(labels_path)
        non_skip = tmp[tmp["Q5_verdict"] != "skipped"]
        if "duration_s" in non_skip.columns and len(non_skip) > 0:
            med = non_skip["duration_s"].median()
            rest_min = med * len(remaining) / 60
            st.caption(
                f"Temps median : {med:.0f}s par station -- "
                f"Temps restant estime : ~{rest_min:.0f} min"
            )

    st.markdown("#### Par strate")
    by_str = sample.copy()
    by_str["done"] = by_str["_key"].isin(done_keys)
    prog = by_str.groupby("stratum").agg(
        fait=("done", "sum"), total=("_key", "count")
    ).reset_index()
    for _, r in prog.iterrows():
        color = STRATUM_COLORS.get(r["stratum"], "#999")
        pct = int(100 * r["fait"] / r["total"]) if r["total"] > 0 else 0
        bar = "|" * (pct // 10) + "." * (10 - pct // 10)
        st.markdown(
            f"<span class='stratum-badge' style='background:{color};'>"
            f"{r['stratum']}</span> "
            f"<code>{bar}</code> {r['fait']}/{r['total']}",
            unsafe_allow_html=True,
        )


# =====================================================================
# Ecran de fin
# =====================================================================

if len(remaining) == 0:
    st.title("Annotation terminee")
    st.success(
        f"L'ensemble des {n_total} stations a ete annote par "
        f"{annotator_name}. Les labels sont sauvegardes dans "
        f"`{labels_path.name}`."
    )
    st.markdown(
        "**Etape suivante** : demandez au second annotateur de lancer "
        "sa propre session, puis calculez la fiabilite inter-annotateurs :\n\n"
        "```bash\n"
        "python -m experiments.annotation.compute_reliability \\\n"
        f"    --labels1 {labels_path.name} \\\n"
        "    --labels2 labels_<autre>.csv \\\n"
        "    --output reliability_report.json\n"
        "```"
    )
    st.stop()


# =====================================================================
# Station courante
# =====================================================================

row = remaining.iloc[0]
station_key = row["_key"]

if station_key not in st.session_state.start_times:
    st.session_state.start_times[station_key] = time.time()

lat = float(row["lat"]) if pd.notna(row.get("lat")) else None
lon = float(row["lon"]) if pd.notna(row.get("lon")) else None
stratum = row.get("stratum", "unknown")
stratum_color = STRATUM_COLORS.get(stratum, "#999")
guideline = STRATUM_GUIDELINES.get(stratum, "")


# =====================================================================
# En-tete
# =====================================================================

st.markdown(
    f"<div style='display:flex; align-items:center; gap:0.6rem; "
    f"margin-bottom:0.2rem;'>"
    f"<span style='font-size:1.15rem; font-weight:700;'>"
    f"Station {n_done + 1} sur {n_total}</span>"
    f"<span class='stratum-badge' style='background:{stratum_color};'>"
    f"{stratum}</span>"
    f"<span style='color:#888; font-size:0.8rem;'>"
    f"{row.get('system_id', '')} / {row.get('station_id', '')}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='guideline-box'>{guideline}</div>",
    unsafe_allow_html=True,
)


# =====================================================================
# Deux colonnes : Carte | Metadonnees + Flags
# =====================================================================

col_left, col_right = st.columns([3, 2])

with col_left:
    if lat is not None and lon is not None:
        delta = 0.005
        osm_embed = (
            f"https://www.openstreetmap.org/export/embed.html?"
            f"bbox={lon - delta},{lat - delta},{lon + delta},{lat + delta}"
            f"&layer=cyclemap&marker={lat},{lon}"
        )
        st.components.v1.iframe(osm_embed, height=420, scrolling=False)

        osm_url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}&layers=C"
        cyclosm_url = f"https://www.cyclosm.org/#map=17/{lat}/{lon}/cyclosm"
        streetview_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        gmaps_url = f"https://www.google.com/maps/@{lat},{lon},18z"
        osm_query_url = f"https://www.openstreetmap.org/query?lat={lat}&lon={lon}#map=18/{lat}/{lon}"

        st.markdown(
            f"<div class='links-bar'>"
            f"<a class='primary' href='{streetview_url}' target='_blank'>Street View (verification terrain)</a>"
            f"<a href='{osm_url}' target='_blank'>OSM plein ecran</a>"
            f"<a href='{cyclosm_url}' target='_blank'>CyclOSM (pistes cyclables)</a>"
            f"<a href='{osm_query_url}' target='_blank'>Identifier les objets OSM</a>"
            f"<a href='{gmaps_url}' target='_blank'>Google Maps</a>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "La carte affiche la couche CycleMap (OpenCycleMap) avec les "
            "amenagements cyclables. Le marqueur rouge indique la station "
            "a evaluer. Utilisez Street View pour verifier la presence "
            "physique de bornes velo."
        )
    else:
        st.warning("Coordonnees indisponibles pour cette station.")


with col_right:
    st.markdown("**Metadonnees GBFS**")

    meta_pairs = [
        ("Operateur", row.get("operator_name")),
        ("Ville", row.get("city")),
        ("Type declare", row.get("station_type")),
        ("Capacite brute", row.get("capacity")),
        ("Confiance audit", row.get("audit_confidence")),
        ("Coordonnees", f"{lat:.5f}, {lon:.5f}" if lat and lon else None),
    ]
    for label, val in meta_pairs:
        v = str(val) if pd.notna(val) else "---"
        st.markdown(
            f"<div><span class='meta-label'>{label}</span><br>"
            f"<span class='meta-value'>{v}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**Verdict du pipeline (flags A1--A7)**")
    any_flag = False
    for i in range(1, 8):
        col_name = f"flag_A{i}"
        is_on = bool(row.get(col_name, False)) if col_name in row.index else False
        if is_on:
            any_flag = True
        css = "flag-on" if is_on else "flag-off"
        marker = "ACTIF" if is_on else "---"
        desc = FLAG_LABELS.get(f"A{i}", "")
        st.markdown(
            f"<div class='flag-row'>"
            f"<span class='{css}'>A{i} [{marker}]</span> "
            f"<span style='color:#999; font-size:0.76rem;'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not any_flag:
        st.markdown(
            "<div style='color:#27ae60; font-size:0.84rem; "
            "font-weight:600; margin-top:0.3rem;'>"
            "Aucun flag declenche -- le pipeline considere cette station "
            "comme propre.</div>",
            unsafe_allow_html=True,
        )


# =====================================================================
# Formulaire d'annotation
# =====================================================================

st.markdown("---")
st.markdown("### Votre evaluation")

qa, qb = st.columns(2)

with qa:
    a_q1 = st.radio(
        "**Q1.** Cette station fait-elle partie d'un reseau de velos en libre-service ?",
        ["oui", "non", "indetermine"],
        index=None,
        key="q1",
        help=(
            "Repondez 'oui' si le systeme concerne des velos ou VAE. "
            "Repondez 'non' s'il s'agit exclusivement de voitures "
            "(autopartage) ou de trottinettes sans aucun velo. "
            "Repondez 'indetermine' si vous ne pouvez pas trancher."
        ),
    )

    a_q3 = st.radio(
        "**Q3.** Cette station existe-t-elle physiquement a ces coordonnees ?",
        ["oui", "non", "indetermine"],
        index=None,
        key="q3",
        help=(
            "Consultez Street View ou l'imagerie satellite. Cherchez "
            "des bornes, des arceaux, un totem ou de la signaletique "
            "velo. Si rien n'est visible et que l'imagerie est recente, "
            "repondez 'non'."
        ),
    )

    a_q5 = st.radio(
        "**Q5.** Verdict global :",
        [
            "propre",
            "anomalie confirmee",
            "faux positif du pipeline",
            "indetermine",
        ],
        index=None,
        key="q5",
        help=(
            "Propre : la station est legitime et correctement classee. "
            "Anomalie confirmee : le pipeline a raison de la signaler. "
            "Faux positif : le pipeline la signale a tort. "
            "Indetermine : impossible de trancher avec les elements disponibles."
        ),
    )

with qb:
    a_q2 = st.radio(
        "**Q2.** La capacite declaree correspond-elle a un nombre physique de bornes ?",
        ["oui", "non", "NaN", "indetermine"],
        index=None,
        key="q2",
        help=(
            "Oui : la valeur (ex : 20) correspond a un nombre reel "
            "d'emplacements physiques. Non : c'est un placeholder "
            "(ex : 100 partout, ou un estimateur statistique). "
            "NaN : le champ est vide dans le flux GBFS."
        ),
    )

    a_q4 = st.radio(
        "**Q4.** Ces coordonnees sont-elles dans le perimetre raisonnable du reseau ?",
        ["oui", "non"],
        index=None,
        key="q4",
        help=(
            "Regardez la carte : la station est-elle coherente avec "
            "le reste du reseau (les autres stations du meme systeme "
            "sont visibles sur la couche CycleMap), ou est-elle "
            "completement isolee / dans un autre pays ?"
        ),
    )

    notes = st.text_area(
        "Remarques (facultatif)",
        value="",
        placeholder=(
            "Observations utiles : date de l'imagerie Street View, "
            "station en travaux, bornes retirees, doute sur le type "
            "de vehicule..."
        ),
        key="notes",
        height=100,
    )


# =====================================================================
# Boutons
# =====================================================================

all_answered = all([a_q1, a_q2, a_q3, a_q4, a_q5])

btn_save, btn_skip = st.columns([2, 1])

with btn_save:
    if st.button(
        "Enregistrer et passer a la suivante",
        type="primary",
        disabled=not all_answered,
        use_container_width=True,
    ):
        elapsed = time.time() - st.session_state.start_times.get(
            station_key, time.time()
        )
        new_row = {
            "system_id": row["system_id"],
            "station_id": row["station_id"],
            "stratum": stratum,
            "lat": lat,
            "lon": lon,
            "Q1_is_bikeshare": a_q1,
            "Q2_capacity_physical": a_q2,
            "Q3_exists_at_coords": a_q3,
            "Q4_within_perimeter": a_q4,
            "Q5_verdict": a_q5,
            "annotator": annotator_id,
            "notes": notes,
            "duration_s": round(elapsed, 1),
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        if labels_path.exists():
            existing = pd.read_csv(labels_path)
            updated = pd.concat(
                [existing, pd.DataFrame([new_row])], ignore_index=True
            )
        else:
            updated = pd.DataFrame([new_row])
        updated.to_csv(labels_path, index=False)
        st.session_state.start_times.pop(station_key, None)
        st.rerun()

with btn_skip:
    if st.button("Passer cette station", use_container_width=True):
        skip_row = {
            "system_id": row["system_id"],
            "station_id": row["station_id"],
            "stratum": stratum,
            "lat": lat,
            "lon": lon,
            "Q1_is_bikeshare": "passe",
            "Q2_capacity_physical": "passe",
            "Q3_exists_at_coords": "passe",
            "Q4_within_perimeter": "passe",
            "Q5_verdict": "skipped",
            "annotator": annotator_id,
            "notes": "",
            "duration_s": 0,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        if labels_path.exists():
            existing = pd.read_csv(labels_path)
            updated = pd.concat(
                [existing, pd.DataFrame([skip_row])], ignore_index=True
            )
        else:
            updated = pd.DataFrame([skip_row])
        updated.to_csv(labels_path, index=False)
        st.session_state.start_times.pop(station_key, None)
        st.rerun()

if not all_answered:
    st.caption(
        "Repondez aux cinq questions ci-dessus pour activer "
        "le bouton Enregistrer."
    )
