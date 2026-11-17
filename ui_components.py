"""Reusable Streamlit display components."""

import numpy as np
import streamlit as st

from morphology import reference_similarity


def show_metrics(metrics):
    names = list(metrics.keys())
    columns = st.columns(4)

    for index, name in enumerate(names):
        columns[index % 4].metric(
            name,
            f"{metrics[name] * 100:.2f}%"
        )


def show_mask(mask, caption):
    st.image(
        np.asarray(mask, dtype=np.uint8) * 255,
        caption=caption,
        width=220
    )


def show_morphology(morphology):

    cyto_shape = morphology["cyto_shape"]
    nucleus_shape = morphology["nucleus_shape"]

    st.subheader("Cellular Morphological Measurements")

    # Area measurements
    st.markdown("**Area Measurements**")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Nuclear Area",
        f"{morphology['nucleus_area']:,} px²"
    )

    c2.metric(
        "Cytoplasmic Area",
        f"{morphology['cytoplasm_area']:,} px²"
    )

    c3.metric(
        "N:C Area Ratio",
        f"{morphology['nc_ratio']:.4f}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Nuclear Area Coverage",
        f"{morphology['nucleus_percentage']:.2f}%"
    )

    c2.metric(
        "Cytoplasmic Area Coverage",
        f"{morphology['cytoplasm_percentage']:.2f}%"
    )

    c3.metric(
        "Total Segmented Coverage",
        f"{morphology['total_segmented_percentage']:.2f}%"
    )

    # Nuclear shape measurements
    st.markdown("**Nuclear Shape Measurements**")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Nuclear Perimeter",
        f"{nucleus_shape['perimeter_pixels']:.1f} px"
    )

    c2.metric(
        "Equivalent Diameter",
        f"{nucleus_shape['equivalent_diameter_pixels']:.1f} px"
    )

    c3.metric(
        "Nuclear Circularity",
        f"{nucleus_shape['circularity']:.4f}"
    )

    c4.metric(
        "Nuclear Compactness",
        f"{nucleus_shape['compactness']:.4f}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Nuclear Eccentricity",
        f"{nucleus_shape['eccentricity']:.4f}"
    )

    c2.metric(
        "Nuclear Solidity",
        f"{nucleus_shape['solidity']:.4f}"
    )

    c3.metric(
        "Nucleus Connected Regions",
        str(nucleus_shape["connected_regions"])
    )

    c4.metric(
        "Cytoplasm Connected Regions",
        str(cyto_shape["connected_regions"])
    )

    # Model probabilities
    if "cytoplasm_mean_probability" in morphology:

        st.markdown("**Model Probability Statistics**")

        c1, c2 = st.columns(2)

        c1.metric(
            "Mean Cytoplasm Probability",
            f"{morphology['cytoplasm_mean_probability']:.2f}%"
        )

        c2.metric(
            "Mean Nucleus Probability",
            f"{morphology['nucleus_mean_probability']:.2f}%"
        )

    # Research comparison
    st.subheader("Research-Reference Morphological Similarity")

    scores = reference_similarity(morphology["nc_ratio"])

    best_category = max(scores, key=scores.get)

    st.metric(
        "Closest N:C Reference Group",
        best_category,
        f"{scores[best_category]:.1f}% Relative Similarity"
    )

    full_forms = {
        "NILM": "Negative for Intraepithelial Lesion or Malignancy",
        "ASC-US": "Atypical Squamous Cells of Undetermined Significance",
        "LSIL": "Low-grade Squamous Intraepithelial Lesion",
        "ASC-H": "Atypical Squamous Cells — cannot exclude HSIL",
        "HSIL": "High-grade Squamous Intraepithelial Lesion",
        "SCC": "Squamous Cell Carcinoma"
    }

    table_data = []

    for category, score in scores.items():

        table_data.append({
            "Abbreviation": category,
            "Full Form": full_forms[category],
            "Relative Similarity": f"{score:.2f}%"
        })

    st.table(table_data)

    st.caption(
        "Research Reference: Evaluation of the Cytonucleomorphometric "
        "Parameters for Cases Diagnosed as Squamous Cell Abnormality "
        "on Conventional Cervico-Vaginal Pap Smears."
    )