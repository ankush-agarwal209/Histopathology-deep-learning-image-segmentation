"""Streamlit entry point for the CX22 cervical cytology analysis project."""

from pathlib import Path

import streamlit as st
from PIL import Image

from config import DEVICE
from evaluation import evaluate_full_test_set, evaluate_image, find_test_pairs
from inference import (
    create_overlay,
    image_to_bytes,
    load_model,
    predict_masks,
)
from morphology import calculate_morphology
from ui_components import show_metrics, show_morphology


st.set_page_config(
    page_title="Histopathology Image Analysis",
    page_icon="🔬",
    layout="wide",
)

st.title("Histopathology Image Analysis")
st.write("CX22 Cervical Cytology Segmentation and Morphological Reference Analysis")
st.caption(f"Inference device: {'GPU' if DEVICE.type == 'cuda' else 'CPU'}")

custom_tab, test_tab = st.tabs(["Custom Image Segmentation", "Test Dataset Evaluation"])


with custom_tab:
    st.header("Custom Image Segmentation")
    upload = st.file_uploader(
        "Upload a cervical cytology image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="custom_upload",
    )
    if upload is not None and st.button("Run Segmentation", type="primary", key="run_custom"):
        try:
            model = load_model()
            image = Image.open(upload).convert("RGB")
            resized, probabilities, cytoplasm, nucleus = predict_masks(model, image)
            cytoplasm_image = Image.fromarray(cytoplasm * 255)
            nucleus_image = Image.fromarray(nucleus * 255)
            morphology = calculate_morphology(cytoplasm, nucleus, probabilities)

            st.session_state["custom_result"] = {
                "image": resized,
                "cyto": cytoplasm_image,
                "nucleus": nucleus_image,
                "cyto_overlay": create_overlay(resized, cytoplasm_image, (0, 255, 0)),
                "nucleus_overlay": create_overlay(resized, nucleus_image, (255, 0, 0)),
                "morphology": morphology,
                "cyto_probability": float(probabilities[0].mean() * 100),
                "nucleus_probability": float(probabilities[1].mean() * 100),
            }
        except Exception as error:
            st.error(f"Segmentation failed: {error}")

    if "custom_result" in st.session_state:
        result = st.session_state["custom_result"]
        st.subheader("Predicted Masks")
        c1, c2, c3 = st.columns(3)
        c1.image(result["image"], caption="Original Image", width=220)
        c2.image(result["cyto"], caption="Predicted Cytoplasm", width=220)
        c3.image(result["nucleus"], caption="Predicted Nucleus", width=220)

        st.subheader("Segmentation Overlays")
        c1, c2 = st.columns(2)
        c1.image(result["cyto_overlay"], caption="Cytoplasm Overlay", width=260)
        c2.image(result["nucleus_overlay"], caption="Nucleus Overlay", width=260)

        show_morphology(result["morphology"])
        st.subheader("Download Segmentation Masks")
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download Cytoplasm Mask",
            image_to_bytes(result["cyto"]),
            "cytoplasm_mask.png",
            "image/png",
        )
        c2.download_button(
            "Download Nucleus Mask",
            image_to_bytes(result["nucleus"]),
            "nucleus_mask.png",
            "image/png",
        )


with test_tab:
    st.header("Test Dataset Evaluation")
    st.write("Select a held-out Test image. Its corresponding annotated masks load automatically.")
    try:
        pairs, unmatched = find_test_pairs()
        if not pairs:
            st.error("No matching test images and ground-truth masks were found.")
        else:
            st.caption(
                f"Matched test images: {len(pairs)}"
                + (f" | Images missing masks: {len(unmatched)}" if unmatched else "")
            )
            names = [pair[0].name for pair in pairs]
            selected_name = st.selectbox("Select a Test image", names, key="test_image_choice")
            selected_pair = pairs[names.index(selected_name)]

            if st.button("Evaluate Selected Image", type="primary", key="evaluate_one"):
                try:
                    model = load_model()
                    st.session_state["single_test_result"] = evaluate_image(model, *selected_pair)
                    st.session_state["single_test_name"] = selected_name
                except Exception as error:
                    st.error(f"Evaluation failed: {error}")

            if st.session_state.get("single_test_name") == selected_name and "single_test_result" in st.session_state:
                result = st.session_state["single_test_result"]
                st.subheader(f"Selected image: {selected_name}")

                st.markdown("**Cytoplasm comparison**")
                c1, c2, c3 = st.columns(3)
                c1.image(result["image"], caption="Original Image", width=220)
                c2.image(result["gt_cyto"] * 255, caption="Ground-Truth Cytoplasm", width=220)
                c3.image(result["pred_cyto"] * 255, caption="Predicted Cytoplasm", width=220)
                st.markdown("**Cytoplasm metrics**")
                show_metrics(result["cyto_metrics"])

                st.divider()
                st.markdown("**Nucleus comparison**")
                c1, c2, c3 = st.columns(3)
                c1.image(result["image"], caption="Original Image", width=220)
                c2.image(result["gt_nucleus"] * 255, caption="Ground-Truth Nucleus", width=220)
                c3.image(result["pred_nucleus"] * 255, caption="Predicted Nucleus", width=220)
                st.markdown("**Nucleus metrics**")
                show_metrics(result["nucleus_metrics"])
                show_morphology(result["morphology"])

                c1, c2 = st.columns(2)
                c1.download_button(
                    "Download Predicted Cytoplasm",
                    image_to_bytes(Image.fromarray(result["pred_cyto"] * 255)),
                    f"{Path(selected_name).stem}_cytoplasm_prediction.png",
                    "image/png",
                )
                c2.download_button(
                    "Download Predicted Nucleus",
                    image_to_bytes(Image.fromarray(result["pred_nucleus"] * 255)),
                    f"{Path(selected_name).stem}_nucleus_prediction.png",
                    "image/png",
                )

            st.divider()
            st.subheader("Overall Test Dataset Performance")
            if st.button("Evaluate Entire Test Dataset", key="evaluate_all"):
                try:
                    model = load_model()
                    progress = st.progress(0, text="Evaluating test dataset...")

                    def update_progress(fraction, current, total):
                        progress.progress(fraction, text=f"Evaluating {current}/{total}")

                    st.session_state["full_test_metrics"] = evaluate_full_test_set(
                        model, pairs, progress_callback=update_progress
                    )
                    progress.empty()
                    st.session_state["full_test_count"] = len(pairs)
                except Exception as error:
                    st.error(f"Full test evaluation failed: {error}")

            if "full_test_metrics" in st.session_state:
                averages = st.session_state["full_test_metrics"]
                st.caption(
                    f"Macro-average across {st.session_state['full_test_count']} matched test images; "
                    "each image contributes equally."
                )
                st.markdown("**Cytoplasm — test-set average**")
                show_metrics(averages["Cytoplasm"])
                st.markdown("**Nucleus — test-set average**")
                show_metrics(averages["Nucleus"])

    except Exception as error:
        st.error(f"Test dataset could not be loaded: {error}")
        st.code(
            "Cx22_Multi_Train/Test/\n"
            "├── images/\n"
            "└── gt_mask/\n"
            "    ├── Cytoplasm/Sem_seg/\n"
            "    └── Nucleus/Sem_seg/"
        )
