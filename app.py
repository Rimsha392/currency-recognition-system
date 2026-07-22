"""
app.py

Purpose
-------
Streamlit web application for the Currency Recognition System.

This is a VISUAL REDESIGN ONLY. Every piece of ML logic -- loading a
model, loading a class mapping, preprocessing an image, running inference
-- is still imported directly from predict.py, completely unchanged. The
multi-currency architecture (utils/currency_registry.py: active vs.
coming-soon currencies, the single-vs-multi-currency branch) is also
unchanged. Only how results are PRESENTED has changed.

Run with:
    streamlit run app.py

Design system (see .streamlit/config.toml for the native theme layer)
-----------------------------------------------------------------------
Grounded in the actual subject -- banknotes -- rather than a generic
SaaS look:
  - Color: deep emerald (banknote security-ink green) as the brand color,
    warm ivory "paper" background, antique gold hairline accents. Each
    denomination also gets its own accent color loosely reflecting that
    note's real-world dominant color (Rs.10 green, Rs.50 violet, etc.) --
    this is information (which note is which), not decoration.
  - Type: Fraunces (a characterful serif) used sparingly for the hero
    title and the large denomination numeral only; Inter for all other
    UI text; a monospace face for confidence percentages, giving them a
    "measuring instrument" feel fitting for a recognition tool.
  - Signature element: a large, faint "₨" watermark glyph behind the hero
    title -- real banknotes carry watermarks, so this is a detail pulled
    from the subject itself rather than a decorative flourish.

Where native vs. custom is used, and why
-------------------------------------------
- st.progress's fill color is themed globally via .streamlit/config.toml
  (primaryColor) -- the robust, version-safe way to recolor a native
  widget, instead of targeting Streamlit's internal CSS classes directly.
- st.container(border=True), st.metric, st.selectbox, st.file_uploader,
  st.spinner are all used natively and unmodified in behavior.
- Custom HTML/CSS is used ONLY where native components genuinely can't
  express the design: the hero watermark, pill-style currency badges, the
  large colored denomination numeral, and the per-denomination-colored
  mini bars in the Top 3 list (st.progress only supports one global
  theme color, not an arbitrary color per instance).
"""

from pathlib import Path

import streamlit as st

from predict import (
    load_index_to_class,
    load_trained_model,
    predict_denomination,
    preprocess_image,
)
from utils.currency_registry import get_active_currencies, get_coming_soon_currencies

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Currency Recognition System",
    page_icon="💰",
    layout="wide",
)

# ----------------------------------------------------------------------
# Denomination accent colors -- a presentation-only mapping.
# ----------------------------------------------------------------------
# Loosely reflects each note's real-world dominant color. This lives in
# app.py (not predict.py) because it is purely cosmetic and has nothing
# to do with the model -- a good example of keeping presentation concerns
# out of the ML pipeline. DEFAULT_ACCENT is used for any label not in this
# map, so a future denomination or currency never breaks the UI.
DEFAULT_ACCENT = "#1F5C4A"  # brand emerald, used as a safe fallback
DENOMINATION_ACCENTS = {
    "10": "#2E7D32",    # green
    "20": "#C8622A",    # burnt orange
    "50": "#6A3C82",    # violet
    "100": "#2E6B4F",   # teal-green
    "500": "#7A3E73",   # mauve
    "1000": "#5B5A63",  # charcoal-grey
    "5000": "#8A5A2B",  # brown-gold
}


def get_accent(label: str) -> str:
    return DENOMINATION_ACCENTS.get(label, DEFAULT_ACCENT)


# ----------------------------------------------------------------------
# Design system: fonts + custom CSS
# ----------------------------------------------------------------------
# Only what native Streamlit components cannot express is here: web font
# imports, the hero watermark, pill badges, the large denomination
# numeral, and per-item colored mini bars for the Top 3 list. Everything
# else (cards, the main confidence bar, metrics) stays native.
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

        .block-container {
            max-width: 960px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            margin: 0 auto;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ---------------- Hero ---------------- */
        .hero {
            position: relative;
            text-align: center;
            overflow: hidden;
            padding: 1.75rem 1rem 1.25rem 1rem;
            border-bottom: 1px solid #E4DCC4;
            margin-bottom: 1.5rem;
            animation: fadeInUp 0.6s ease-out;
        }
        .hero-watermark {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-family: 'Fraunces', serif;
            font-size: 9rem;
            font-weight: 600;
            color: #1F5C4A;
            opacity: 0.06;
            pointer-events: none;
            user-select: none;
            z-index: 0;
        }
        .hero-title {
            position: relative;
            z-index: 1;
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 2.5rem;
            letter-spacing: -0.01em;
            margin-bottom: 0.25rem;
            color: #1C1B17;
        }
        .hero-subtitle {
            position: relative;
            z-index: 1;
            font-size: 1.05rem;
            color: #6b6a63;
        }

        /* ---------------- Currency pills ---------------- */
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
            margin: 1.1rem 0 0.4rem 0;
        }
        .pill-active {
            background: #1F5C4A;
            color: #FAF8F2;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 500;
            border-bottom: 2px solid #B08D3E;
        }
        .pill-soon {
            background: transparent;
            color: #9a9890;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            border: 1px dashed #d8d4c4;
        }
        .pill-label {
            text-align: center;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #a39d8a;
            margin-top: 0.6rem;
        }

        /* ---------------- Prediction card ---------------- */
        .eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9a9890;
            margin-bottom: 0.2rem;
        }
        .denom-numeral {
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 2.6rem;
            line-height: 1.1;
            margin: 0.1rem 0 0.3rem 0;
        }
        .denom-code {
            font-size: 0.95rem;
            color: #6b6a63;
            margin-bottom: 0.8rem;
        }

        /* Monospace treatment for confidence figures -- gives numbers a
           "measuring instrument" feel. Targets Streamlit's own metric
           value element; if this data-testid ever changes in a future
           Streamlit release, the value simply falls back to the default
           font -- a harmless, graceful degradation. */
        [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace;
        }

        /* ---------------- Top-3 ranked rows ---------------- */
        .rank-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 0.55rem 0;
        }
        .rank-badge {
            flex-shrink: 0;
            width: 1.6rem;
            height: 1.6rem;
            border-radius: 50%;
            background: #F1ECDE;
            color: #6b6a63;
            font-size: 0.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .rank-body {
            flex-grow: 1;
        }
        .rank-top-line {
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }
        .rank-percent {
            font-family: 'IBM Plex Mono', monospace;
            color: #6b6a63;
        }
        .rank-bar-track {
            width: 100%;
            height: 6px;
            border-radius: 999px;
            background: #EDE8D9;
            overflow: hidden;
        }
        .rank-bar-fill {
            height: 100%;
            border-radius: 999px;
        }

        /* ---------------- Footer ---------------- */
        .footer-rule {
            border: none;
            border-top: 1px solid #E4DCC4;
            margin: 2.5rem 0 1rem 0;
        }
        .footer-text {
            text-align: center;
            font-size: 0.88rem;
            color: #9a9890;
        }
        .footer-version {
            text-align: center;
            font-size: 0.78rem;
            color: #c2beac;
            margin-top: 0.2rem;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (prefers-reduced-motion: reduce) {
            .hero { animation: none; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero section
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-watermark">Rs</div>
        <div class="hero-title">💰 Currency Recognition System</div>
        <div class="hero-subtitle">AI-powered banknote recognition</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Supported currencies -- rendered as pill badges, straight from the
# registry. Never hardcoded: add a currency to utils/currency_registry.py
# and it appears here automatically, in the correct pill style.
# ----------------------------------------------------------------------
active_currencies = get_active_currencies()
coming_soon_currencies = get_coming_soon_currencies()

pills_html = '<div class="pill-label">Supported currencies</div><div class="pill-row">'
for currency in active_currencies:
    pills_html += f'<span class="pill-active">{currency.display_label}</span>'
pills_html += "</div>"

if coming_soon_currencies:
    pills_html += '<div class="pill-label">Future versions</div><div class="pill-row">'
    for currency in coming_soon_currencies:
        pills_html += f'<span class="pill-soon">{currency.flag} {currency.code}</span>'
    pills_html += "</div>"

st.markdown(pills_html, unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------------
# Currency selection (unchanged logic)
# ----------------------------------------------------------------------
# With exactly one active currency (V1's reality), skip straight to it --
# no dropdown for a single, forced choice. The moment a second active
# currency exists in the registry, this same code shows a selectbox
# automatically. No UI logic changes are needed to support that later.
if not active_currencies:
    st.error("No active currencies are configured. Add one in utils/currency_registry.py.")
    st.stop()

if len(active_currencies) == 1:
    selected_currency = active_currencies[0]
    st.markdown(
        f'<p style="text-align:center; color:#6b6a63;">Currently recognizing: '
        f'<strong>{selected_currency.display_label}</strong></p>',
        unsafe_allow_html=True,
    )
else:
    selected_label = st.selectbox(
        "Select currency",
        options=[c.display_label for c in active_currencies],
    )
    selected_currency = next(
        c for c in active_currencies if c.display_label == selected_label
    )

st.write("")


# ----------------------------------------------------------------------
# Cached model + class mapping loading (unchanged logic)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets(model_path: Path, class_indices_path: Path):
    """Load a currency's model and class mapping exactly once per server
    process, keyed by (model_path, class_indices_path).
    """
    model = load_trained_model(model_path)
    index_to_class = load_index_to_class(class_indices_path)
    return model, index_to_class


try:
    with st.spinner("Loading model..."):
        model, index_to_class = load_assets(
            selected_currency.model_path, selected_currency.class_indices_path
        )
except (FileNotFoundError, RuntimeError) as e:
    st.error(
        f"⚠️ Could not load the trained model for {selected_currency.display_label}.\n\n"
        f"**Details:** {e}\n\n"
        f"Make sure `train.py` has been run and the model files referenced in "
        f"`utils/currency_registry.py` exist before launching this app."
    )
    st.stop()

# ----------------------------------------------------------------------
# Image upload (native drag-and-drop via st.file_uploader)
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    f"Upload an image of a {selected_currency.country} banknote",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Drag and drop a file here, or click to browse.",
)

if uploaded_file is not None:
    # Two-column layout: framed image on the left, prediction card on the
    # right. Streamlit stacks columns vertically on narrow screens
    # automatically, so this stays responsive with no extra CSS.
    col_image, col_results = st.columns([1, 1], gap="large")

    with col_image:
        with st.container(border=True):
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col_results:
        try:
            with st.spinner("🔍 Analyzing note..."):
                # uploaded_file behaves like a file-like object (BytesIO),
                # which preprocess_image() accepts directly -- identical
                # preprocessing to the CLI, zero duplicated code.
                image_array = preprocess_image(uploaded_file)
                result = predict_denomination(model, image_array, index_to_class)

        except ValueError as e:
            st.error(f"⚠️ {e}")
        else:
            predicted_label = result["predicted_label"]
            confidence = result["predicted_confidence"]
            accent = get_accent(predicted_label)

            with st.container(border=True):
                st.markdown('<div class="eyebrow">Prediction</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="denom-numeral" style="color:{accent};">'
                    f'Rs. {predicted_label}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="denom-code">{selected_currency.flag} '
                    f'{selected_currency.code} &middot; {selected_currency.country}</div>',
                    unsafe_allow_html=True,
                )

                # Native confidence metric + progress bar. The progress
                # bar's color comes from .streamlit/config.toml's
                # primaryColor -- no custom CSS needed for this one.
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(min(max(confidence, 0.0), 1.0))

                st.markdown("---")
                st.markdown("**Top 3 Predictions**")

                # Numbered rank badges are appropriate here (unlike a
                # generic 01/02/03 decoration) because this genuinely is
                # a ranked sequence -- the order communicates real
                # information about relative model confidence.
                for rank, entry in enumerate(result["top_k"], start=1):
                    label = entry["label"]
                    conf = entry["confidence"]
                    bar_color = get_accent(label)
                    bar_width = max(conf * 100, 2)  # keep a sliver visible even near 0%

                    st.markdown(
                        f"""
                        <div class="rank-row">
                            <div class="rank-badge">{rank}</div>
                            <div class="rank-body">
                                <div class="rank-top-line">
                                    <span>Rs. {label}</span>
                                    <span class="rank-percent">{conf * 100:.2f}%</span>
                                </div>
                                <div class="rank-bar-track">
                                    <div class="rank-bar-fill"
                                         style="width:{bar_width:.1f}%; background:{bar_color};">
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
else:
    st.info("👆 Upload an image to get started.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown('<hr class="footer-rule">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer-text">Built with TensorFlow + MobileNetV2 + Streamlit</div>'
    '<div class="footer-version">v1.0 &middot; Pakistani Rupee (PKR) &middot; '
    'more currencies coming soon</div>',
    unsafe_allow_html=True,
)