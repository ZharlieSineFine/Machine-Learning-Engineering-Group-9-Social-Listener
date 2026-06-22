"""Shared design system for the dashboard — fonts, light/dark palette, top bar.

Both pages (app.py = Marketing, pages/mlops_monitor.py = MLOps Monitor) call
``inject(active_theme())`` at the top, then ``topbar(active)``. The theme is a
runtime light/dark toggle persisted in ``st.session_state`` (shared across pages).

Colours come from the BrewLeaf design (design/ folder). HTML/CSS in the pages
references the CSS variables (``var(--teal)`` …) so a single toggle re-themes
everything; Plotly (which can't resolve CSS vars) reads concrete hex from
``palette()``.

Owner: Amelia.
"""
from __future__ import annotations

import streamlit as st

# --- palettes (concrete values; mirrored as CSS vars by inject) -------------
DARK = {
    "page": "#16181D", "card": "#1E2128", "card2": "#23262E", "border": "#2E3039",
    "text": "#E8E6DF", "muted": "#888780", "faint": "#5C5B55",
    "teal": "#1D9E75", "red": "#E24B4A", "amber": "#EF9F27", "brown": "#5C3A21", "blue": "#4A90D9",
    "teal_soft": "rgba(29,158,117,.14)", "red_soft": "rgba(226,75,74,.12)",
    "good_bg": "#162414", "good_bd": "#2A5C2A", "good_fg": "#6FCF6F",
    "bad_bg": "#2E1A1A", "bad_bd": "#6B2222", "bad_fg": "#F09595",
    "alert_bg": "#23171A", "row_hl": "rgba(29,158,117,.08)", "grid": "rgba(255,255,255,.05)",
    "blue_soft": "rgba(74,144,217,.16)",
}
LIGHT = {
    "page": "#F2F0E8", "card": "#FFFFFF", "card2": "#FAF8F1", "border": "#E4E0D5",
    "text": "#23211C", "muted": "#86837A", "faint": "#A8A498",
    "teal": "#147A57", "red": "#C73B3A", "amber": "#B9760E", "brown": "#5C3A21", "blue": "#2F6FB0",
    "teal_soft": "rgba(20,122,87,.12)", "red_soft": "rgba(199,59,58,.10)",
    "good_bg": "#E8F1E4", "good_bd": "#BCD7B4", "good_fg": "#2E7D32",
    "bad_bg": "#FAE9E8", "bad_bd": "#E7C4C1", "bad_fg": "#B23B39",
    "alert_bg": "#FBF1ED", "row_hl": "rgba(20,122,87,.07)", "grid": "rgba(0,0,0,.06)",
    "blue_soft": "rgba(47,111,176,.16)",
}

# CSS-variable name aliases for use inside the pages' HTML f-strings. Importing
# these (instead of hex literals) makes the existing markup theme-aware.
TEAL = "var(--teal)"; RED = "var(--red)"; AMBER = "var(--amber)"
BROWN = "var(--brown)"; BLUE = "var(--blue)"
CARD_BG = "var(--card)"; CARD2 = "var(--card2)"; PAGE_BG = "var(--page)"
BORDER = "var(--border)"; TEXT_PRI = "var(--text)"; TEXT_SEC = "var(--muted)"
FAINT = "var(--faint)"


def active_theme() -> str:
    """Current theme from the persisted toggle (default dark)."""
    return "light" if st.session_state.get("theme_radio") == "Light" else "dark"


def palette(theme: str | None = None) -> dict:
    """Concrete colours for the active theme (for Plotly / anything not CSS)."""
    return LIGHT if (theme or active_theme()) == "light" else DARK


def inject(theme: str) -> None:
    """Inject fonts, the CSS variables for ``theme``, and base page styling."""
    c = palette(theme)
    vars_css = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in c.items())
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Hanken+Grotesk:wght@400;500;600;700&family=Spline+Sans+Mono:wght@400;500&display=swap');
        :root {{ {vars_css}; }}
        .stApp {{ background: var(--page); }}
        .block-container {{ padding: 1.4rem 2rem 2.5rem; max-width: 1320px; }}
        #MainMenu, header, footer {{ visibility: hidden; }}
        html, body, [class*="css"], .stMarkdown, p, li, span, div {{
            font-family: 'Hanken Grotesk', sans-serif; color: var(--text);
        }}
        h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif; color: var(--text) !important; }}
        hr {{ border-color: var(--border) !important; margin: 0.4rem 0; }}
        /* segmented radios (theme toggle + Daily/Weekly) */
        div[role="radiogroup"] {{ gap: 4px; }}
        div[role="radiogroup"] label {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 7px; padding: 3px 12px; margin: 0;
        }}
        div[role="radiogroup"] label p {{ color: var(--muted) !important; font-size: 12px; }}
        .stPlotlyChart {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def topbar(active: str) -> None:
    """Top bar: View switch (Marketing / MLOps) on the left, Dark/Light toggle on the right."""
    def _link(label: str, href: str, on: bool) -> str:
        style = (
            "background:var(--teal);color:#fff;"
            if on else "background:transparent;color:var(--muted);"
        )
        return (
            f'<a href="{href}" target="_self" style="text-decoration:none;'
            f'padding:7px 15px;border-radius:7px;font-size:13px;font-weight:600;{style}">{label}</a>'
        )

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            '<span style="font-size:10px;color:var(--muted);text-transform:uppercase;'
            'letter-spacing:.1em;font-weight:600;">View</span>'
            '<div style="display:flex;gap:6px;background:var(--card);border:1px solid var(--border);'
            'border-radius:10px;padding:4px;">'
            + _link("🍃 Marketing", "/", active == "marketing")
            + _link("🔬 MLOps Monitor", "/mlops_monitor", active == "mlops")
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.radio(
            "Theme", ["Dark", "Light"],
            key="theme_radio", horizontal=True, label_visibility="collapsed",
        )
