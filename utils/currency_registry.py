"""
utils/currency_registry.py

Purpose
-------
Single source of truth for "which currencies does this app support, and
where does each one's trained model live." This is the entire multi-
currency extensibility mechanism for the project -- deliberately kept as
a plain list of small config objects, not a database, plugin system, or
dynamic loader, because a 7-currency ceiling doesn't justify that
complexity. Simple, boring, and easy to extend beats clever here.

How to add a new currency later
--------------------------------
1. Train a model for that currency using train.py, producing its own
   .keras file and class_indices.json (e.g. models/usd_model.keras).
2. Add one CurrencyConfig entry below with status="active" and the
   correct model_path / class_indices_path.
3. That's it. app.py already branches on "how many active currencies
   exist" -- once there are 2+, it automatically shows a selection
   dropdown instead of skipping straight to the single supported
   currency. No app.py code changes are required.

Why this lives outside app.py and predict.py
----------------------------------------------
predict.py already accepts model_path/class_indices_path as arguments to
its loader functions (see load_trained_model / load_index_to_class) --
it has no hardcoded assumption about there being exactly one currency.
That means the ONLY new piece of infrastructure multi-currency support
needs is "a place to list the currencies and their file paths," which is
exactly what this module is. Keeping it separate from app.py also means
a future non-Streamlit interface (an API, a CLI batch tool) could reuse
the same registry.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------
# Status values
# ----------------------------------------------------------------------
# "active"      -> a trained model exists; users can select and use it.
# "coming_soon" -> listed in the UI for roadmap visibility, but not
#                  selectable -- no model_path is expected to exist yet.
STATUS_ACTIVE = "active"
STATUS_COMING_SOON = "coming_soon"


@dataclass(frozen=True)
class CurrencyConfig:
    """Metadata for a single currency the app knows about.

    model_path / class_indices_path are Optional because "coming_soon"
    entries have no trained model yet -- they exist purely so the UI can
    advertise the roadmap without pretending a model is ready.
    """
    code: str            # ISO-style currency code, e.g. "PKR"
    country: str         # Display name, e.g. "Pakistan"
    flag: str            # Emoji flag, e.g. "🇵🇰"
    status: str          # STATUS_ACTIVE or STATUS_COMING_SOON
    model_path: Optional[Path] = None
    class_indices_path: Optional[Path] = None

    @property
    def display_label(self) -> str:
        """Human-readable label used in dropdowns and lists, e.g.
        '🇵🇰 Pakistan (PKR)'.
        """
        return f"{self.flag} {self.country} ({self.code})"


# ----------------------------------------------------------------------
# The registry itself
# ----------------------------------------------------------------------
# V1 ships with exactly one active currency: Pakistani Rupee. The rest are
# listed as "coming_soon" purely for the UI's roadmap section -- flip one
# to "active" (with real model paths) once it has a trained model, and it
# becomes selectable with zero other code changes.
CURRENCY_REGISTRY: list[CurrencyConfig] = [
    CurrencyConfig(
        code="PKR",
        country="Pakistan",
        flag="🇵🇰",
        status=STATUS_ACTIVE,
        model_path=Path("models/currency_model.keras"),
        class_indices_path=Path("models/class_indices.json"),
    ),
    CurrencyConfig(code="USD", country="United States", flag="🇺🇸", status=STATUS_COMING_SOON),
    CurrencyConfig(code="EUR", country="Eurozone", flag="🇪🇺", status=STATUS_COMING_SOON),
    CurrencyConfig(code="INR", country="India", flag="🇮🇳", status=STATUS_COMING_SOON),
    CurrencyConfig(code="GBP", country="United Kingdom", flag="🇬🇧", status=STATUS_COMING_SOON),
]


def get_active_currencies() -> list[CurrencyConfig]:
    """Currencies with a trained model, selectable in the app right now."""
    return [c for c in CURRENCY_REGISTRY if c.status == STATUS_ACTIVE]


def get_coming_soon_currencies() -> list[CurrencyConfig]:
    """Currencies listed on the roadmap but not yet selectable."""
    return [c for c in CURRENCY_REGISTRY if c.status == STATUS_COMING_SOON]