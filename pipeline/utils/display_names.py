#!/usr/bin/env python3
"""Canonical publication-facing labels used in rendered text.

Use these constants for figure titles, legends, axis labels, and user-facing
status strings. Keep internal keys such as ``cook``/``leifer``/``gaba_a``
unchanged in data processing code.
"""

from __future__ import annotations

# Canonical publication labels.
COOK_SYNAPSES_2019 = "Cook_Synapses_2019"
RANDI_OPTOGENETICS_2023 = "Randi_Optogenetics_2023"
BENTLEY_MONOAMINES_2016 = "Bentley_Monoamines_2016"
YEMINI_GABA_2021 = "Yemini_GABA_2021"

# Convenience aliases for readability at call sites.
STRUCTURAL_LABEL = COOK_SYNAPSES_2019
FUNCTIONAL_LABEL = RANDI_OPTOGENETICS_2023
MONOAMINE_LABEL = BENTLEY_MONOAMINES_2016
GABA_LABEL = YEMINI_GABA_2021

