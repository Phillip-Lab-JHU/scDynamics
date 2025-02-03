# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/MaxHalford/prince
# ==============================================================================

from __future__ import annotations

import importlib.metadata

#from .ca import CA
from .famd import FAMD
#from .gpa import GPA
#from .mca import MCA
#from .mfa import MFA
from .pca import PCA

#__all__ = ["CA", "FAMD", "MCA", "MFA", "PCA", "GPA", "datasets"]
__all__ = ["FAMD", "PCA"]