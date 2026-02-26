from pydantic import BaseModel
from typing import Any, Optional

class DriftResponse(BaseModel):
    drift_detected: bool
    details: Optional[Any] = None  # Replace Any with the expected details type

# Add this model at the appropriate location in schemas.py. If there are other related models, modify as needed.