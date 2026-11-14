
import math
import numpy as np

FIELDS = (
    "traffic_intensity",
    "queue_utilization",
    "estimated_latency",
    "bandwidth_utilization",
    "ids_risk_score",
    "device_trust_score",
    "service_health",
    "active_security_policy",
)

ALLOWED_ORIGINS = {
    "MEASURED",
    "IDS_INFERENCE",
    "DATASET_PROXY",
    "SYNTHETIC",
    "ESTIMATED",
    "POLICY_STATE",
}

FORBIDDEN_FIELDS = {
    "binary_label",
    "is_malicious",
    "ground_truth_label",
}

def build_observation(values, provenance):
    """
    Validate and construct an 8-element float32 observation.

    This function validates structure, range and provenance labels.
    It does NOT verify whether sources are trustworthy or properly
    calibrated to the RL training distribution.
    """

    if not isinstance(values, dict):
        raise ValueError("values must be a dictionary")

    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")

    if set(values) & FORBIDDEN_FIELDS:
        raise ValueError("Hidden ground-truth field detected")

    if set(provenance) & FORBIDDEN_FIELDS:
        raise ValueError("Hidden ground-truth provenance detected")

    if set(values) != set(FIELDS):
        raise ValueError("Missing or unexpected observation fields")

    if set(provenance) != set(FIELDS):
        raise ValueError("Incomplete provenance contract")

    result = []

    for field in FIELDS:
        value = values[field]

        if type(value) not in (int, float):
            raise ValueError(f"{field}: invalid numeric type")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{field}: non-finite value")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field}: outside [0, 1]")

        origin = provenance[field]

        if origin not in ALLOWED_ORIGINS:
            raise ValueError(f"{field}: unknown provenance")

        result.append(value)

    vector = np.asarray(result, dtype=np.float32)

    if vector.shape != (8,):
        raise ValueError("Incorrect observation dimension")

    return vector
