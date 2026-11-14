
import numpy as np

def transform_raw_features(raw_features, artifact):
    """
    Accept a mapping of all 78 raw CICIDS features.
    Scale 78 -> clip -> select 70, without fitting.

    This validates structure and numerical values, not whether
    the raw features were extracted correctly from network traffic.
    """
    if not isinstance(raw_features, dict):
        raise ValueError("Expected a raw-feature dictionary")

    raw_names = list(artifact["raw_feature_columns"])
    selected_names = list(artifact["selected_feature_columns"])
    indices = np.asarray(artifact["selected_indices"])

    if len(raw_names) != 78 or len(selected_names) != 70:
        raise ValueError("Invalid frozen feature dimensions")

    if (indices.shape != (70,)
        or not np.issubdtype(indices.dtype, np.integer)
        or len(np.unique(indices)) != 70
        or np.any(indices < 0)
        or np.any(indices >= 78)):
        raise ValueError("Invalid feature selection indices")

    if [raw_names[int(i)] for i in indices] != selected_names:
        raise ValueError("Frozen feature order mismatch")

    if set(raw_features) != set(raw_names):
        raise ValueError("Missing or unexpected raw features")

    values = []

    for name in raw_names:
        value = raw_features[name]

        if type(value) not in (int, float, np.float32, np.float64,
                               np.int32, np.int64):
            raise ValueError(f"Invalid feature type: {name}")

        values.append(float(value))

    x = np.asarray(values, dtype=np.float64).reshape(1, 78)

    if not np.isfinite(x).all():
        raise ValueError("Non-finite raw input")

    scaler = artifact["scaler"]

    if scaler.n_features_in_ != 78:
        raise ValueError("Scaler dimension mismatch")

    scaled = scaler.transform(x)

    if artifact["clip_after_transform"] is True:
        scaled = np.clip(scaled, 0.0, 1.0)
    else:
        raise ValueError("Expected post-transform clipping")

    selected = np.asarray(
        scaled[:, indices], dtype=np.float32
    )

    if selected.shape != (1, 70):
        raise ValueError("Incorrect transformed shape")

    if not np.isfinite(selected).all():
        raise ValueError("Non-finite transformed features")

    return selected
