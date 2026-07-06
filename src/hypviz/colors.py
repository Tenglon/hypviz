"""Per-point colors for the three atlas encodings, from the validated dataviz
palette (sequential blue ramp for magnitude; categorical hues for identity)."""
import numpy as np

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


def by_scalar(values, ramp=SEQ):
    """Map a continuous quantity (depth, norm) onto a sequential ramp."""
    v = np.asarray(values, float)
    v = (v - v.min()) / (np.ptp(v) or 1)
    idx = np.clip((v * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
    return [ramp[i] for i in idx]


def by_category(labels, palette=CAT):
    """Assign a categorical hue per distinct label, in first-seen order."""
    order = {u: i for i, u in enumerate(dict.fromkeys(map(str, labels)))}
    return [palette[order[str(l)] % len(palette)] for l in labels]
