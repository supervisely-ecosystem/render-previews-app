import numpy as np


def get_thickness(render: np.ndarray, thickness_factor: float) -> int:
    render_height, render_width, _ = render.shape
    return int(max(render_height, render_width) * thickness_factor / 100)
