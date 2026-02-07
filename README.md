# MathVisualizations

Organized repository for reproducible math-art outputs. All of the art here is available on [My Pursuit of Pendulums Website](https://pursuitofpendulums-1.onrender.com/), as well.

## Structure

- `notebooks/core/`: notebooks that generate final art assets.
- `notebooks/exploration/`: experiments and intermediate workflows.
- `artifacts/intensity/`: intensity-map image outputs.
- `artifacts/lyapunov/`: Lyapunov/fractal image outputs.
- `artifacts/l2norm/`: L2/divergence image outputs.
- `artifacts/l2norm/sources/`: CSV source grids used by L2 notebooks.
- `artifacts/music/`: generated and supporting music assets (`.mid`, `.sf2`, optional `.wav`).
- `artifacts/intermediate/`: intermediate `.npy` arrays used between notebooks.

## Canonical notebooks

- `notebooks/core/image.ipynb`
- `notebooks/core/image_w_smooth.ipynb`
- `notebooks/core/music.ipynb`
- `notebooks/core/math_art_more.ipynb`
- `notebooks/core/triple_pendulum.ipynb`

## Notes

- Notebook file paths were updated to write outputs into `artifacts/`.
- Exploration notebooks are kept for traceability, but final outputs are under `artifacts/`.
