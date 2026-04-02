# Game of Life — `game_of_life.py`

Python script for simulating Conway's Game of Life on a 2D square lattice with periodic boundary conditions.

## game_of_life.py

This script has all the functions and classes to either run an animation or measurements of the simulation. 
The user needs to put in different arguements and customise the animation or measurement conditions.

For the random initial lattice condition, each cell is alive with probability 'p_alive'.
For the glider initial lattice condition, there is a 5-cell glider placed at the centre of the lattice.
For the blinker initial lattice condition, there is a 3-cell vertical blinker placed at a random location.

### Arguments
- 'n', Lattice size (n x n), Default = 50
- 'steps', Number of simulation steps, Default = 1000
- 'init', Initial configuration of the lattice: 'random', 'glider', or 'blinker', Default = 'random'
- 'mode', Mode of 'ani' (animation) or 'mea' (measurements), Default = 'ani'
- 'p_alive', Probability of a cell starting as alive (random init only), Default = 0.5

### Output
All outputs are saved relative to the script's directory:

```
outputs/
├── datafiles/     # Raw measurement data (.txt)
└── plots/         # Saved figures (.png, 300 dpi)
```

## Command line examples

### Animation (`--mode ani`)

Display an animation of the lattice evolving under Game of Life rules. Works with any `--init` choice.

```
python game_of_life.py --mode ani --init random --steps 500
python game_of_life.py --mode ani --init glider
python game_of_life.py --mode ani --init blinker
```

### Measurements (`--mode mea`)

Run data collection depending on the `--init` choice:

**Random init** — measures the distribution of equilibration times across multiple simulations.

```
python game_of_life.py --mode mea --init random --steps 1000 --n 50
```

**Glider init** — tracks the glider's centre of mass over time and fits a linear model to estimate its velocity.

```
python game_of_life.py --mode mea --init glider --steps 1000 --n 50
```

---

## Outputs

All outputs are saved relative to the script's directory:

```
outputs/
├── datafiles/     # Raw measurement data (.txt)
└── plots/         # Saved figures (.png, 300 dpi)
```
