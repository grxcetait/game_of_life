#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 09:28:33 2026

@author: gracetait
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import argparse
from scipy.optimize import curve_fit
from numba import njit

@njit
def gol_step_numba(lattice):
    """
    Performs one step of Conway's Game of Life.
    Uses Numba for high-speed synchronous updates.
    """
    n, m = lattice.shape
    new_lattice = np.empty_like(lattice)
    
    for i in range(n):
        for j in range(m):
            # Explicitly count 8 neighbors with periodic boundaries
            # Numba handles these loops much faster than np.roll
            live_neighbors = (
                lattice[(i - 1) % n, (j - 1) % m] +
                lattice[(i - 1) % n, j] +
                lattice[(i - 1) % n, (j + 1) % m] +
                lattice[i, (j - 1) % m] +
                lattice[i, (j + 1) % m] +
                lattice[(i + 1) % n, (j - 1) % m] +
                lattice[(i + 1) % n, j] +
                lattice[(i + 1) % n, (j + 1) % m]
            )
            
            # Application of rules
            if lattice[i, j] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    new_lattice[i, j] = 0
                else:
                    new_lattice[i, j] = 1
            else:
                if live_neighbors == 3:
                    new_lattice[i, j] = 1
                else:
                    new_lattice[i, j] = 0
                    
    return new_lattice

@njit
def calculate_sum_numba(lattice):
    """Fast sum of alive sites."""
    return np.sum(lattice)

@njit
def run_until_equilibrium_numba(lattice, max_time):
    """
    Runs a single simulation until max_time or equilibrium.
    Returns the time taken and the final active sites list.
    """
    # Numba doesn't handle complex list comparisons well, 
    # so we track the last few total counts in a small array
    history = np.zeros(15, dtype=np.int32)
    time = 0
    
    current_lattice = lattice.copy()
    
    while time < max_time:
        current_lattice = gol_step_numba(current_lattice)
        total = np.sum(current_lattice)
        
        # Shift history
        for i in range(14):
            history[i] = history[i+1]
        history[14] = total
        
        time += 1
        
        if time > 15:
            # Check Period 1 (Static)
            is_static = True
            for i in range(10):
                if history[14-i] != history[13-i]:
                    is_static = False
                    break
            if is_static: break
            
            # Check Period 2
            is_p2 = True
            for i in range(10):
                if history[14-i] != history[12-i]:
                    is_p2 = False
                    break
            if is_p2: break
            
    return time

class GameOfLife(object):
    """
    A class to simulate Conway's Game of Life on a 2D square lattice 
    with periodic boundary conditions.
    """
    
    def __init__(self, n, init_cond, p_alive):
        """
        Initialise the Game of Life parameters.

        Parameters
        ----------
        n : int
            Dimension of the square lattice (n x n).
        init_cond : str
            The initial configuration type ('random', 'glider', or 'blinker').
        p_alive : float
            Probability that a cell starts as 'alive' (1) in random initialisation.

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the lattice
        self.init_cond = init_cond # Choice of the initial lattice 
        self.lattice = None
        self.p_alive = p_alive # Choice of number of cells that are alive
        
    def initialise(self):
        """
        Create the initial lattice state based on the user-selected configuration.
        States are mapped as: 0 (Dead), 1 (Alive).

        Returns
        -------
        None.

        """
        
        # Create a two-dimensional square lattice
        # Where 0 is dead and 1 is live
        # Create the initial lattice based on user input
        if self.init_cond == 'random':
        
            # Create a lattice according to a probability
            # Where p_alive is the probability a site is alive (1)
            # The default is p_alive = 0.5
            self.lattice = np.random.choice([0, 1], size = (self.n, self.n),
                                            p = [1 - self.p_alive, self.p_alive])
            
        elif self.init_cond == 'glider':
            
            # Create a lattice that has a glider
            self.lattice = np.zeros((self.n, self.n))
            
            # Start the glider in the middle of the lattice
            i = self.n // 2 
            j = self.n // 2 
            
            # Put in a glider
            self.lattice[i, j] = 1
            self.lattice[(i + 1) % self.n, (j + 1) % self.n] = 1
            self.lattice[(i + 2) % self.n, (j - 1) % self.n] = 1
            self.lattice[(i + 2) % self.n, j] = 1
            self.lattice[(i + 2) % self.n, (j + 1) % self.n] = 1
            
        else:
            
            # Create a lattice
            self.lattice = np.zeros((self.n, self.n))
            
            # Choose a random site
            i = np.random.randint(self.n)
            j = np.random.randint(self.n)
            
            # Put in a blinker
            self.lattice[i, j] = 1
            self.lattice[i - 1, j] = 1
            self.lattice[i + 1, j] = 1

    def total_alive_sites(self):
        """
        Count the total number of alive cells (state 1) in the current lattice.

        Returns
        -------
        int
            Sum of all values in the lattice array.

        """
        
        # Calculate and return the total number of alive cells in the lattice
        return np.sum(self.lattice)
    
    def update_lattice(self):
        """
        Update the lattice using the Numba-compiled function.
        """
        self.lattice = gol_step_numba(self.lattice.astype(np.int32))
        
    def center_of_mass(self):
        """
        Calculate the center of mass (COM) of all alive cells. 
        Used primarily for tracking spaceships like gliders.

        Returns
        -------
        float, float
            The x and y coordinates of the center of mass. 
            Returns (np.nan, np.nan) if the system is dead or touching a boundary.

        """
        
        # Locate where the lattice has alive cells (1) this is the glider
        glider = np.argwhere(self.lattice == 1)
        
        # If there are no alive cells, the pattern died out
        if len(glider) == 0:
            print("The system is dead. RIP.")
            return np.nan, np.nan
        
        # If the glider is at the boundaries, return np.nan
        if np.any(glider == self.n - 1) or np.any(glider == 0):
            
            return np.nan, np.nan
 
        # Otherwise, calculate the mean of the coordinates
        x_cm = np.mean(glider[:, 0]) 
        y_cm = np.mean(glider[:, 1]) 
        
        # Return the centre of mass
        return x_cm, y_cm 
    

class Simulation(object):
    """
    A class to manage the execution and visualisation of Game of Life experiments,
    including animations, equilibrium distributions, and spaceship tracking.
    """
    
    def __init__(self, n, init_cond, steps, p_alive):
        """
        Initialise simulation parameters.

        Parameters
        ----------
        n : int
            Lattice size.
        init_cond : str
            Initial configuration type.
        steps : int
            Number of steps for animation or measurement trials.
        p_alive : float
            Density of alive cells for random init.

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the square lattice
        self.init_cond = init_cond # Choice of the initial lattice 
        self.steps = steps # Choice of number of steps for the animation
        self.p_alive = p_alive # Choice of number of cells that are alive
        
    def animate(self, steps):
        """
        Execute and display a real-time animation of the Game of Life evolution.

        Parameters
        ----------
        steps : int
            Number of frames to animate.

        Returns
        -------
        None.

        """
        
        # Initialise the lattice using the GameOfLife class
        game_of_life = GameOfLife(self.n, self.init_cond, self.p_alive)
        game_of_life.initialise()
        
        # Define the figure and axes for the animation
        fig, ax = plt.subplots()
        
        # Define custom cmap
        red_black_cmap = ListedColormap(["#000000", "#FF0000"]) 
        
        # Initialise the image object
        # vmin/vmax ensure 0 is black and 1 is red consistently
        im = ax.imshow(game_of_life.lattice, cmap = red_black_cmap,
                       vmin = 0, vmax = 1)
        
        # Run the animation for the total number of steps
        for s in range(steps):
            
            # Update lattice
            game_of_life.update_lattice_faster()
            
            # Update the animation 
            im.set_data(game_of_life.lattice)
            ax.set_title(f"Step: {s}")
            
            # Keep the image up while the script is running
            plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
    def equilibrium_check(self, active_sites, ind_cond = 10):
        """
        Check if the simulation has reached equilibrium by identifying 
        static states (Period 1) or oscillators (Period 2 and 3).

        Parameters
        ----------
        active_sites : list
            Record of the total number of alive cells.
        ind_cond : int, optional
            The minimum window size required for the check. Default is 10.

        Returns
        -------
        bool
            True if equilibrium (periodicity) is detected, False otherwise.

        """
 
        # If the array if not long enough, return False
        if len(active_sites) < ind_cond:
            return False
        
        # Check if it is static
        # Checks to see if the last 10 elements are the same
        if active_sites[-10:] == active_sites[-11:-1]:
            #print("System is static (Period 1)")
            #print(active_sites[-10:])
            return True
        
        # Check if it is oscillating with a period of 2
        # Checks to see if the last 10 elements are oscillating
        elif active_sites[-10:] == active_sites[-12:-2]:
            print("Oscilalting with period of 2")
            print(active_sites[-12:])
            return True
        
        # Check if it is oscillating with a period of 3
        # Checks to see if the last 9 elements are oscillating
        elif active_sites[-9:] == active_sites[-12:-3]:
            print("Oscilalting with period of 3")
            print(active_sites[-12:])
            return True
        
        # If not, it is not in equilibrium
        else:
            return False
        
    def equilibrium_measurements(self, filename, steps):
        """
        Run multiple random simulations and record the time taken for each 
        to reach an absorbing or oscillating state.

        Parameters
        ----------
        filename : str
            File to save the equilibration times.
        steps : int
            The number of independent simulations to run.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # Make empty lists to hold data points
        equilibriation_time = []
        #no_active_sites = []
        
        # Define maximum time until it cuts out
        max_time = 10000 
        
        # Iterate through simulation steps
        for s in range(steps):
            print(f"\rSimulating step = {s + 1}/{steps}...", end="", flush=True)
            
            # Initialise the lattice using the GameOfLife class
            game_of_life = GameOfLife(self.n, self.init_cond, self.p_alive)
            game_of_life.initialise()
            
            # Run the optimised Numba runner
            time_taken = run_until_equilibrium_numba(
                game_of_life.lattice.astype(np.int32), 
                max_time)
            
            equilibriation_time.append(time_taken)
            
        print() # Add this to move to a new line after the bar reaches 100%
            
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the specified file
        with open(file_path, "w") as f:
            for i in range(len(equilibriation_time)):
                
                # Get the time it took
                t = equilibriation_time[i]
  
                f.write(f"{t}\n")
                
    def glider_measurements(self, steps, filename):
        """
        Track a glider's center of mass over time, accounting for periodic 
        boundary conditions, to estimate displacement.

        Parameters
        ----------
        steps : int
            Number of steps to track the glider.
        filename : str
            File to save the COM coordinates and time.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # Initialise the lattice using the GameOfLife class
        game_of_life = GameOfLife(self.n, self.init_cond, self.p_alive)
        game_of_life.initialise()
        
        # Create empty lists to hold datapoints
        x = []
        y = []
        r = []
        time = []
        
        # Define differences in coordinates as 0 for now
        x_diff = 0
        y_diff = 0
        
        # Start counts of boundary jumps at 0
        boundary_jump_x = 0
        boundary_jump_y = 0
        
        # Run the measurements for the total number of steps
        for s in range(steps):
            #print(f"Simulating step = {s}/{steps}")
            
            # Update lattice
            game_of_life.update_lattice()
            
            # Calculate glider center of mass
            x_raw, y_raw = game_of_life.center_of_mass()
            
            # If the glider is crossing a boundary, do not append to list
            if np.isnan(x_raw) or np.isnan(y_raw):
                continue
            
            # If the length of the array is long enough
            if len(x) > 0:
            
                # Calculate the difference between current and previous location
                x_diff = x_raw - (x[-1] - boundary_jump_x * self.n)
                y_diff = y_raw - (y[-1] - boundary_jump_y * self.n)

            # Take into account periodic boundary conditions
            if x_diff > self.n / 2:
                boundary_jump_x -= 1
                
            elif x_diff < -self.n / 2:
                boundary_jump_x += 1
                
            if y_diff > self.n / 2:
                boundary_jump_y -= 1
                
            elif y_diff < -self.n / 2:
                boundary_jump_y += 1
        
                
            # Calculate the corrected centre of mass 
            # Taking into account periodic boundaries
            x_cm = x_raw + boundary_jump_x * self.n
            y_cm = y_raw + boundary_jump_y * self.n
            r_cm = np.sqrt(x_cm**2 + y_cm**2)
            
            # Append to the lists
            x.append(x_cm)
            y.append(y_cm)
            r.append(r_cm)
            time.append(s)
                
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the specified file
        with open(file_path, "w") as f:
            for i in range(len(x)):
                
                f.write(f"{x[i]},{y[i]},{r[i]},{time[i]}\n")
                
    def f(self, x, A, B): 
        """
        Simple linear function for curve fitting (y = Ax + B).
        
        In the context of the glider tracking, A represents the velocity (speed) 
        of the glider, and B represents the initial position offset.

        Parameters
        ----------
        x : float or numpy.ndarray
            The independent variable (time steps).
        A : float
            The slope of the line, representing the velocity (cells per step).
        B : float
            The y-intercept, representing the initial position at t=0.

        Returns
        -------
        float or numpy.ndarray
            The predicted position(s) corresponding to the input time step(s).

        """
        return A * x + B
            
    def plot_glider_measurements(self, filename):
        """
        Plot the displacement of the glider (x, y, and radial) over time and 
        fit a linear model to estimate velocity.

        Parameters
        ----------
        filename : str
            Data file containing recorded coordinates.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        # Create an empty list to store input data
        input_data = []

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" []\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            
        # Make empty lists to store data
        x_cm = []
        y_cm = []
        r_cm = []
        time = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 4):
            
            # Obtain vlaue from input data
            x = float(input_data[i])
            y = float(input_data[i+1])
            r = float(input_data[i+2])
            t = float(input_data[i+3])
            
            # Append to lists
            x_cm.append(x)
            y_cm.append(y)
            r_cm.append(r)
            time.append(t)
            
        # Convert to arrays
        time = np.array(time)
        x_cm = np.array(x_cm)
        y_cm = np.array(y_cm)
        r_cm = np.array(r_cm)

        # Create a mask where x_cm is NOT nan
        mask = np.isfinite(x_cm) & np.isfinite(time)

        # Apply mask to both lists to keep them synchronized
        time = time[mask]
        x_cm = x_cm[mask]
        y_cm = y_cm[mask]
        r_cm = r_cm[mask]
            
        fig, ax = plt.subplots(1,3, figsize = (16,4.5)) 
        plt.rcParams['font.size'] = 10
        t = np.arange(0,len(r_cm),1)   
        
        # --- Subplot 0: X-position ---
        popt, pcov = curve_fit(self.f, time, x_cm)
        ax[0].set_title("Glider X-position versus time")             
        ax[0].set_ylabel("Glider X-position")   
        ax[0].set_xlabel("Time")  
        # Plot the best fit line
        ax[0].plot(time, self.f(time, *popt), 'k-', label=f'Fit: {popt[0]:.3f}t + {popt[1]:.3f}')
        # Plot the raw data
        ax[0].plot(time, x_cm, 'bs', markersize=2, label='Data')
        ax[0].legend(loc="upper right")
        
        # --- Subplot 1: Y-position ---
        popt, pcov = curve_fit(self.f, time, y_cm)
        ax[1].set_title("Glider Y-position versus time")             
        ax[1].set_ylabel("Glider Y-position")   
        ax[1].set_xlabel("Time")
        # Plot the best fit line
        ax[1].plot(time, self.f(time, *popt), 'k-', label=f'Fit: {popt[0]:.3f}t + {popt[1]:.3f}')
        # Plot the raw data
        ax[1].plot(time, y_cm, 'ro', markersize=2, label='Data')
        ax[1].legend(loc="upper right")
        
        # --- Subplot 2: R-position ---
        popt, pcov = curve_fit(self.f, time, r_cm)
        ax[2].set_title("Glider R-position versus time")             
        ax[2].set_ylabel("Glider R-position")   
        ax[2].set_xlabel("Time") 
        # Plot the best fit line
        ax[2].plot(time, self.f(time, *popt), 'k-', label=f'Fit: {popt[0]:.3f}t + {popt[1]:.3f}')
        # Plot the raw data
        ax[2].plot(time, r_cm, 'yo', markersize=2, label='Data')
        ax[2].legend(loc="upper right")
        
        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
            
    def plot_equilibrium_distribution(self, filename):
        """
        Generate a histogram showing the distribution of times needed for 
        random Game of Life lattices to reach equilibrium.

        Parameters
        ----------
        filename : str
            Data file containing recorded times.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        # Create empty plots
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 10))

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            
        # Make empty lists to store data
        equilibriation_time = []
        #no_active_sites = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data)):
            
            # Obtain vlaue from input data
            time = float(input_data[i])
            
            # Append to lists
            equilibriation_time.append(time)
            
        # Calculate the mean and median of the equilibriation time
        mean_val = np.mean(equilibriation_time)
        median_val = np.median(equilibriation_time)
    
        # Plot a histogram of the equilibriation time
        ax1.hist(equilibriation_time, bins = 30,
                color = "skyblue", edgecolor = "black")
        ax1.set_ylabel("Counts", fontsize = 14)
        ax1.set_xlabel("Time to equilibriate", fontsize = 14)
        ax1.set_title("Distribution of the time to equilibriation", fontsize = 16)
        
        # Add vertical lines for mean and median
        ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='dashdot', linewidth=2, 
                    label=f'Median: {median_val:.2f}')
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True)
        ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
        
        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellular Automata: Game of Life")

    # User input parameters
    parser.add_argument("--n", type = int, default = 50, help = "Lattice size (n x n)")
    parser.add_argument("--steps", type = int, default = 1000, help = "Number of simulation steps")
    parser.add_argument("--init", type = str, default = "random", 
                        choices = ["random", "glider", "blinker"],
                        help = "Initial state for Game of Life")
    parser.add_argument("--mode", type = str, default = "ani", 
                        choices = ["ani", "mea"],
                        help = "Animation or measurements")
    parser.add_argument("--p_alive", type = float, default = 0.5, help = "Probability of a site being alive")
        
    args = parser.parse_args()
        
    # Game of Life uses n, init, and steps
    sim = Simulation(n = args.n, init_cond = args.init, steps = args.steps, p_alive = args.p_alive)
    
    if args.init == "random" and args.mode == "mea":
        
        filename = f"game_of_life_equilibrium_distribution_{args.steps}_systems_11.txt" # Change the name
        sim.equilibrium_measurements(filename, steps = args.steps)
        sim.plot_equilibrium_distribution(filename)
       
    elif args.init == "glider" and args.mode == "mea":

        filename = f"game_of_life_glider_measurements_{args.steps}_steps_5.txt"
        sim.glider_measurements(filename = filename, steps = args.steps)
        sim.plot_glider_measurements(filename)
        
    else:
    
        sim.animate(steps = args.steps)
    