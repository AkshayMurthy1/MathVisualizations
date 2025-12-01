from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.integrate import solve_ivp
import pandas as pd



def double_pendulum_simple(t,state,l1=1, l2=1, m1=1, m2=1, g=9.8):
    theta1, theta2, theta1_dot, theta2_dot = state

    #use matrices to solved coupled differential equations
    theta_diff = theta1-theta2
    A = np.array([[(m1+m2)*l1,m2*l2*np.cos(theta_diff)],[l1*np.cos(theta_diff),l2]])
    F = np.array([-(m1+m2)*g*np.sin(theta1)-m2*l2*theta2_dot**2*np.sin(theta_diff),-g*np.sin(theta2)+theta1_dot**2*np.sin(theta_diff)])

    theta_dot_dot = np.linalg.solve(A,F)
    return theta1_dot,theta2_dot,theta_dot_dot[0],theta_dot_dot[1]

def solution_points(state0,time,dt=0.1):
    solution = solve_ivp(
        double_pendulum_simple,
        t_span=[0,time],
        y0=state0,
        t_eval=np.arange(0,time,dt)
    )
    #print(solution.y[0:2].shape)
    solution.y[0, :] = solution.y[0, :] % (2 * np.pi)  # Normalize theta1
    solution.y[1, :] = solution.y[1, :] % (2 * np.pi)  # Normalize theta2
    #print(solution.y.shape)
    return solution.y.T #return all



def compute_divergence(theta1, theta2):
    state0 = [theta1, theta2, 0, 0]
    s = 1e-3
    state0p = [theta1 + s, theta2 + s, 0, 0]
    sol1 = solution_points(state0, 10)
    sol2 = solution_points(state0p, 10)
    return np.mean(np.linalg.norm(sol1 - sol2, axis=1))

def heatmap():
    #ini_conditions = np.linspace(-1*np.pi, np.pi, 400)
    ini_conditions = np.linspace(0,2*np.pi,400)
    grid = np.zeros((len(ini_conditions), len(ini_conditions)))

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_divergence)(theta1, theta2)
        for theta1 in ini_conditions
        for theta2 in ini_conditions
    )

    # Convert results to 2D grid
    grid = np.array(results).reshape(len(ini_conditions), len(ini_conditions))
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv("divergence_heatmap.csv", index=False)
    fig,ax = plt.subplots(1, 3, figsize=(7,14))
    plt.imshow(grid,cmap="plasma",ax=ax[0])
    plt.show()
    # plt.colorbar(label="Divergence")
    # plt.title("Heatmap of Chaos Indicators")
    # plt.xlabel("Initial theta 1")
    # plt.ylabel("Initial theta 2")
    plt.imshow(grid, cmap="inferno",ax=ax[1])
    plt.show()
    plt.imshow(grid,cmap="hsv",ax=ax[2])
    plt.show()
    
    

heatmap()