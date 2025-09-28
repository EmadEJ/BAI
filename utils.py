import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Approximation of Lambert W function
def Cg(x):
    return x + np.log(x)

# divergence of bernouli d(delta, 1-delta)
def dB(x):
    return x * np.log(x / (1-x)) + (1-x) * np.log((1-x) / x)

def hidden_action_sampler(distribution, n_samples = None):
    if n_samples:
        return np.random.choice(np.arange(len(distribution)), size = n_samples, p = distribution)
        
    return np.random.choice(np.arange(len(distribution)), p = distribution)


def context_initiator(n, k, mode = "random"):

    if mode == "random":
        samples = np.random.uniform(low=0.0, high=1.0, size=(n, k))
        row_sums = samples.sum(axis=1, keepdims=True)
        normalized_samples = samples / row_sums
        
    if mode == "random with min 1/2k":
        samples = np.random.uniform(low=(k-1)/(2*k-1), high=1.0, size=(n, k))
        row_sums = samples.sum(axis=1, keepdims=True)
        normalized_samples = samples / row_sums
        
    if mode == "random with min 1/4k":
        samples = np.random.uniform(low=1/4, high=1.0, size=(n, k))
        row_sums = samples.sum(axis=1, keepdims=True)
        normalized_samples = samples / row_sums

    return normalized_samples


def convert_back_to_X_space(A, z):#solve for Ax=z

    n = A.shape[0]

    # optimization variable
    x = cp.Variable(n)
    constraints = [x >= 0, cp.sum(x) == 1]

    # objective function: minimize ||Ax - b||² (least squares)
    objective = cp.Minimize(cp.norm(A.T @ x - z, 2))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status != cp.OPTIMAL:
        return False
    
    return x.value


def find_projected_on_simplex_equivalent_in_X_space(A, y):
    # minimize |Ax-y| subject to x on simplex

    n = A.shape[0]           
    x = cp.Variable(n)          

    objective = cp.Minimize(cp.norm(A.T @ x - y, 2)**2)
    constraints = [x >= 0, cp.sum(x) == 1]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return x.value


def plot_mus(list1, list2, list3):
    plt.figure(figsize=(20, 15))

    l1 = mean_of_columns(list1)
    l2 = mean_of_columns(list2)
    l3 = mean_of_columns(list3)
    
    plt.plot(np.arange(len(l1)), l1, color='blue', label='NSTS with known contexts')
    plt.plot(np.arange(len(l2)), l2, color='orange', label='NSTS with unknown contexts')
    plt.plot(np.arange(len(l3)), l3, color='green', label='TS')

    plt.xlabel("rounds")
    plt.ylabel("mean of distance of estimated w of each algorithm with actual w ")
    plt.title("Plot of Two Lists of Lists")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


def mean_of_columns(list_of_lists):
    max_length = max(len(sublist) for sublist in list_of_lists)
    
    means = []
    for i in range(max_length):
        column = [sublist[i] for sublist in list_of_lists if i < len(sublist)]
        
        if column:
            means.append(sum(column) / len(column))
        else:
            means.append(None)  

    return means


def gaussian_kl(mu1, mu2):
    return (mu1 - mu2)**2 / 2


def categorical_kl(p, q):
    return sum([p[i] * np.log(p[i]/q[i]) for i in range(len(p))])

def to_cartesian(tri_coords):
    x2 = tri_coords[:, 1]
    x3 = tri_coords[:, 2]
    
    h = np.sqrt(3) / 2
    
    X = x2 + 0.5 * x3
    Y = h * x3
    
    return X, Y

def draw_simplex_heatmap(Ts):
    """
    Generates and plots a heatmap on the 2-simplex from scattered data,
    highlighting the optimal point (maximum Z value).
    """
    # 2. Prepare Data
    tri_coords = np.array(list(Ts.keys()))
    Z = np.array(list(Ts.values()))
    h = np.sqrt(3) / 2 

    # --- Find Optimal Point ---
    # Find the barycentric coordinates (key) with the maximum value (Z)
    optimal_bary_coords_tuple = max(Ts, key=Ts.get)
    optimal_Z_value = Ts[optimal_bary_coords_tuple]
    
    # Convert the optimal barycentric coordinates to Cartesian
    optimal_bary_coords = np.array(optimal_bary_coords_tuple).reshape(1, 3)
    optimal_X, optimal_Y = to_cartesian(optimal_bary_coords)
    optimal_X = optimal_X[0]
    optimal_Y = optimal_Y[0]

    # 3. Convert ALL Barycentric to Cartesian for plotting
    X, Y = to_cartesian(tri_coords)
    points = np.column_stack((X, Y))

    # 4. Create a Fine Grid for Interpolation
    grid_x, grid_y = np.mgrid[0:1:100j, 0:h:100j]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # 5. Interpolate Z values
    grid_Z = griddata(points, Z, grid_points, method='linear')
    grid_Z = grid_Z.reshape(grid_x.shape)

    # 6. Mask outside the Simplex
    tol = 1e-9
    
    mask = (grid_y < 0 - tol) | \
           (grid_y > np.sqrt(3) * grid_x + tol) | \
           (grid_y > -np.sqrt(3) * (grid_x - 1) + tol)
    
    grid_Z_masked = np.ma.masked_where(mask, grid_Z)

    # 7. Plotting
    plt.figure(figsize=(8, 7))

    # Plot the interpolated data (Heatmap)
    contour = plt.contourf(grid_x, grid_y, grid_Z_masked, levels=50, cmap='viridis', extend='both')
    plt.colorbar(contour, label='Function F value')

    # Draw the Simplex Boundary (Vertices: V1(0,0), V2(1,0), V3(0.5, h))
    simplex_vertices = np.array([[0, 0], [1, 0], [0.5, h], [0, 0]])
    plt.plot(simplex_vertices[:, 0], simplex_vertices[:, 1], 'k-', linewidth=1.5)

    # --- Plot the Optimal Point ---
    plt.plot(
        optimal_X, optimal_Y, 
        marker='*', 
        markersize=15, 
        color='red', 
        linestyle='', 
        label=f'Optimal Point\nZ={optimal_Z_value:.4f}'
    )

    # Add the coordinates to the plot
    coord_text = (
        f"Optimal w:\n"
        f"$w_1={optimal_bary_coords_tuple[0]:.2f}, "
        f"w_2={optimal_bary_coords_tuple[1]:.2f}, "
        f"w_3={optimal_bary_coords_tuple[2]:.2f}$"
    )
    
    # Choose a position near the optimal point, adjusted to avoid overlap with the star
    # Using the bottom right corner for the label if the point is too central/high
    text_x = 0.9
    text_y = h * 0.75
    
    # Check if the text needs to be moved if the star is too close to the label position
    if optimal_X > 0.8 and optimal_Y > h * 0.6:
        text_x = 0.1
        text_y = h * 0.75

    plt.text(
        text_x, text_y, 
        coord_text, 
        color='red', 
        fontsize=10, 
        ha='center', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.5")
    )


    # Add vertex labels (x1, x2, x3 component labels)
    plt.text(0, 0, r'$w_1=1$', ha='center', va='top', fontsize=12, color='black')
    plt.text(1, 0, r'$w_2=1$', ha='center', va='top', fontsize=12, color='black')
    plt.text(0.5, h, r'$w_3=1$', ha='center', va='bottom', fontsize=12, color='black')

    # Final plot settings
    plt.title('Ts for different ws', fontsize=14)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, h + 0.1)
    plt.axis('off')  # Turn off standard Cartesian axes
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
