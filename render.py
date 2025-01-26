from DataExtract.extract_stairs_data import sample_stairs_data as original
from DataExtract.extract_stairs_data import plot_matrix, plot_matrix_3d
import main
from objects import SimulationSettings, StairsUsage, Stairs, Material, Environment
import sliding_distance as sd

def heightmap_to_obj(heightmap, output_path, cell_size=0.01):
    """
    Convert a 2D heightmap into a solid OBJ mesh with:
      - top surface from heightmap
      - flat bottom at z=0
      - vertical side walls
    """

    # 1) Basic dimensions
    rows = len(heightmap)
    cols = len(heightmap[0])
    
    # 2) Create list of vertices for top surface
    #    We'll store them in row-major order.
    #    Vertex indices in OBJ are 1-based, so we will handle that offset later.
    top_vertices = []
    for i in range(rows):
        for j in range(cols):
            x = j * cell_size
            y = i * cell_size
            z = heightmap[i][j]
            top_vertices.append((x, y, z))
    
    # 3) Create list of vertices for bottom surface (z=0)
    bottom_vertices = []
    for i in range(rows):
        for j in range(cols):
            x = j * cell_size
            y = i * cell_size
            z = 0.0
            bottom_vertices.append((x, y, z))
    
    # We'll combine all vertices into one big list:
    #    [ top_surface_vertices..., bottom_surface_vertices... ]
    all_vertices = top_vertices + bottom_vertices
    
    # Helper function to get the 1-based OBJ index of top or bottom vertex (i,j)
    def top_index(i, j):
        return i * cols + j + 1  # +1 because OBJ vertex indexing starts at 1
    def bottom_index(i, j):
        # bottom vertices come after the top in our array
        return rows * cols + (i * cols + j + 1)
    
    # 4) Create faces for the top surface
    # We can triangulate each cell as two triangles:
    #   (i,   j), (i,   j+1), (i+1, j)
    #   (i+1, j), (i,   j+1), (i+1, j+1)
    top_faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # 2 triangles forming a square
            # Vertex order is typically recommended to be counterclockwise
            v1 = top_index(i, j)
            v2 = top_index(i, j + 1)
            v3 = top_index(i + 1, j)
            v4 = top_index(i + 1, j + 1)
            
            # first triangle
            top_faces.append((v1, v2, v3))
            # second triangle
            top_faces.append((v3, v2, v4))
    
    # 5) Create faces for the bottom surface
    # Triangulate in the same manner, but note that the bottom is reversed 
    # or you can keep the same orientation if you want consistent normals.
    bottom_faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = bottom_index(i, j)
            v2 = bottom_index(i, j + 1)
            v3 = bottom_index(i + 1, j)
            v4 = bottom_index(i + 1, j + 1)
            
            # For the bottom, we might invert the winding for the normal direction.
            bottom_faces.append((v1, v3, v2))
            bottom_faces.append((v3, v4, v2))
    
    # 6) Create faces for the sides
    # The perimeter has 4 edges, but we only need to iterate around the boundary
    # row=0, row=rows-1, col=0, col=cols-1
    side_faces = []
    
    # -- top/bottom boundary indices:
    # top_index(i, j), bottom_index(i, j)
    # we connect these with two triangles or a quad.

    # 6a) Front edge: i=0, for j in [0..cols-2]
    i = 0
    for j in range(cols - 1):
        top_v1 = top_index(i, j)
        top_v2 = top_index(i, j+1)
        bot_v1 = bottom_index(i, j)
        bot_v2 = bottom_index(i, j+1)
        # Triangulate the quad: top_v1 -> top_v2 -> bot_v1 -> bot_v2
        side_faces.append((top_v1, bot_v1, top_v2))
        side_faces.append((top_v2, bot_v1, bot_v2))
    
    # 6b) Back edge: i=rows-1, for j in [0..cols-2]
    i = rows - 1
    for j in range(cols - 1):
        top_v1 = top_index(i, j)
        top_v2 = top_index(i, j+1)
        bot_v1 = bottom_index(i, j)
        bot_v2 = bottom_index(i, j+1)
        # Triangulate the quad
        side_faces.append((top_v1, top_v2, bot_v1))
        side_faces.append((top_v2, bot_v2, bot_v1))
    
    # 6c) Left edge: j=0, for i in [0..rows-2]
    j = 0
    for i in range(rows - 1):
        top_v1 = top_index(i, j)
        top_v2 = top_index(i+1, j)
        bot_v1 = bottom_index(i, j)
        bot_v2 = bottom_index(i+1, j)
        # Triangulate the quad
        side_faces.append((top_v1, top_v2, bot_v1))
        side_faces.append((top_v2, bot_v2, bot_v1))
    
    # 6d) Right edge: j=cols-1, for i in [0..rows-2]
    j = cols - 1
    for i in range(rows - 1):
        top_v1 = top_index(i, j)
        top_v2 = top_index(i+1, j)
        bot_v1 = bottom_index(i, j)
        bot_v2 = bottom_index(i+1, j)
        # Triangulate the quad
        side_faces.append((top_v1, bot_v1, top_v2))
        side_faces.append((top_v2, bot_v1, bot_v2))

    # Combine all faces
    all_faces = top_faces + bottom_faces + side_faces
    
    # 7) Write OBJ file
    with open(output_path, 'w') as f:
        # Write vertices
        for vx, vy, vz in all_vertices:
            f.write(f"v {vx} {vy} {vz}\n")
        
        # Optionally, we can write normals (vn) or texture coords (vt)
        # but not strictly required for a simple solid mesh.

        # Write faces
        for face in all_faces:
            # Face is (v1, v2, v3) -> "f v1 v2 v3"
            f.write("f {} {} {}\n".format(*face))

def plot_matrix(matrix):
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Matrix Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def plot_matrix_3d(matrix):

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(matrix[1],matrix[2],matrix[0], cmap='viridis')

    # Set labels
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Height')

    # Show the plot
    plt.show()

def render(matrix,output=False):
    #plot_matrix(matrix)
    plot_matrix_3d(matrix)
    if output:
        output_file = "stairs.obj"
        heightmap_to_obj(matrix, output_file, cell_size=0.01)
        print(f"OBJ file saved to: {output_file}")
    
if __name__ == "__main__":
    model = SimulationSettings(1, Environment(1, 1), Stairs(200, 50, 50, Material(1, sd.H_rubber_kPa, sd.sliding_distance)), StairsUsage(1, 1, False))
    render(model.final_shape[0,0])
    

