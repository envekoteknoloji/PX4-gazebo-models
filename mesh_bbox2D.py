import numpy as np
import collada
import math
from collada.scene import Scene, Node, GeometryNode

def get_2d_bounding_rect_vertices(dae_file, projection_plane='xy'):
    """
    Calculate the 2D bounding rectangle of a mesh from a COLLADA (.dae) file
    and return the four corner vertices.

    Args:
        dae_file (str): Path to the COLLADA file
        projection_plane (str): Plane to project onto ('xy', 'xz', or 'yz')

    Returns:
        list: Four vertices of the bounding rectangle as [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    # Load the COLLADA file
    mesh = collada.Collada(dae_file)

    # Initialize bounds with extreme values
    min_coords = [float('inf'), float('inf')]
    max_coords = [float('-inf'), float('-inf')]

    # Set axis indices based on projection plane
    if projection_plane == 'xy':
        axes = [0, 1]  # x and y axes
    elif projection_plane == 'xz':
        axes = [0, 2]  # x and z axes
    elif projection_plane == 'yz':
        axes = [1, 2]  # y and z axes
    else:
        raise ValueError("Projection plane must be 'xy', 'xz', or 'yz'")

    # Process each geometry and apply transformations
    for scene in mesh.scenes:
        process_scene_nodes(scene.nodes, np.identity(4), min_coords, max_coords, axes)

    # Extract the four vertices of the bounding rectangle
    min_x, min_y = min_coords
    max_x, max_y = max_coords

    # Return vertices in clockwise order starting from bottom-left
    vertices = [
        (min_x, min_y),  # Bottom-left
        (max_x, min_y),  # Bottom-right
        (max_x, max_y),  # Top-right
        (min_x, max_y),  # Top-left
    ]

    return vertices

def process_scene_nodes(nodes, transform_matrix, min_coords, max_coords, axes):
    """
    Process each node in the scene, applying transformations and checking vertices.

    Args:
        nodes (list): List of scene nodes
        transform_matrix (numpy.ndarray): Current transformation matrix
        min_coords (list): Minimum coordinates [x, y]
        max_coords (list): Maximum coordinates [x, y]
        axes (list): Indices of axes to use for projection
    """
    for node in nodes:
        # Apply node's transformation matrix if available
        current_transform = transform_matrix.copy()
        if hasattr(node, 'matrix') and node.matrix is not None:
            current_transform = np.dot(current_transform, node.matrix)

        # Process geometry nodes
        if isinstance(node, GeometryNode):
            process_geometry(node.geometry, current_transform, min_coords, max_coords, axes)

        # Process child nodes recursively
        if hasattr(node, 'children') and node.children:
            process_scene_nodes(node.children, current_transform, min_coords, max_coords, axes)

def process_geometry(geometry, transform_matrix, min_coords, max_coords, axes):
    """
    Process geometry, applying transformations to vertices and updating bounds.

    Args:
        geometry (collada.geometry.Geometry): Geometry object
        transform_matrix (numpy.ndarray): Transformation matrix to apply
        min_coords (list): Minimum coordinates [x, y]
        max_coords (list): Maximum coordinates [x, y]
        axes (list): Indices of axes to use for projection
    """
    for primitive in geometry.primitives:
        # Get vertex data
        vertex_data = primitive.vertex

        # Apply transformation to each vertex
        for i in range(len(vertex_data)):
            # Convert vertex to homogeneous coordinates (add 1 for w)
            homogeneous_vertex = np.append(vertex_data[i], 1.0)

            # Apply transformation matrix
            transformed_vertex = np.dot(transform_matrix, homogeneous_vertex)

            # Convert back to 3D coordinates (divide by w if necessary)
            if transformed_vertex[3] != 0:
                transformed_vertex = transformed_vertex[:3] / transformed_vertex[3]
            else:
                transformed_vertex = transformed_vertex[:3]

            # Update bounds based on projected coordinates
            min_coords[0] = min(min_coords[0], transformed_vertex[axes[0]])
            min_coords[1] = min(min_coords[1], transformed_vertex[axes[1]])
            max_coords[0] = max(max_coords[0], transformed_vertex[axes[0]])
            max_coords[1] = max(max_coords[1], transformed_vertex[axes[1]])

# Example usage
if __name__ == "__main__":
    file_path = "/home/user/workspace/PX4-Autopilot/Tools/simulation/gz/terrains/tomek1/media/tomek1.dae"
    bounding_rect_vertices = get_2d_bounding_rect_vertices(file_path)

    print("2D Bounding Rectangle Vertices:")
    for i, vertex in enumerate(bounding_rect_vertices):
        print(f"Vertex {i+1}: ({vertex[0]:.4f}, {vertex[1]:.4f})")

    # Calculate width and height
    width = bounding_rect_vertices[1][0] - bounding_rect_vertices[0][0]
    height = bounding_rect_vertices[2][1] - bounding_rect_vertices[1][1]
    print(f"\nWidth: {width:.4f}")
    print(f"Height: {height:.4f}")

    # Calculate center point
    center_x = (bounding_rect_vertices[0][0] + bounding_rect_vertices[2][0]) / 2
    center_y = (bounding_rect_vertices[0][1] + bounding_rect_vertices[2][1]) / 2
    print(f"Center: ({center_x:.4f}, {center_y:.4f})")
