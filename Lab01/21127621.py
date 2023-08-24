import numpy as np
from collections import deque
import heapq

def DFS(matrix, start, end):
    """
    DFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited: dictionary
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    path = []
    visited = {}

    # Define a recursive function to visit each node
    def visit(node):
        # Mark the node as visited
        visited[node] = None

        # If the node is the end node, return True to terminate recursion
        if node == end:
            return True

        # Visit each neighbor of the current node
        for neighbor, is_connected in enumerate(matrix[node]):
            if is_connected and neighbor not in visited:
                visited[neighbor] = node
                if visit(neighbor):
                    return True

        # If end node is not found in this branch, backtrack and remove current node from the path
        del visited[node]
        return False

    # Start visiting nodes from the start node
    visit(start)

    # Reconstruct the path from start to end using the visited dictionary
    node = end
    while node is not None:
        path.insert(0, node)
        node = visited.get(node)

    return visited, path

def BFS(matrix, start, end):
    """
    BFS algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    # initialize an empty queue and a dictionary to keep track of visited nodes
    queue = deque()
    visited = {}

    # add the starting node to the queue and mark it as visited
    queue.append(start)
    visited[start] = None

    # loop until the queue is empty
    while queue:
        # remove the first node from the queue
        current_node = queue.popleft()

        # if we've reached the end node, construct the path and return it
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = visited[current_node]
            path.reverse()
            return visited, path

        # for each neighbor of the current node that hasn't been visited yet,
        # add it to the queue and mark it as visited with the current node as its parent
        for neighbor, weight in enumerate(matrix[current_node]):
            if weight != 0 and neighbor not in visited:
                queue.append(neighbor)
                visited[neighbor] = current_node

    # if we've explored the whole graph and haven't found the end node, return None
    return visited, None

def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    path = []
    visited = {}
    queue = [(0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == end:
            path.append(node)
            while node != start:
                path.append(visited[node])
                node = visited[node]
            path.reverse()
            return visited, path
        if node not in visited:
            visited[node] = None
            for i, j in enumerate(matrix[node]):
                if j != 0:
                    heapq.heappush(queue, (cost + j, i))
                    if i not in visited:
                        visited[i] = node
    return visited, path

def euclidean_distance(pos1, pos2):
    """Euclidean distance between two points"""
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: eclid distance based positions parameter
     Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # Initialize the start node
    pq = [(euclidean_distance(pos[start], pos[end]), start, [start])]  # priority queue, heurisitc + start node + path to start node
    visited = {start: None}  # dictionary to store visited nodes and their previous nodes

    while pq:
        _, node, path = heapq.heappop(pq)  # get the node with the lowest f-score
        if node == end:
            return visited, path  # path found

        # explore the neighbors of the current node
        for neighbor in np.where(matrix[node, :] > 0)[0]:
            # calculate the new f-score for the neighbor
            f_score = euclidean_distance(pos[neighbor], pos[end]) + euclidean_distance(pos[neighbor], pos[start]) + len(path)
            # add the neighbor to the priority queue
            heapq.heappush(pq, (f_score, neighbor, path + [neighbor]))
            # mark the neighbor as visited
            if neighbor not in visited:
                visited[neighbor] = node

    return visited, []  # no path found


def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm 
    heuristic : edge weights
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
   
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # Initialize variables
    path = []
    visited = {}
    queue = [(start, 0)]
    
    # While the queue is not empty
    while queue:
        # Sort the queue based on the heuristic
        queue.sort(key=lambda x: x[1])
        
        # Get the next node from the queue
        node, cost = queue.pop(0)
        
        # If the node has already been visited, skip it
        if node in visited:
            continue
        
        # Add the node to the visited dictionary
        visited[node] = None
        
        # If the node is the end node, construct the path and return it
        if node == end:
            path.append(end)
            while visited[path[-1]] != start:
                path.append(visited[path[-1]])
            path.append(start)
            path.reverse()
            return visited, path
        
        # Add the node's neighbors to the queue
        for neighbor, weight in enumerate(matrix[node]):
            if weight > 0:
                queue.append((neighbor, weight))
    
    # If no path was found, return None
    return None, None

