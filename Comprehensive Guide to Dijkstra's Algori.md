## Comprehensive Guide to Dijkstra's Algorithm with Presentation Example 

![alt text](AD87C7B7-1406-4FA8-9702-19C32E6EA8E9.jpeg)

### 1. Presentation Example
Consider the following weighted graph:

This represents the same graph as the one in the previous example, but with a different representation of the edges.

```mermaid
graph TD
    0((0)) -->|4| 1((1))
    0 -->|7| 6((6))
    1 -->|9| 2((2))
    1 -->|11| 6((6))
    1 -->|20| 7((7))
    2 -->|6| 3((3))
    2 -->|2| 4((4))
    3 -->|5| 5((5))
    3 -->|10| 4((4))
    4 -->|15| 5((5))
    4 -->|1| 7((7))
    4 -->|5| 8((8))
    5 -->|12| 8((8))
    6 -->|1| 7((7))
    7 -->|3| 8((8))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5,6,7,8 fill:#bbf,stroke:#333,stroke-width:2px
```

## Dijkstra's Algorithm: Comparative Analysis

### Graph Structures

#### Example 1 (Original Graph) Nodes 0-5 
```mermaid
graph TD
    0((0)) -->|7| 1((1))
    0 -->|9| 2((2))
    0 -->|14| 5((5))
    1 -->|10| 2((2))
    1 -->|15| 3((3))
    2 -->|11| 3((3))
    2 -->|2| 5((5))
    3 -->|6| 4((4))
    4 -->|9| 5((5))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5 fill:#bbf,stroke:#333,stroke-width:2px
```

#### Example 2 (Extended Graph) Nodes 0-8
```mermaid
graph TD
    0((0)) -->|4| 1((1))
    0 -->|7| 6((6))
    1 -->|9| 2((2))
    1 -->|11| 6((6))
    1 -->|20| 7((7))
    2 -->|6| 3((3))
    2 -->|2| 4((4))
    3 -->|5| 5((5))
    3 -->|10| 4((4))
    4 -->|15| 5((5))
    4 -->|1| 7((7))
    4 -->|5| 8((8))
    5 -->|12| 8((8))
    6 -->|1| 7((7))
    7 -->|3| 8((8))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5,6,7,8 fill:#bbf,stroke:#333,stroke-width:2px
```

### Algorithm Processing Steps
```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant D as Distance Array
    participant P as Path Tracker
    
    Note over PQ,P: Algorithm Initialization
    PQ->>D: Set initial distances (∞)
    D->>D: Set source distance (0)
    
    loop While Priority Queue Not Empty
        PQ->>D: Get node with min distance
        D->>P: Update shortest paths
        P->>PQ: Add unvisited neighbors
    end
```
### Detailed Analysis for Example 1 (0 to 5)
```mermaid
sequenceDiagram
    participant S as Start_0
    participant P as Process
    participant E as End_5
    
    Note over S: Initial state
    S->>P: Distance[0] = 0
    Note over P: Visit node 0
    P->>P: Update neighbors
    Note right of P: 1: min(∞, 7) = 7
    Note right of P: 2: min(∞, 9) = 9
    Note right of P: 5: min(∞, 14) = 14
    
    P->>P: Visit node 1
    Note right of P: 2: min(9, 7+10) = 9
    Note right of P: 3: min(∞, 7+15) = 22
    
    P->>P: Visit node 2
    Note right of P: 5: min(14, 9+2) = 11
    
    P->>E: Final shortest path
    Note over E: Distance[5] = 11

```
#### Priority Queue State Transitions
```java
// Example progression of priority queue states
Step 1: [(0,0)]
Step 2: [(1,4), (6,7)]
Step 3: [(6,7), (2,13), (7,8)]
Step 4: [(7,8), (2,13)]
Step 5: [(8,11), (2,13)]
```

#### Path Tracking
```java
Map<Integer, Integer> previousNodes = new HashMap<>();
// After algorithm completion:
// 8 → 7 → 6 → 0
```

### Key Findings
Example 1 (0 to 5):
- Shortest path: 0 → 2 → 5
- Total distance: 11 units
- Key decision point: Choosing path through node 2 instead of direct path (14)

Example 2 (0 to 8):
- Shortest path: 0 → 6 → 7 → 8
- Total distance: 11 units
- Key decision point: Using path through nodes 6 and 7 instead of longer alternatives


This practical example demonstrates how Dijkstra's algorithm:
- Always selects the minimum current distance node
- Updates distances through edge relaxation
- Maintains an optimal substructure
- Guarantees the shortest path upon completion

### Sample Path Analysis: 0 to 8 using a sequence diagram to emulate the call stack frames using shapes. 
Let's trace how Dijkstra's algorithm finds the shortest path from node 0 to node 8.

```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant D as Distance Array
    participant P as Path Tracking
    Note over PQ,P: Initial State
    Note over D: Node 0: 0<br/>All others: ∞
    rect rgb(200, 255, 200)
        Note over PQ: Step 1 - Process Node 0
        PQ->>D: Update neighbors
        Note over D: Node 1: 4<br/>Node 6: 7
    end
    rect rgb(200, 220, 255)
        Note over PQ: Step 2 - Process Node 1
        PQ->>D: Update neighbors
        Note over D: Node 2: 13<br/>Node 6: 7 (unchanged)<br/>Node 7: 24
    end
    rect rgb(255, 220, 220)
        Note over PQ: Step 3 - Process Node 6
        PQ->>D: Update neighbors
        Note over D: Node 7: 8
    end
    rect rgb(220, 255, 220)
        Note over PQ: Step 4 - Process Node 7
        PQ->>D: Update neighbors
        Note over D: Node 8: 11
    end
```

### Shortest Path Discovery
```mermaid
graph TD
    subgraph Final Path
        0((0)) -->|7| 6((6))
        6 -->|1| 7((7))
        7 -->|3| 8((8))
        
        style 0 fill:#f9f,stroke:#333,stroke-width:4px
        style 6,7,8 fill:#9f9,stroke:#333,stroke-width:4px
    end

    subgraph Distance Table
        DT[Final Distances:<br/>0→8: 11 units<br/>Path: 0→6→7→8]
    end
```

### 2. Algorithm Implementation
```java
public class DijkstraExample {
    public static Map<Integer, Integer> findShortestPath(WeightedGraph graph, int start) {
        Map<Integer, Integer> distances = new HashMap<>();
        PriorityQueue<Node> pq = new PriorityQueue<>(
            (a, b) -> distances.get(a) - distances.get(b)
        );
        
        // Initialize distances
        for (int node = 0; node < 9; node++) {
            distances.put(node, Integer.MAX_VALUE);
        }
        distances.put(start, 0);
        pq.offer(new Node(start));
        while (!pq.isEmpty()) {
            Node current = pq.poll();
            
            for (Edge edge : graph.getEdges(current.id)) {
                int newDist = distances.get(current.id) + edge.weight;
                if (newDist < distances.get(edge.dest)) {
                    distances.put(edge.dest, newDist);
                    pq.offer(new Node(edge.dest));
                }
            }
        }
        return distances;
    }
}
```

### 3. Step-by-Step Path Discovery
Key Decision Points in Example Graph:
Initial Choice at Node 0:

Option 1: 0→1 (weight: 4)

Option 2: 0→6 (weight: 7) ✓

From Node 6:

Direct path to 7 (weight: 1) ✓

From Node 7:

Direct path to 8 (weight: 3) ✓

### Alternative Paths Analysis:
```mermaid
graph TD
    subgraph Alternative Routes
        P1[Path 1: 0→1→2→4→8<br/>Cost: 20]
        P2[Path 2: 0→6→7→8<br/>Cost: 11]
        P3[Path 3: 0→1→6→7→8<br/>Cost: 19]
    end
```

### 4. Implementation Details


### Implementation Considerations

#### Priority Queue Maintenance is Crucial for Efficiency

Efficient management of the priority queue is essential for the performance of Dijkstra's algorithm. The priority queue is used to select the next node with the smallest tentative distance:

- **Data Structure Choice**: Using a binary heap allows for O(log V) time complexity for insertion and extraction operations, making it suitable for most practical applications.
- **Update Operations**: When a node's distance is updated, the priority queue must reflect this change to maintain correct ordering.
- **Optimization**: Minimizing the number of decrease-key operations can improve efficiency, as they can be costly depending on the implementation.

Proper priority queue maintenance ensures the algorithm operates efficiently, especially on large graphs.

#### Path Reconstruction Requires Tracking Previous Nodes

To determine the actual shortest path, it's necessary to keep track of the previous node for each node visited:

- **Previous Node Map**: Maintain a mapping from each node to its predecessor on the shortest path.
- **Updating Predecessors**: Whenever a node's tentative distance is updated via edge relaxation, also update its predecessor.
- **Path Reconstruction**: After the algorithm completes, reconstruct the path by backtracking from the destination node using the predecessor map.

This tracking allows for the recovery of the shortest path, not just the shortest distance.

#### Edge Relaxation Determines Optimal Path Selection

Edge relaxation is the process of adjusting the tentative distances to neighboring nodes:

- **Relaxation Step**: For each adjacent node, calculate the new tentative distance through the current node.
- **Distance Comparison**: If the calculated distance is less than the existing tentative distance, update it.
- **Consistent Application**: Repeatedly applying edge relaxation ensures all nodes are evaluated for potential shorter paths.

Edge relaxation is fundamental to the algorithm's ability to find the shortest paths accurately.

#### Time Complexity: O((V + E) log V) with Binary Heap

The algorithm's time complexity depends on the data structures used:

- **Vertices (V)**: Each vertex may be inserted into or extracted from the priority queue, operations that take O(log V) time.
- **Edges (E)**: Each edge may cause a decrease-key operation in the priority queue during relaxation.
- **Overall Complexity**: Combining these, the total time complexity is O((V + E) log V) when a binary heap is used for the priority queue.

This complexity makes the algorithm efficient for graphs where E is not significantly larger than V.

#### Space Complexity: O(V + E) with Adjacency List

The algorithm's space requirements include:

- **Graph Storage**: Using an adjacency list requires O(V + E) space to store all vertices and edges.
- **Auxiliary Data Structures**: Additional space for the priority queue, distance map, and predecessor map, each requiring O(V) space.
- **Total Space Usage**: Combining these, the space complexity remains O(V + E), which is efficient for sparse graphs.

Efficient space utilization is crucial for handling large graphs without excessive memory consumption.

### Implementation Considerations
- Priority Queue maintenance is crucial for efficiency
- Path reconstruction requires tracking previous nodes
- Edge relaxation determines optimal path selection
- Time complexity: O((V + E) log V) with binary heap
- Space complexity: O(V + E) with adjacency list


## Comprehensive Guide to Dijkstra's Algorithm: Theory, Implementation, and Applications
The main idea behind Dijkstra's Algorithm is to efficiently find the shortest path between a starting node (source node) and every other node in a graph with non-negative edge weights.

It achieves this by using a greedy approach:

### Key Findings of ***Shortest Path*** in given Graph
Example 1 (0 to 5):
- Shortest path: 0 → 2 → 5
- Total distance: 11 units
- Key decision point: Choosing path through node 2 instead of direct path (14)


Example 2 (0 to 8):
- Shortest path: 0 → 6 → 7 → 8
- Total distance: 11 units
- Key decision point: Using path through nodes 6 and 7 instead of longer alternatives


# Comprehensive Guide to Dijkstra's Algorithm Implementation

## 1. Algorithm Overview
Dijkstra's algorithm finds the shortest paths between nodes in a weighted graph, which may represent networks, roads, or any weighted relationships.

### Core Components

```mermaid
graph TD
    subgraph Components
        PQ[Priority Queue] --> |Maintains Order| NP[Node Processing]
        NP --> |Updates| DT[Distance Tracking]
        DT --> |Records| PT[Path Tracking]
        PT --> |Enables| PR[Path Reconstruction]
    end
```

## 2. Implementation Structure

### Base Graph Structure
```java
public class WeightedGraph<T> {
    private Map<T, Map<T, Integer>> adjacencyList;
    
    public void addEdge(T source, T destination, int weight) {
        adjacencyList.computeIfAbsent(source, k -> new HashMap<>())
                    .put(destination, weight);
    }
}
```

### Priority Queue Management
```java
PriorityQueue<Node> pq = new PriorityQueue<>((a, b) -> 
    distances.get(a) - distances.get(b));
```

## 3. Algorithm Execution Process

```mermaid
sequenceDiagram
    participant Init as Initialization
    participant PQ as Priority Queue
    participant Process as Processing
    participant Path as Path Tracking
    
    Init->>PQ: Set source distance to 0
    Init->>PQ: Set all other distances to ∞
    
    loop While PQ not empty
        PQ->>Process: Get minimum distance node
        Process->>Process: Process all neighbors
        Process->>Path: Update distances and paths
        Process->>PQ: Add updated nodes
    end
    Path->>Path: Reconstruct shortest path
```

## 4. Example Walkthrough of Processing Steps and Path Reconstruction

### Processing Steps
Initialize distances:

Node 0: 0

All other nodes: ∞

Priority Queue States:

```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant D as Distances
    
    Note over PQ: Initial: [(0,0)]
    PQ->>D: Process 0
    Note over D: Update neighbors
    Note over PQ: Next: [(1,7),(2,9),(5,14)]
    PQ->>D: Process 1
    Note over PQ: Next: [(2,9),(5,11),(3,22)]
```

## 5. Path Reconstruction

```mermaid
graph TD
    subgraph Path Tracking Structure
        direction LR
        N0[Node 0] -->|"prev[1]=0"| N1[Node 1]
        N1 -->|"prev[2]=1"| N2[Node 2]
        N2 -->|"prev[5]=2"| N5[Node 5]
    end
```

### Implementation:
```java
private List<Node> reconstructPath(Node destination) {
    List<Node> path = new ArrayList<>();
    Node current = destination;
    
    while (current != null) {
        path.add(0, current);
        current = previousNodes.get(current);
    }
    return path;
}
```

## 6. Edge Relaxation Process

```mermaid
graph LR
    subgraph Edge Relaxation
        C((Current)) -->|"weight w"| N((Neighbor))
        style C fill:#f9f
        style N fill:#bbf
    end
    
    subgraph Decision
        direction TB
        D{{"If current.distance + w < neighbor.distance"}}
        U[Update neighbor distance]
        P[Update previous node]
    end
```

## 7. Performance Characteristics

### Time Complexity
With Binary Heap: O((V + E) log V)
With Fibonacci Heap: O(E + V log V)

### Space Complexity
Adjacency List: O(V + E)
Priority Queue: O(V)
Distance/Previous Arrays: O(V)

## 8. Optimization Techniques

### Priority Queue Optimization
```java
// Use offers instead of updates
if (newDistance < distances.get(neighbor)) {
    pq.offer(new Node(neighbor, newDistance));
    distances.put(neighbor, newDistance);
}
```

### Memory Management
```java
// Use primitive arrays for small graphs
int[] distances = new int[vertices];
int[] previous = new int[vertices];
```

### Early Termination
```java
if (current.equals(destination)) {
    break; // Found shortest path to destination
}
```

## 9. Best Practices

### Input Validation
```java
public void validateInput(Graph graph) {
    if (graph == null || graph.isEmpty()) {
        throw new IllegalArgumentException("Invalid graph");
    }
}
```

### Edge Case Handling
```java
if (source.equals(destination)) {
    return Collections.singletonList(source);
}
```

### Negative Weight Detection
```java
for (Edge edge : graph.getEdges()) {
    if (edge.weight < 0) {
        throw new IllegalArgumentException("Negative weights not supported");
    }
}
```

## 10. Common Applications
- Network Routing
- Social Networks
- Geographic Maps
- Game AI Pathfinding
- Resource Distribution

## 11. Testing Strategies
```java
@Test
public void testShortestPath() {
    WeightedGraph graph = new WeightedGraph();
    // Add test edges
    List<Node> path = dijkstra(graph, source, destination);
    assertNotNull(path);
    assertEquals(expectedDistance, getPathDistance(path));
}
```


## 1. Priority Queue Maintenance

```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant VN as Visited Nodes
    participant DS as Distance State
    Note over PQ: Initial Queue State<br/>[(0,0)]
    
    PQ->>DS: Poll Node 0
    Note over DS: Process neighbors of 0<br/>Update distances
    DS->>PQ: Add neighbors with priorities:<br/>(1,7), (2,9), (5,14)
    
    Note over PQ: Queue State<br/>[(1,7), (2,9), (5,14)]
    
    PQ->>DS: Poll Node 1 (lowest distance)
    Note over DS: Process neighbors of 1<br/>Update distances
    DS->>PQ: Add/Update neighbors:<br/>(2,17), (3,22)
    
    Note over PQ: Queue State<br/>[(2,9), (5,14), (3,22)]
```

### Priority Queue Efficiency:
```java
// Binary Heap Implementation
PriorityQueue<Node> pq = new PriorityQueue<>((a, b) -> 
    distances.get(a) - distances.get(b));

// Key Operations:
// O(log n) insertion
pq.offer(new Node(vertex, distance));

// O(log n) removal of minimum element
Node current = pq.poll();

// O(1) peek at minimum element
Node next = pq.peek();
```

## 2. Path Reconstruction

```mermaid
graph TD
    subgraph Path Tracking
        0((0)) -->|"prev[1]=0"| 1((1))
        1 -->|"prev[2]=1"| 2((2))
        2 -->|"prev[5]=2"| 5((5))
        
        style 0 fill:#f9f,stroke:#333
        style 1,2,5 fill:#bbf,stroke:#333
    end

    subgraph Previous Nodes Map
        PM[Previous Nodes HashMap:<br/>5 → 2<br/>2 → 1<br/>1 → 0]
    end
```

### Path Tracking Implementation:
```java
private Map<Node, Node> previousNodes = new HashMap<>();

private void trackPath(Node current, Node neighbor, int newDistance) {
    if (newDistance < distances.get(neighbor)) {
        distances.put(neighbor, newDistance);
        previousNodes.put(neighbor, current);  // Track the path
    }
}

private List<Node> reconstructPath(Node destination) {
    List<Node> path = new ArrayList<>();
    Node current = destination;
    
    while (current != null) {
        path.add(0, current);  // Add to front of list
        current = previousNodes.get(current);  // Move to previous node
    }
    return path;
}
```

## 3. Edge Relaxation

```mermaid
sequenceDiagram
    participant C as Current Node
    participant N as Neighbor Node
    participant D as Distance Array
    
    Note over C,N: Edge Relaxation Process
    C->>N: Check edge weight (w)
    Note over C,D: Current distance = d
    Note over N,D: Neighbor distance = n
    
    alt d + w < n
        N->>D: Update distance
        Note over D: New distance = d + w
        N->>D: Update previous node
    else d + w >= n
        Note over D: Keep current distance
    end
```

### Edge Relaxation Implementation:
```java
private void relaxEdge(Node current, Node neighbor, int weight) {
    int currentDistance = distances.get(current);
    int neighborDistance = distances.get(neighbor);
    int newDistance = currentDistance + weight;
    
    if (newDistance < neighborDistance) {
        // Update distance
        distances.put(neighbor, newDistance);
        // Update previous node
        previousNodes.put(neighbor, current);
        // Add to priority queue with new priority
        pq.offer(neighbor);
    }
}
```

```mermaid
graph TD
    subgraph "Processing State"
        PQ["Priority Queue\nCurrent: (2,9)"]
        DM["Distance Map\n0:0, 1:7, 2:9, 3:∞, 4:∞, 5:14"]
        PM["Previous Map\n1←0, 2←0, 5←0"]
    end

    subgraph "Edge Relaxation"
        direction LR
        C((2)) -->|"weight=2"| N((5))
        Note["Old distance to 5: 14\nNew possible distance: 11\n11 < 14, so update"]
    end
    
    subgraph "Updated State"
        PQ2["Priority Queue\nNext: (5,11)"]
        DM2["Distance Map\n0:0, 1:7, 2:9, 3:∞, 4:∞, 5:11"]
        PM2["Previous Map\n1←0, 2←0, 5←2"]
    end
```

### Key Optimizations:
- Priority Queue Management
  - Maintain heap property after every update
  - Only add nodes when their distance improves
  - Use decrease-key operation when available

- Path Tracking
  - Only update previous node when finding shorter path
  - Store only essential path information
  - Reconstruct path only when needed

- Edge Relaxation
  - Process each edge at most once
  - Early termination when possible
  - Skip relaxation if current node's distance is infinity

## Comprehensive Guide to Dijkstra's Algorithm Implementation

### Overview
Dijkstra's Algorithm is a fundamental graph algorithm used to find the shortest paths between nodes in a weighted graph. This document provides a detailed exploration of its implementation, with a focus on a generic Java-based approach that supports multiple data types.

### Core Components

#### Graph Representation

1. Interface-based Design
The implementation uses a flexible interface-based approach:

```java
public interface Graph<T> {
    void addVertex(T vertex);
    void addEdge(T source, T destination, int weight);
    void addEdge(T source, T destination);
    Set<T> getNeighbors(T vertex);
    // ... other methods
}
```

2. Implementation Options

#### Adjacency List Implementation
```java
public class GraphList<T> implements Graph<T> {
    private final Map<T, Set<T>> adjacencyList;
    // Provides O(|V| + |E|) space complexity
    // Efficient for sparse graphs
}
```

#### Adjacency Matrix Implementation
```java
public class GraphMatrix<T extends Integer> implements Graph<T> {
    private final int[][] adjacencyMatrix;
    // Provides O(|V|²) space complexity
    // Efficient for dense graphs
}
```

#### Generic Node Structure
The implementation uses a generic node class to support various data types:

```java
public class GenericNode<T> {
    private T data;
    private List<GenericNode<T>> neighbors;
    private GenericNode<T> left;
    private GenericNode<T> right;
    
    // Constructor and methods
}
```

### Algorithm Implementation

#### Core Algorithm Structure
```java
public class DijkstraGeneric<T> {
    public Map<GenericNode<T>, Integer> findShortestPath(
            Map<GenericNode<T>, Map<GenericNode<T>, Integer>> graph,
            GenericNode<T> source,
            GenericNode<T> destination) {
        
        Map<GenericNode<T>, Integer> distances = new HashMap<>();
        Map<GenericNode<T>, GenericNode<T>> previousNodes = new HashMap<>();
        PriorityQueue<GenericNode<T>> queue = new PriorityQueue<>(
            (a, b) -> distances.getOrDefault(a, Integer.MAX_VALUE)
                    - distances.getOrDefault(b, Integer.MAX_VALUE)
        );
        
        // Initialize distances
        initializeDistances(graph, source, distances);
        
        // Process nodes
        processNodes(graph, distances, previousNodes, queue, destination);
        
        // Reconstruct path
        reconstructPath(previousNodes, destination);
        
        return distances;
    }
}
```

#### Key Components Breakdown

##### Initialization Phase
```java
private void initializeDistances(
        Map<GenericNode<T>, Map<GenericNode<T>, Integer>> graph,
        GenericNode<T> source,
        Map<GenericNode<T>, Integer> distances) {
    
    for (GenericNode<T> node : graph.keySet()) {
        distances.put(node, Integer.MAX_VALUE);
    }
    distances.put(source, 0);
}
```

##### Node Processing
```java
private void processNodes(
        Map<GenericNode<T>, Map<GenericNode<T>, Integer>> graph,
        Map<GenericNode<T>, Integer> distances,
        Map<GenericNode<T>, GenericNode<T>> previousNodes,
        PriorityQueue<GenericNode<T>> queue,
        GenericNode<T> destination) {
    
    while (!queue.isEmpty()) {
        GenericNode<T> current = queue.poll();
        if (current.equals(destination)) break;
        
        processNeighbors(current, graph, distances, previousNodes, queue);
    }
}
```

##### Path Reconstruction
```java
private void reconstructPath(
        Map<GenericNode<T>, GenericNode<T>> previousNodes,
        GenericNode<T> destination) {
    
    List<GenericNode<T>> path = new ArrayList<>();
    GenericNode<T> current = destination;
    
    while (current != null) {
        path.add(0, current);
        current = previousNodes.get(current);
    }
    
    printPath(path);
}
```

### Performance Characteristics

#### Time Complexity
With Binary Heap: O((|V| + |E|) log |V|)
With Fibonacci Heap: O(|E| + |V| log |V|)

#### Space Complexity
Adjacency List: O(|V| + |E|)
Adjacency Matrix: O(|V|²)

### Best Practices and Optimization Techniques

#### Priority Queue Optimization
- Use a binary heap for most cases
- Consider Fibonacci heap for very large graphs

#### Memory Management
- Use adjacency list for sparse graphs
- Use adjacency matrix for dense graphs

#### Edge Case Handling
- Validate input parameters
- Handle disconnected graphs
- Check for negative weights

### Common Pitfalls and Solutions

#### Negative Edge Weights
- Solution: Validate weights during edge addition
- Alternative: Use Bellman-Ford algorithm

#### Infinite Loops
- Solution: Track visited nodes
- Implement cycle detection

#### Memory Overflow
- Solution: Implement iterative deepening
- Use memory-efficient data structures

### Testing Strategies

#### Unit Tests
```java
@Test
public void testShortestPath() {
    GraphList<Integer> graph = new GraphList<>();
    // Add test edges
    assertEquals(expectedDistance, 
        algorithm.findShortestPath(graph, source, destination));
}
```

#### Edge Cases
- Empty graph
- Single node
- Disconnected components
- Maximum possible path length

### Conclusion
This implementation of Dijkstra's Algorithm provides a flexible, generic approach that can be adapted to various use cases while maintaining performance and reliability. The use of generics allows for versatile application across different data types, while the interface-based design ensures extensibility and maintainability.

## Comprehensive Guide to Dijkstra's Algorithm with Presentation Example 

![alt text](AD87C7B7-1406-4FA8-9702-19C32E6EA8E9.jpeg)


### 1. Presentation Example
Consider the following weighted graph:

This represets the same graph as the one in the previous example, but with a different representation of the edges.

```mermaid
graph TD
    0((0)) -->|4| 1((1))
    0 -->|7| 6((6))
    1 -->|9| 2((2))
    1 -->|11| 6((6))
    1 -->|20| 7((7))
    2 -->|6| 3((3))
    2 -->|2| 4((4))
    3 -->|5| 5((5))
    3 -->|10| 4((4))
    4 -->|15| 5((5))
    4 -->|1| 7((7))
    4 -->|5| 8((8))
    5 -->|12| 8((8))
    6 -->|1| 7((7))
    7 -->|3| 8((8))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5,6,7,8 fill:#bbf,stroke:#333,stroke-width:2px
```

### Sample Path Analysis: 0 to 8
Let's trace how Dijkstra's algorithm finds the shortest path from node 0 to node 8.

```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant D as Distance Array
    participant P as Path Tracking
    Note over PQ,P: Initial State
    Note over D: Node 0: 0<br/>All others: ∞
    rect rgb(200, 255, 200)
        Note over PQ: Step 1 - Process Node 0
        PQ->>D: Update neighbors
        Note over D: Node 1: 4<br/>Node 6: 7
    end
    rect rgb(200, 220, 255)
        Note over PQ: Step 2 - Process Node 1
        PQ->>D: Update neighbors
        Note over D: Node 2: 13<br/>Node 6: 7 (unchanged)<br/>Node 7: 24
    end
    rect rgb(255, 220, 220)
        Note over PQ: Step 3 - Process Node 6
        PQ->>D: Update neighbors
        Note over D: Node 7: 8
    end
    rect rgb(220, 255, 220)
        Note over PQ: Step 4 - Process Node 7
        PQ->>D: Update neighbors
        Note over D: Node 8: 11
    end
```

### Shortest Path Discovery
```mermaid
graph TD
    subgraph Final Path
        0((0)) -->|7| 6((6))
        6 -->|1| 7((7))
        7 -->|3| 8((8))
        
        style 0 fill:#f9f,stroke:#333,stroke-width:4px
        style 6,7,8 fill:#9f9,stroke:#333,stroke-width:4px
    end

    subgraph Distance Table
        DT[Final Distances:<br/>0→8: 11 units<br/>Path: 0→6→7→8]
    end
```

### 2. Algorithm Implementation
```java
public class DijkstraExample {
    public static Map<Integer, Integer> findShortestPath(WeightedGraph graph, int start) {
        Map<Integer, Integer> distances = new HashMap<>();
        PriorityQueue<Node> pq = new PriorityQueue<>(
            (a, b) -> distances.get(a) - distances.get(b)
        );
        
        // Initialize distances
        for (int node = 0; node < 9; node++) {
            distances.put(node, Integer.MAX_VALUE);
        }
        distances.put(start, 0);
        pq.offer(new Node(start));
        while (!pq.isEmpty()) {
            Node current = pq.poll();
            
            for (Edge edge : graph.getEdges(current.id)) {
                int newDist = distances.get(current.id) + edge.weight;
                if (newDist < distances.get(edge.dest)) {
                    distances.put(edge.dest, newDist);
                    pq.offer(new Node(edge.dest));
                }
            }
        }
        return distances;
    }
}
```

### 3. Step-by-Step Path Discovery
Key Decision Points in Example Graph:
Initial Choice at Node 0:

Option 1: 0→1 (weight: 4)
Option 2: 0→6 (weight: 7) ✓

From Node 6:

Direct path to 7 (weight: 1) ✓

From Node 7:

Direct path to 8 (weight: 3) ✓

### Alternative Paths Analysis:
```mermaid
graph TD
    subgraph Alternative Routes
        P1[Path 1: 0→1→2→4→8<br/>Cost: 20]
        P2[Path 2: 0→6→7→8<br/>Cost: 11]
        P3[Path 3: 0→1→6→7→8<br/>Cost: 19]
    end
```

### 4. Implementation Details

#### Priority Queue State Transitions
```java
// Example progression of priority queue states
Step 1: [(0,0)]
Step 2: [(1,4), (6,7)]
Step 3: [(6,7), (2,13), (7,24)]
Step 4: [(7,8), (2,13)]
Step 5: [(8,11), (2,13)]
```

#### Path Tracking
```java
Map<Integer, Integer> previousNodes = new HashMap<>();
// After algorithm completion:
// 8 → 7 → 6 → 0
```

This practical example demonstrates how Dijkstra's algorithm:
- Always selects the minimum current distance node
- Updates distances through edge relaxation
- Maintains an optimal substructure
- Guarantees the shortest path upon completion

## Dijkstra's Algorithm: Comparative Analysis

### Graph Structures

#### Example 1 (Original Graph) Nodes 0-5 
```mermaid
graph TD
    0((0)) -->|7| 1((1))
    0 -->|9| 2((2))
    0 -->|14| 5((5))
    1 -->|10| 2((2))
    1 -->|15| 3((3))
    2 -->|11| 3((3))
    2 -->|2| 5((5))
    3 -->|6| 4((4))
    4 -->|9| 5((5))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5 fill:#bbf,stroke:#333,stroke-width:2px
```

#### Example 2 (Extended Graph) Nodes 0-8
```mermaid
graph TD
    0((0)) -->|4| 1((1))
    0 -->|7| 6((6))
    1 -->|9| 2((2))
    1 -->|11| 6((6))
    1 -->|20| 7((7))
    2 -->|6| 3((3))
    2 -->|2| 4((4))
    3 -->|5| 5((5))
    3 -->|10| 4((4))
    4 -->|15| 5((5))
    4 -->|1| 7((7))
    4 -->|5| 8((8))
    5 -->|12| 8((8))
    6 -->|1| 7((7))
    7 -->|3| 8((8))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5,6,7,8 fill:#bbf,stroke:#333,stroke-width:2px
```

### Algorithm Processing Steps
```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant D as Distance Array
    participant P as Path Tracker
    
    Note over PQ,P: Algorithm Initialization
    PQ->>D: Set initial distances (∞)
    D->>D: Set source distance (0)
    
    loop While Priority Queue Not Empty
        PQ->>D: Get node with min distance
        D->>P: Update shortest paths
        P->>PQ: Add unvisited neighbors
    end
```
### Detailed Analysis for Example 1 (0 to 5)
```mermaid
sequenceDiagram
    participant S as Start_0
    participant P as Process
    participant E as End_5
    
    Note over S: Initial state
    S->>P: Distance[0] = 0
    Note over P: Visit node 0
    P->>P: Update neighbors
    Note right of P: 1: min(∞, 7) = 7
    Note right of P: 2: min(∞, 9) = 9
    Note right of P: 5: min(∞, 14) = 14
    
    P->>P: Visit node 1
    Note right of P: 2: min(9, 7+10) = 9
    Note right of P: 3: min(∞, 7+15) = 22
    
    P->>P: Visit node 2
    Note right of P: 5: min(14, 9+2) = 11
    
    P->>E: Final shortest path
    Note over E: Distance[5] = 11

```

### Detailed Analysis for Example 2 (0 to 8)
```mermaid
sequenceDiagram
    participant S as Start_0
    participant P as Process
    participant E as End_8
    
    Note over S: Initial state
    S->>P: Distance[0] = 0
    Note over P: Visit node 0
    P->>P: Update neighbors
    Note right of P: 1: min(∞, 4) = 4
    Note right of P: 6: min(∞, 7) = 7
    
    P->>P: Visit node 1
    Note right of P: 2: min(∞, 4+9) = 13
    Note right of P: 6: min(7, 4+11) = 7
    Note right of P: 7: min(∞, 4+20) = 24
    
    P->>P: Visit node 6
    Note right of P: 7: min(24, 7+1) = 8
    
    P->>P: Visit node 7
    Note right of P: 8: min(∞, 8+3) = 11
    
    P->>E: Final shortest path
    Note over E: Distance[8] = 11
```



### Key Findings
Example 1 (0 to 5):
- Shortest path: 0 → 2 → 5
- Total distance: 11 units
- Key decision point: Choosing path through node 2 instead of direct path (14)

Example 2 (0 to 8):
- Shortest path: 0 → 6 → 7 → 8
- Total distance: 11 units
- Key decision point: Using path through nodes 6 and 7 instead of longer alternatives

### Implementation Considerations

#### Priority Queue Maintenance is Crucial for Efficiency

Efficient management of the priority queue is essential for the performance of Dijkstra's algorithm. The priority queue is used to select the next node with the smallest tentative distance:

- **Data Structure Choice**: Using a binary heap allows for O(log V) time complexity for insertion and extraction operations, making it suitable for most practical applications.
- **Update Operations**: When a node's distance is updated, the priority queue must reflect this change to maintain correct ordering.
- **Optimization**: Minimizing the number of decrease-key operations can improve efficiency, as they can be costly depending on the implementation.

Proper priority queue maintenance ensures the algorithm operates efficiently, especially on large graphs.

#### Path Reconstruction Requires Tracking Previous Nodes

To determine the actual shortest path, it's necessary to keep track of the previous node for each node visited:

- **Previous Node Map**: Maintain a mapping from each node to its predecessor on the shortest path.
- **Updating Predecessors**: Whenever a node's tentative distance is updated via edge relaxation, also update its predecessor.
- **Path Reconstruction**: After the algorithm completes, reconstruct the path by backtracking from the destination node using the predecessor map.

This tracking allows for the recovery of the shortest path, not just the shortest distance.

#### Edge Relaxation Determines Optimal Path Selection

Edge relaxation is the process of adjusting the tentative distances to neighboring nodes:

- **Relaxation Step**: For each adjacent node, calculate the new tentative distance through the current node.
- **Distance Comparison**: If the calculated distance is less than the existing tentative distance, update it.
- **Consistent Application**: Repeatedly applying edge relaxation ensures all nodes are evaluated for potential shorter paths.

Edge relaxation is fundamental to the algorithm's ability to find the shortest paths accurately.

#### Time Complexity: O((V + E) log V) with Binary Heap

The algorithm's time complexity depends on the data structures used:

- **Vertices (V)**: Each vertex may be inserted into or extracted from the priority queue, operations that take O(log V) time.
- **Edges (E)**: Each edge may cause a decrease-key operation in the priority queue during relaxation.
- **Overall Complexity**: Combining these, the total time complexity is O((V + E) log V) when a binary heap is used for the priority queue.

This complexity makes the algorithm efficient for graphs where E is not significantly larger than V.

#### Space Complexity: O(V + E) with Adjacency List

The algorithm's space requirements include:

- **Graph Storage**: Using an adjacency list requires O(V + E) space to store all vertices and edges.
- **Auxiliary Data Structures**: Additional space for the priority queue, distance map, and predecessor map, each requiring O(V) space.
- **Total Space Usage**: Combining these, the space complexity remains O(V + E), which is efficient for sparse graphs.

Efficient space utilization is crucial for handling large graphs without excessive memory consumption.

### Implementation Considerations
- Priority Queue maintenance is crucial for efficiency
- Path reconstruction requires tracking previous nodes
- Edge relaxation determines optimal path selection
- Time complexity: O((V + E) log V) with binary heap
- Space complexity: O(V + E) with adjacency list


## Comprehensive Guide to Dijkstra's Algorithm: Theory, Implementation, and Applications
The main idea behind Dijkstra's Algorithm is to efficiently find the shortest path between a starting node (source node) and every other node in a graph with non-negative edge weights.

It achieves this by using a greedy approach:

### Key Findings of Shortest Path in given Graph
Example 1 (0 to 5):
- Shortest path: 0 → 2 → 5
- Total distance: 11 units
- Key decision point: Choosing path through node 2 instead of direct path (14)


Example 2 (0 to 8):
- Shortest path: 0 → 6 → 7 → 8
- Total distance: 11 units
- Key decision point: Using path through nodes 6 and 7 instead of longer alternatives


# Comprehensive Guide to Dijkstra's Algorithm Implementation

## 1. Algorithm Overview
Dijkstra's algorithm finds the shortest paths between nodes in a weighted graph, which may represent networks, roads, or any weighted relationships.

### Core Components

```mermaid
graph TD
    subgraph Components
        PQ[Priority Queue] --> |Maintains Order| NP[Node Processing]
        NP --> |Updates| DT[Distance Tracking]
        DT --> |Records| PT[Path Tracking]
        PT --> |Enables| PR[Path Reconstruction]
    end
```

## 2. Implementation Structure

### Base Graph Structure
```java
public class WeightedGraph<T> {
    private Map<T, Map<T, Integer>> adjacencyList;
    
    public void addEdge(T source, T destination, int weight) {
        adjacencyList.computeIfAbsent(source, k -> new HashMap<>())
                    .put(destination, weight);
    }
}
```

### Priority Queue Management
```java
PriorityQueue<Node> pq = new PriorityQueue<>((a, b) -> 
    distances.get(a) - distances.get(b));
```

## 3. Algorithm Execution Process

```mermaid
sequenceDiagram
    participant Init as Initialization
    participant PQ as Priority Queue
    participant Process as Processing
    participant Path as Path Tracking
    
    Init->>PQ: Set source distance to 0
    Init->>PQ: Set all other distances to ∞
    
    loop While PQ not empty
        PQ->>Process: Get minimum distance node
        Process->>Process: Process all neighbors
        Process->>Path: Update distances and paths
        Process->>PQ: Add updated nodes
    end
    Path->>Path: Reconstruct shortest path
```

## 4. Example Walkthrough

### Sample Graph
```mermaid
graph TD
    0((0)) -->|7| 1((1))
    0 -->|9| 2((2))
    0 -->|14| 5((5))
    1 -->|10| 2((2))
    1 -->|15| 3((3))
    2 -->|11| 3((3))
    2 -->|2| 5((5))
    3 -->|6| 4((4))
    4 -->|9| 5((5))
    style 0 fill:#f9f,stroke:#333,stroke-width:4px
    style 1,2,3,4,5 fill:#bbf,stroke:#333,stroke-width:2px
```

### Processing Steps
Initialize distances:

Node 0: 0

All other nodes: ∞

Priority Queue States:

```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant D as Distances
    
    Note over PQ: Initial: [(0,0)]
    PQ->>D: Process 0
    Note over D: Update neighbors
    Note over PQ: Next: [(1,7),(2,9),(5,14)]
    PQ->>D: Process 1
    Note over PQ: Next: [(2,9),(5,11),(3,22)]
```

## 5. Path Reconstruction

```mermaid
graph TD
    subgraph Path Tracking Structure
        direction LR
        N0[Node 0] -->|"prev[1]=0"| N1[Node 1]
        N1 -->|"prev[2]=1"| N2[Node 2]
        N2 -->|"prev[5]=2"| N5[Node 5]
    end
```

### Implementation:
```java
private List<Node> reconstructPath(Node destination) {
    List<Node> path = new ArrayList<>();
    Node current = destination;
    
    while (current != null) {
        path.add(0, current);
        current = previousNodes.get(current);
    }
    return path;
}
```

## 6. Edge Relaxation Process

```mermaid
graph LR
    subgraph Edge Relaxation
        C((Current)) -->|"weight w"| N((Neighbor))
        style C fill:#f9f
        style N fill:#bbf
    end
    
    subgraph Decision
        direction TB
        D{{"If current.distance + w < neighbor.distance"}}
        U[Update neighbor distance]
        P[Update previous node]
    end
```

## 7. Performance Characteristics

### Time Complexity
With Binary Heap: O((V + E) log V)
With Fibonacci Heap: O(E + V log V)

### Space Complexity
Adjacency List: O(V + E)
Priority Queue: O(V)
Distance/Previous Arrays: O(V)

## 8. Optimization Techniques

### Priority Queue Optimization
```java
// Use offers instead of updates
if (newDistance < distances.get(neighbor)) {
    pq.offer(new Node(neighbor, newDistance));
    distances.put(neighbor, newDistance);
}
```

### Memory Management
```java
// Use primitive arrays for small graphs
int[] distances = new int[vertices];
int[] previous = new int[vertices];
```

### Early Termination
```java
if (current.equals(destination)) {
    break; // Found shortest path to destination
}
```

## 9. Best Practices

### Input Validation
```java
public void validateInput(Graph graph) {
    if (graph == null || graph.isEmpty()) {
        throw new IllegalArgumentException("Invalid graph");
    }
}
```

### Edge Case Handling
```java
if (source.equals(destination)) {
    return Collections.singletonList(source);
}
```

### Negative Weight Detection
```java
for (Edge edge : graph.getEdges()) {
    if (edge.weight < 0) {
        throw new IllegalArgumentException("Negative weights not supported");
    }
}
```

## 10. Common Applications
- Network Routing
- Social Networks
- Geographic Maps
- Game AI Pathfinding
- Resource Distribution

## 11. Testing Strategies
```java
@Test
public void testShortestPath() {
    WeightedGraph graph = new WeightedGraph();
    // Add test edges
    List<Node> path = dijkstra(graph, source, destination);
    assertNotNull(path);
    assertEquals(expectedDistance, getPathDistance(path));
}
```


## 1. Priority Queue Maintenance

```mermaid
sequenceDiagram
    participant PQ as Priority Queue
    participant VN as Visited Nodes
    participant DS as Distance State
    Note over PQ: Initial Queue State<br/>[(0,0)]
    
    PQ->>DS: Poll Node 0
    Note over DS: Process neighbors of 0<br/>Update distances
    DS->>PQ: Add neighbors with priorities:<br/>(1,7), (2,9), (5,14)
    
    Note over PQ: Queue State<br/>[(1,7), (2,9), (5,14)]
    
    PQ->>DS: Poll Node 1 (lowest distance)
    Note over DS: Process neighbors of 1<br/>Update distances
    DS->>PQ: Add/Update neighbors:<br/>(2,17), (3,22)
    
    Note over PQ: Queue State<br/>[(2,9), (5,14), (3,22)]
```

### Priority Queue Efficiency:
```java
// Binary Heap Implementation
PriorityQueue<Node> pq = new PriorityQueue<>((a, b) -> 
    distances.get(a) - distances.get(b));

// Key Operations:
// O(log n) insertion
pq.offer(new Node(vertex, distance));

// O(log n) removal of minimum element
Node current = pq.poll();

// O(1) peek at minimum element
Node next = pq.peek();
```

## 2. Path Reconstruction

```mermaid
graph TD
    subgraph Path Tracking
        0((0)) -->|"prev[1]=0"| 1((1))
        1 -->|"prev[2]=1"| 2((2))
        2 -->|"prev[5]=2"| 5((5))
        
        style 0 fill:#f9f,stroke:#333
        style 1,2,5 fill:#bbf,stroke:#333
    end

    subgraph Previous Nodes Map
        PM[Previous Nodes HashMap:<br/>5 → 2<br/>2 → 1<br/>1 → 0]
    end
```

### Path Tracking Implementation:
```java
private Map<Node, Node> previousNodes = new HashMap<>();

private void trackPath(Node current, Node neighbor, int newDistance) {
    if (newDistance < distances.get(neighbor)) {
        distances.put(neighbor, newDistance);
        previousNodes.put(neighbor, current);  // Track the path
    }
}

private List<Node> reconstructPath(Node destination) {
    List<Node> path = new ArrayList<>();
    Node current = destination;
    
    while (current != null) {
        path.add(0, current);  // Add to front of list
        current = previousNodes.get(current);  // Move to previous node
    }
    return path;
}
```

## 3. Edge Relaxation

```mermaid
sequenceDiagram
    participant C as Current Node
    participant N as Neighbor Node
    participant D as Distance Array
    
    Note over C,N: Edge Relaxation Process
    C->>N: Check edge weight (w)
    Note over C,D: Current distance = d
    Note over N,D: Neighbor distance = n
    
    alt d + w < n
        N->>D: Update distance
        Note over D: New distance = d + w
        N->>D: Update previous node
    else d + w >= n
        Note over D: Keep current distance
    end
```

### Edge Relaxation Implementation:
```java
private void relaxEdge(Node current, Node neighbor, int weight) {
    int currentDistance = distances.get(current);
    int neighborDistance = distances.get(neighbor);
    int newDistance = currentDistance + weight;
    
    if (newDistance < neighborDistance) {
        // Update distance
        distances.put(neighbor, newDistance);
        // Update previous node
        previousNodes.put(neighbor, current);
        // Add to priority queue with new priority
        pq.offer(neighbor);
    }
}
```

```mermaid
graph TD
    subgraph "Processing State"
        PQ["Priority Queue\nCurrent: (2,9)"]
        DM["Distance Map\n0:0, 1:7, 2:9, 3:∞, 4:∞, 5:14"]
        PM["Previous Map\n1←0, 2←0, 5←0"]
    end

    subgraph "Edge Relaxation"
        direction LR
        C((2)) -->|"weight=2"| N((5))
        Note["Old distance to 5: 14\nNew possible distance: 11\n11 < 14, so update"]
    end
    
    subgraph "Updated State"
        PQ2["Priority Queue\nNext: (5,11)"]
        DM2["Distance Map\n0:0, 1:7, 2:9, 3:∞, 4:∞, 5:11"]
        PM2["Previous Map\n1←0, 2←0, 5←2"]
    end
```

### Key Optimizations:
- Priority Queue Management
  - Maintain heap property after every update
  - Only add nodes when their distance improves
  - Use decrease-key operation when available

- Path Tracking
  - Only update previous node when finding shorter path
  - Store only essential path information
  - Reconstruct path only when needed

- Edge Relaxation
  - Process each edge at most once
  - Early termination when possible
  - Skip relaxation if current node's distance is infinity

## Comprehensive Guide to Dijkstra's Algorithm Implementation

### Overview
Dijkstra's Algorithm is a fundamental graph algorithm used to find the shortest paths between nodes in a weighted graph. This document provides a detailed exploration of its implementation, with a focus on a generic Java-based approach that supports multiple data types.

### Core Components

#### Graph Representation

1. Interface-based Design
The implementation uses a flexible interface-based approach:

```java
public interface Graph<T> {
    void addVertex(T vertex);
    void addEdge(T source, T destination, int weight);
    void addEdge(T source, T destination);
    Set<T> getNeighbors(T vertex);
    // ... other methods
}
```

2. Implementation Options

#### Adjacency List Implementation
```java
public class GraphList<T> implements Graph<T> {
    private final Map<T, Set<T>> adjacencyList;
    // Provides O(|V| + |E|) space complexity
    // Efficient for sparse graphs
}
```

#### Adjacency Matrix Implementation
```java
public class GraphMatrix<T extends Integer> implements Graph<T> {
    private final int[][] adjacencyMatrix;
    // Provides O(|V|²) space complexity
    // Efficient for dense graphs
}
```

#### Generic Node Structure
The implementation uses a generic node class to support various data types:

```java
public class GenericNode<T> {
    private T data;
    private List<GenericNode<T>> neighbors;
    private GenericNode<T> left;
    private GenericNode<T> right;
    
    // Constructor and methods
}
```

### Algorithm Implementation

#### Core Algorithm Structure
```java
public class DijkstraGeneric<T> {
    public Map<GenericNode<T>, Integer> findShortestPath(
            Map<GenericNode<T>, Map<GenericNode<T>, Integer>> graph,
            GenericNode<T> source,
            GenericNode<T> destination) {
        
        Map<GenericNode<T>, Integer> distances = new HashMap<>();
        Map<GenericNode<T>, GenericNode<T>> previousNodes = new HashMap<>();
        PriorityQueue<GenericNode<T>> queue = new PriorityQueue<>(
            (a, b) -> distances.getOrDefault(a, Integer.MAX_VALUE)
                    - distances.getOrDefault(b, Integer.MAX_VALUE)
        );
        
        // Initialize distances
        initializeDistances(graph, source, distances);
        
        // Process nodes
        processNodes(graph, distances, previousNodes, queue, destination);
        
        // Reconstruct path
        reconstructPath(previousNodes, destination);
        
        return distances;
    }
}
```

#### Key Components Breakdown

##### Initialization Phase
```java
private void initializeDistances(
        Map<GenericNode<T>, Map<GenericNode<T>, Integer>> graph,
        GenericNode<T> source,
        Map<GenericNode<T>, Integer> distances) {
    
    for (GenericNode<T> node : graph.keySet()) {
        distances.put(node, Integer.MAX_VALUE);
    }
    distances.put(source, 0);
}
```

##### Node Processing
```java
private void processNodes(
        Map<GenericNode<T>, Map<GenericNode<T>, Integer>> graph,
        Map<GenericNode<T>, Integer> distances,
        Map<GenericNode<T>, GenericNode<T>> previousNodes,
        PriorityQueue<GenericNode<T>> queue,
        GenericNode<T> destination) {
    
    while (!queue.isEmpty()) {
        GenericNode<T> current = queue.poll();
        if (current.equals(destination)) break;
        
        processNeighbors(current, graph, distances, previousNodes, queue);
    }
}
```

##### Path Reconstruction
```java
private void reconstructPath(
        Map<GenericNode<T>, GenericNode<T>> previousNodes,
        GenericNode<T> destination) {
    
    List<GenericNode<T>> path = new ArrayList<>();
    GenericNode<T> current = destination;
    
    while (current != null) {
        path.add(0, current);
        current = previousNodes.get(current);
    }
    
    printPath(path);
}
```

### Performance Characteristics

#### Time Complexity
With Binary Heap: O((|V| + |E|) log |V|)
With Fibonacci Heap: O(|E| + |V| log |V|)

#### Space Complexity
Adjacency List: O(|V| + |E|)
Adjacency Matrix: O(|V|²)

### Best Practices and Optimization Techniques

#### Priority Queue Optimization
- Use a binary heap for most cases
- Consider Fibonacci heap for very large graphs

#### Memory Management
- Use adjacency list for sparse graphs
- Use adjacency matrix for dense graphs

#### Edge Case Handling
- Validate input parameters
- Handle disconnected graphs
- Check for negative weights

### Common Pitfalls and Solutions

#### Negative Edge Weights
- Solution: Validate weights during edge addition
- Alternative: Use Bellman-Ford algorithm

#### Infinite Loops
- Solution: Track visited nodes
- Implement cycle detection

#### Memory Overflow
- Solution: Implement iterative deepening
- Use memory-efficient data structures

### Testing Strategies

#### Unit Tests
```java
@Test
public void testShortestPath() {
    GraphList<Integer> graph = new GraphList<>();
    // Add test edges
    assertEquals(expectedDistance, 
        algorithm.findShortestPath(graph, source, destination));
}
```

#### Edge Cases
- Empty graph
- Single node
- Disconnected components
- Maximum possible path length

### Conclusion
This implementation of Dijkstra's Algorithm provides a flexible, generic approach that can be adapted to various use cases while maintaining performance and reliability. The use of generics allows for versatile application across different data types, while the interface-based design ensures extensibility and maintainability.

