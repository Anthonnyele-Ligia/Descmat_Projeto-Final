import numpy as np
import matplotlib.pyplot as plt
import heapq
from sympy import Matrix, symbols, Eq, solve

# --- 1. Gerar Matriz de Adjacência ---
def generate_adjacency_matrix():
    """Cria uma matriz de adjacência para representar uma rede."""
    matrix = np.array([
        [0, 10, 0, 0, 30],
        [10, 0, 50, 0, 0],
        [0, 50, 0, 20, 10],
        [0, 0, 20, 0, 60],
        [30, 0, 10, 60, 0]
    ])
    return matrix

# --- 2. Implementar Algoritmo de Dijkstra ---
def dijkstra(matrix, start_node):
    """Encontra o caminho mais curto a partir do nó inicial usando Dijkstra."""
    num_nodes = len(matrix)
    distances = [float('inf')] * num_nodes
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    predecessors = {i: None for i in range(num_nodes)}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        for neighbor, weight in enumerate(matrix[current_node]):
            if weight > 0:  # Existe conexão
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors

# --- 3. Visualizar Rede ---
def plot_network(matrix, shortest_path=None, distances=None):
    """Gera o gráfico da rede com a rota destacada (se fornecida) no estilo Excel."""
    num_nodes = len(matrix)
    positions = {i: (np.cos(2 * np.pi * i / num_nodes), np.sin(2 * np.pi * i / num_nodes)) for i in range(num_nodes)}

    plt.figure(figsize=(12, 12))
    import seaborn as sns
    sns.set_style("darkgrid")

    # Desenhar todas as conexões
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix[i, j] > 0:
                x_values = [positions[i][0], positions[j][0]]
                y_values = [positions[i][1], positions[j][1]]
                plt.plot(x_values, y_values, color='#A6ACAF', linestyle='--', linewidth=1.5, alpha=0.7)
                mid_x = (positions[i][0] + positions[j][0]) / 2
                mid_y = (positions[i][1] + positions[j][1]) / 2
                plt.text(mid_x, mid_y, f'{matrix[i, j]}', fontsize=10, color='black', fontweight='bold', zorder=6)

    # Destacar o caminho mais curto
    if shortest_path:
        for i in range(len(shortest_path) - 1):
            x_values = [positions[shortest_path[i]][0], positions[shortest_path[i + 1]][0]]
            y_values = [positions[shortest_path[i]][1], positions[shortest_path[i + 1]][1]]
            plt.plot(x_values, y_values, color='#1F618D', linewidth=3.5, label='Caminho Mais Curto' if i == 0 else "")

    # Desenhar nós
    for node, (x, y) in positions.items():
        plt.scatter(x, y, color='#2E86C1', s=200, edgecolor='black', zorder=5)
        plt.text(x, y, f'  {node}', fontsize=12, fontweight='bold', color='black', zorder=6)

    # Adicionar distâncias no gráfico
    if distances:
        for node, distance in enumerate(distances):
            plt.text(positions[node][0], positions[node][1] - 0.1, f'Dist: {distance}', fontsize=10, color='green', fontweight='bold', zorder=6)

    # Configurar eixos
    plt.xticks(fontsize=10, color='black', fontweight='bold')
    plt.yticks(fontsize=10, color='black', fontweight='bold')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

    plt.title("Gráfico da Rede de Transporte", fontsize=18, fontweight='bold', color='#34495E')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- 4. Reconstruir o Caminho Mais Curto ---
def reconstruct_path(predecessors, start_node, end_node):
    """Reconstrói o caminho mais curto a partir dos predecessores."""
    path = []
    current = end_node
    while current is not None:
        path.insert(0, current)
        current = predecessors[current]
    return path if path[0] == start_node else []

# --- Execução ---
if __name__ == "__main__":
    # Gerar matriz de adjacência
    adj_matrix = generate_adjacency_matrix()

    # Definir nós de origem e destino
    start_node = 0
    end_node = 3

    # Calcular caminhos mais curtos
    distances, predecessors = dijkstra(adj_matrix, start_node)
    shortest_path = reconstruct_path(predecessors, start_node, end_node)

    # Resultados
    print(f"Distâncias a partir do nó {start_node}: {distances}")
    print(f"Caminho mais curto de {start_node} para {end_node}: {shortest_path}")

    # Visualizar rede
    plot_network(adj_matrix, shortest_path=shortest_path, distances=distances)

    # Exemplo de uso do SymPy: Resolver sistema linear
    x, y, z = symbols('x y z')
    equations = [
        Eq(2 * x + y - z, 1),
        Eq(x + 3 * y + z, 3),
        Eq(x - y + 2 * z, 2)
    ]
    solution = solve(equations, (x, y, z))
    print("Solução do sistema linear:", solution)
