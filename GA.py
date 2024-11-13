from Graph import AdjacencyGraph
import GeneticAlgo
from sympy.abc import x
from matplotlib import pyplot as plt
import networkx
from datetime import timedelta

#e = [(0, 5), (0, 4), (0, 1), (1, 6), (1, 2), (2, 7), (2, 3), (3, 8), (3, 4), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9)]
#e = networkx.petersen_graph().edges
#e = networkx.dorogovtsev_goltsev_mendes_graph(3).edges 
e = networkx.desargues_graph().edges

cost_map = {
    0: 11.7,
    1: 12.2,
    2: 14,
    3: 14.1,
    4: 14.7,
    5: 15.0,
    6: 21.1,
    7: 23.1,
    8: 43.1,
    9: 43.2,
} | {
    c: 100 for c in range(10, 21)
}

print('Loading Graph...')
g = AdjacencyGraph(edges=e)
#print('Calculating chromatic polynomial...')
#print(f'Chromatic Polynomial k={g.minimum_coloring}; ꭓg({g.minimum_coloring})={g.chromatic_polynomial.subs({x: g.minimum_coloring})};')
#estimated_optimum = GeneticAlgo.estimate_optimum(g, cost_map)
#print(f'Optimal solution estimated at ~{estimated_optimum};')
print()

best_coloring, history = GeneticAlgo.GA(
    g,
    cost_map,
    100_000_000,
    initial_population_size=2_000,
    optimal_cost_estimation=239,
    optimal_cost_tol=0,
    optimal_k_estimation=False,
    symetrical_crossing_offspring=10,
    graph_mutate_chance=0.9,
    introduce_n_random_individuals=2_000,
    least_fitting_prune_enabled=False,
    least_fitting_prune_ratio=0.2,
    randomness_survival_pruning_enabled=True,
    randomness_survival_chance=0.2,
    scheduled_pruning=True,
    patience=None,
    deadline=timedelta(hours=5),
)
#print(best.edges)

best = g.apply_coloring(best_coloring)
best.plot(cost_map=cost_map, show=True)

plt.title('K progresion')
plt.xlabel('Iterations')
plt.ylabel('K')
#plt.axhline(y=g.minimum_coloring, color='r', label='optimal solution')
plt.plot(range(len(history['best_k'])), history['best_k'], label='K')
plt.legend()
plt.show()

plt.title('Cost progression')
plt.xlabel('Iterations')
plt.ylabel('Coloring cost')

#plt.axhline(y=estimated_optimum, label='estimated optimum', color='k')
plt.plot(range(len(history['best_cost'])), history['best_cost'], label='MH cost')
plt.legend()
plt.show()

