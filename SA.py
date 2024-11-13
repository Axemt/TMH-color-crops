from Graph import AdjacencyGraph
import SimulatedAnnealing
from sympy.abc import x
from matplotlib import pyplot as plt
import networkx
from datetime import timedelta

#e = [(0, 5), (0, 4), (0, 1), (1, 6), (1, 2), (2, 7), (2, 3), (3, 8), (3, 4), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9)]
#e = networkx.petersen_graph().edges

# 3-colorable, best=530.6
e = networkx.dorogovtsev_goltsev_mendes_graph(4).edges
#e = networkx.desargues_graph().edges

print('Loading Graph...')
g = AdjacencyGraph(edges=e)

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
    c: 100 for c in range(10, len(g.vertices)+1)
}


best_coloring = g.trivial_coloring
g.apply_coloring(best_coloring)


#print('Calculating chromatic polynomial...')
#print(f'Chromatic Polynomial k={g.minimum_coloring}; ꭓg({g.minimum_coloring})={g.chromatic_polynomial.subs({x: g.minimum_coloring})};')
#estimated_optimum = GeneticAlgo.estimate_optimum(g, cost_map)
#print(f'Optimal solution estimated at ~{estimated_optimum};')
print()

best_coloring, history = SimulatedAnnealing.SA(
    g,
    cost_map,
    None,
    1,
    temperature_k=0.99,
    temperature_reduction_method='decay',
    print_enabled=True,
    optimal_cost_estimation=530.6,
    deadline=timedelta(hours=5),
)
#print(best.edges)

#print(history)
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