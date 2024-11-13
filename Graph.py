import numpy as np
from functools import cached_property, cache
from typing import List, Tuple, Dict
from typing_extensions import Self, Sequence
import networkx
from sympy.abc import x
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colormaps
from warnings import warn

Vertex = int
Edge = Tuple[Vertex, Vertex]
Color = int
Coloring = Dict[Vertex, Color]

class AdjacencyGraph:

    def __init__(
            self, 
            edges: List[Edge] =[], 
            coloring: Coloring = {}
        ) -> Self:
        
        self.vertices = []
        self.edges = []
        self.add_edges(edges)

        self.coloring = self.trivial_coloring if coloring == {} else coloring

    def __eq__(self, other) -> bool:

        return isinstance(other, self.__class__) and (self.coloring == other.coloring) and (self.vertices == other.vertices).all()

    def add_vertices(self, *vertices: Sequence[Vertex]):
        """
        Adds Vertices to the graph. Does nothing if the vertex is already present
        """

        self.vertices = np.unique(
            []+list(vertices)
        )


    def add_edges(self, *edges: Sequence[Edge]):
        """
        Adds Edges to the graph. Implicitly adds new vertices defined in the edge
        """

        [self.add_vertices(e) for e in edges]

        self.edges = np.unique(
            self.edges + list(edges),
            axis=1
        )[0]

    def coloring_cost(
        self,
        cost_map: Dict[Color, int | float]
    ) -> float:
  
        colors = list(self.coloring.values())
    
        return sum(
            map(
                lambda c: cost_map[c],
                colors
            )
        )
    
    def apply_coloring(self, c: Coloring) -> Self:

        self.coloring = c
        return self


    def summary(
        self, 
        cost_map: Dict[Color, int | float] = {}
    ) -> str:
        
        res = f'{self.current_coloring_number}-colored graph;' 
        
        coloring_violations = self.coloring_violations

        if coloring_violations > 0:
            res += '\033[91m' + f'IMProper coloring ({coloring_violations} violations)' + '\033[0m'
        else:
            res += '\033[92mProper coloring (No violations)\033[0m'


        if cost_map != {}:
            res += f'; With cost {self.coloring_cost(cost_map):.3f}'

        return res

    @property
    def is_proper_coloring(self) -> bool:
        # lazier than coloring_violations == 0
        
        for (v, u) in self.edges:

            if self.coloring[v] == self.coloring[u]: return False

        return True

    @property
    def coloring_violators(self) -> List[Vertex]:

        violators = []
        for (v, u) in self.edges:

            if self.coloring[v] == self.coloring[u]: violators.extend([u, v])

        
        return list(set(violators))

    @property
    def coloring_violations(self) -> int:

        return len(self.coloring_violators)

    @property
    def trivial_coloring(self) -> Coloring:

        return { v : v for v in self.vertices}

    @property    
    def is_trivial_coloring(self) -> bool:

        return self.coloring == self.trivial_coloring
    
    @property
    def current_coloring_number(self) -> int:

        return np.unique(list(self.coloring.values())).shape[0]
    
    @cached_property
    def chromatic_polynomial(self) -> int:
        # trampa para conseguir la estimacion del optimo
        nxg = networkx.Graph([ tuple(e) for e in self.edges])

        return networkx.chromatic_polynomial(nxg)
    
    @cached_property
    def minimum_coloring(self) -> int:
    
        maximum_coloring = len(self.vertices)
        for n in range(1, maximum_coloring):

            Xi_n = self.chromatic_polynomial.subs({x: n})
            if Xi_n > 0: return n

        return n
    
    def plot(
        self,
        cost_map: Dict[Color, float] ={}, 
        show: bool = False
    ) -> Figure:

        nxg = networkx.Graph([tuple(e) for e in self.edges])

        fig, ax = plt.subplots()
        networkx.draw_networkx(
            nxg, 
            ax=ax,
            #pos=networkx.layout.circular_layout(nxg),
            cmap=colormaps['gist_rainbow'],
            node_color=[ self.coloring[n] for n in nxg.nodes ] # respect order
        )

        title = f'{self.current_coloring_number}-coloring'
        if cost_map != {}:
            title += f'; Coloring cost: {self.coloring_cost(cost_map)}'

        plt.title(title)

        if show:
            warn('Running show=True ma interrupt execution to display graphics; close the window to continue')
            plt.show()

        return fig