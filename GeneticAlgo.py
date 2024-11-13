from Graph import AdjacencyGraph, Color, Vertex, Coloring, Edge
from typing import Callable, List, Dict, Tuple
import numpy as np
from itertools import pairwise
from random import shuffle
from copy import deepcopy
from time import time
from sys import getsizeof
from datetime import timedelta

CLI_BLUE_HIGHLIGHT = '\033[94m'
CLI_GREEN_HIGHLIGHT = '\033[92m'
CLI_RED_HIGHLIGHT = '\033[91m'

def random_coloring(
    vertices: List[Vertex],
    k: int | None = None
) -> Coloring:
    n_vertices = len(vertices)
    k = k if k is not None else n_vertices
    random_colors = np.random.randint(1, k+1, size=n_vertices)

    return dict(zip(vertices, random_colors))


def mutate_coloring(
    cl: List[Coloring], 
    graph_mutate_chance: float,
) -> List[Coloring]:

    graph_mutates = np.random.rand(len(cl)) > (1 - graph_mutate_chance)

    if not any(graph_mutates):
        return []

    res = []
    
    for mutates, k, c in zip(graph_mutates, [ np.unique(list(c.values())).shape[0] for c in cl], cl):

        if mutates:

            origin_color, destination_color = np.random.randint(0, k, 2)

            vertices_with_origin_color = list(filter(
                lambda v: c[v]==origin_color,
                np.unique(list(c.keys()))
            ))

            vertices_with_destination_color = list(filter(
                lambda v: c[v]==destination_color,
                np.unique(list(c.keys()))
            ))

            c.update(
                { v : destination_color for v in vertices_with_origin_color }
                | { v : origin_color for v in vertices_with_destination_color }
            )

        res.append(c)


    return res

def estimate_optimum(g: AdjacencyGraph, cost_map: Dict[Color, float]) -> float:

    k = g.minimum_coloring

    average_optimum_solution_color_cost = (sum(sorted(cost_map.values())[:k])/k)
    return average_optimum_solution_color_cost * len(g.vertices)


def next_generation(
    g: AdjacencyGraph,
    cl: List[Coloring],
    symetrical_crossing_offspring: int,
    least_fitting_prune_enabled: bool,
    least_fitting_prune_ratio: float,
    randomness_survival_pruning_enabled: bool,
    randomness_survival_chance: float,
    graph_mutate_chance: float,
    introduce_n_random_individuals: int,
    random_individuals_with_k: int,
    scheduled_pruning: bool = False,
    pruning_schedule: float = 0,
) -> List[Coloring]:
    
    assert least_fitting_prune_enabled or randomness_survival_pruning_enabled and (least_fitting_prune_enabled != randomness_survival_pruning_enabled), 'At least a pruning method has to be enabled'

    if scheduled_pruning:
        least_fitting_prune_ratio =  min(least_fitting_prune_ratio * pruning_schedule, 1)
        randomness_survival_chance = min(randomness_survival_chance * pruning_schedule, 1)
        graph_mutate_chance = min(pruning_schedule*graph_mutate_chance, 1)

    # 1. Prune the parents
    if randomness_survival_pruning_enabled:
        cl = prune_by_randomness(
            cl,
            survival_chance=randomness_survival_chance
        )

    if least_fitting_prune_enabled:
        cl = prune_by_least_fitting(
                cl, 
                least_fitting_prune_ratio,
                lambda c: g.apply_coloring(c).current_coloring_number
        )

    if cl == []:
        return cl

    # 2. Introduce randomness
    cl += [
        random_coloring(g.vertices, k=random_individuals_with_k)
        for _ in range(introduce_n_random_individuals)
        ]
    

    # 3. Breed the pairs
    offspring = create_offspring(
        select_crossings(
            prune_duplicates(cl)
        ),
        symetrical_crossing_offspring
    )

    # 4. Mutate the offspring
    offspring = mutate_coloring(
        offspring,
        graph_mutate_chance
    )

    next_gen = prune_duplicates(offspring + cl)

    return next_gen


def select_crossings(
    cl: List[Coloring]
) -> List[Tuple[Coloring]]:
    # un poco random yo que se

    return zip( cl, reversed(cl) )


def create_offspring(
    clp: List[Tuple[Coloring]],
    c_max: int
) -> List[Coloring]:
    
    offspring = []

    n_of_vertices = np.unique(list(next(clp)[0].keys())).shape[0]
    c_max = c_max if c_max < n_of_vertices / 2 else n_of_vertices / 2

    for (coloring1, coloring2) in clp:

        if coloring1 == coloring2: continue


        c = np.random.randint(1, c_max+1 )
        # shuffling works with dicts
        shuffle(coloring1)
        shuffle(coloring2) 

        coloring1 = list(coloring1.items())
        coloring2 = list(coloring2.items())

        offsp1 = dict(coloring1[:c] + coloring2[c:-c] + coloring1[-c:])
        offsp2 = dict(coloring2[:c] + coloring1[c:-c] + coloring2[-c:])

        offspring += [offsp1, offsp2]

    return offspring


def prune_by_least_fitting(
    cl: List[Coloring],
    prune_pct: float,
    fitness_fn: Callable[[Coloring], float]
) -> List[Coloring]:
    
    n = max(int((1 - prune_pct) * len(cl)), 0)

    cl_sorted_by_fitness = sort_by_fitness(cl, fitness_fn)

    is_survivor = ([True]*n)+([False]*(len(cl_sorted_by_fitness)-n))
    
    return prune(cl_sorted_by_fitness, is_survivor)


def prune_violators(
    g: AdjacencyGraph,
    cl: List[Coloring]
) -> List[Coloring]:

    return prune(cl, [g.apply_coloring(c).is_proper_coloring for c in cl])


def prune_colorings_over_k(    
    cl: List[Coloring],
    k: int
) -> List[Coloring]:
    
    return prune(cl, [ np.unique( c.values() ).shape[0] == k for c in cl ])

def sort_by_fitness(
    cl: List[Coloring], 
    fitness_fn: Callable[[Coloring], float],
) -> List[Coloring]:
    
    return list(map(
       lambda c_g: c_g[1],
        sorted(
            zip(
                map(
                    fitness_fn,
                    cl
                ),
                cl
            ),
            key= lambda c_g: c_g[0]
        )
   ))


def prune_by_randomness(
    cl: List[Coloring],
    survival_chance: float
) -> List[Coloring]:
    
    is_survivor = np.random.uniform(0, 1, (len(cl))) > (1 - survival_chance)
    return prune(cl, is_survivor)


def prune_duplicates(
        cl: List[Coloring]
    ) -> List[Coloring]:
    unique_dicts = []
    seen = set()

    for d in cl:
        # Convert the dictionary to a tuple of sorted items
        dict_tuple = tuple(sorted(d.items()))
        
        # If we haven't seen this dictionary before, add it to the result
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_dicts.append(d)

    return unique_dicts


def prune(
    cl: List[Coloring],
    is_survivor: List[bool]
) -> List[Coloring]:
    
    return list(
        map(
            lambda g_s: g_s[0],
            filter(
                lambda g_s: g_s[1],
                zip(cl, is_survivor)
            )
        )
    )

def GA(
    g: AdjacencyGraph,
    cost_map: Dict[Color, int | float],
    iter_limit: int,
    initial_population_size: int = 50,
    optimal_cost_estimation: float = False,
    optimal_cost_tol: float | None = 0,
    optimal_k_estimation: int | bool = False,
    symetrical_crossing_offspring: int = 3,
    least_fitting_prune_enabled: bool = True,
    least_fitting_prune_ratio: float = 0.2,
    randomness_survival_pruning_enabled: bool = True,
    randomness_survival_chance: float = 0.9,
    graph_mutate_chance: float = 0.3,
    introduce_n_random_individuals: float = 25,
    scheduled_pruning: bool = False,
    print_enabled: bool = True,
    patience: int | None = None,
    deadline: timedelta | None = None,
) -> Tuple[Coloring, List[float], List[int]]:
    
    start_of_ga_t = time()
    best_cost = float('inf')
    patience = patience if patience is not None else iter_limit
    current_patience = patience
    
    has_extinguished = False
    within_tolerance_optimum = False

    if isinstance(optimal_k_estimation, bool) and not optimal_k_estimation:
        # if false, do not use to guide
        optimal_k_estimation = len(g.vertices)

    cl = [ random_coloring(g.vertices, k=optimal_k_estimation) for _ in range(initial_population_size) ]
    best_coloring = cl[0]
    best_coloring_of_current_pop = best_coloring
    best_k_of_current_pop = g.apply_coloring(best_coloring).current_coloring_number


    history = {}
    history['best_cost'] = []
    history['best_k'] = []
    history['population_size'] = []
    history['population_size_mb'] = []

    try:
        for iterno in range(iter_limit):

            print_str = ''

            start_t = time()
            prune_eta = 1+(iterno/iter_limit)

            # 1. Get the current best
            ranked_proper_colorings_by_cost = sort_by_fitness(
                prune_violators(g, cl),
                lambda cl: g.apply_coloring(cl).coloring_cost(cost_map)
            )



            if len(ranked_proper_colorings_by_cost) > 0:
                best_coloring_of_current_pop = ranked_proper_colorings_by_cost[0]
                best_k_of_current_pop = ( [np.unique(list(c.values())).shape[0] for c in ranked_proper_colorings_by_cost] )[0]

            best_cost_of_current_pop = g.apply_coloring(best_coloring_of_current_pop).coloring_cost(cost_map)

            is_better_k = g.apply_coloring(best_coloring_of_current_pop).current_coloring_number < g.apply_coloring(best_coloring).current_coloring_number
            is_same_k_but_better_cost = (g.apply_coloring(best_coloring_of_current_pop).current_coloring_number == g.apply_coloring(best_coloring).current_coloring_number  and best_cost_of_current_pop < best_cost)
            

            # CLI
            best_cost_highlight = CLI_BLUE_HIGHLIGHT
            best_cost_of_pop_highlight = best_cost_highlight
            if best_cost < best_cost_of_current_pop:
                best_cost_highlight        = CLI_GREEN_HIGHLIGHT
                best_cost_of_pop_highlight = CLI_RED_HIGHLIGHT
            elif best_cost > best_cost_of_current_pop:
                best_cost_highlight        = CLI_RED_HIGHLIGHT
                best_cost_of_pop_highlight = CLI_GREEN_HIGHLIGHT


            best_k_highlight = CLI_BLUE_HIGHLIGHT
            best_k_of_pop_highlight = best_k_highlight
            if g.apply_coloring(best_coloring_of_current_pop).current_coloring_number > g.apply_coloring(best_coloring).current_coloring_number:
                best_k_highlight = CLI_GREEN_HIGHLIGHT
                best_k_of_pop_highlight = CLI_RED_HIGHLIGHT
            elif g.apply_coloring(best_coloring_of_current_pop).current_coloring_number < g.apply_coloring(best_coloring).current_coloring_number:
                best_k_highlight = CLI_RED_HIGHLIGHT
                best_k_of_pop_highlight = CLI_GREEN_HIGHLIGHT

            print_str += f'It{iterno}:\tCurrent Best={best_cost_highlight}{best_cost:.3f}\033[0m; Population best cost={best_cost_of_pop_highlight}{best_cost_of_current_pop:.3f}\033[0m\n'
            print_str += f'\t           K={best_k_highlight}{g.apply_coloring(best_coloring).current_coloring_number}\033[0m      ;            best    K={best_k_of_pop_highlight}{best_k_of_current_pop}\033[0m\n'
            print_str += '-'*130+'\n'
            print_str += '\t\tSolution summary:\n'
            print_str += f'\t\t\tPopulation best: {g.apply_coloring(best_coloring_of_current_pop).summary(cost_map=cost_map)}\n'
            print_str += f'\t\t\tCurrent best   : {g.apply_coloring(best_coloring).summary(cost_map=cost_map)}\n'

            if (is_better_k or is_same_k_but_better_cost) and (best_coloring_of_current_pop != best_coloring) and (g.apply_coloring(best_coloring_of_current_pop).is_proper_coloring):

                reason = ''
                if is_better_k:
                    reason += 'Lower K '
                if is_same_k_but_better_cost:
                    reason += 'Lower cost '

                print_str += '\033[92m' + f'\t\tImprovement detected: Reason: {reason}' + '\033[0m\n'
                # NOTE: To avoid the graph changing into an IMproper coloring during mutation
                #        since copy is by reference, recreate it
                best_coloring = deepcopy(best_coloring_of_current_pop)
                best_cost = best_cost_of_current_pop
                current_patience = patience
            else:
                print_str += '\033[91m'+f'\t\tNo improvement' + '\033[0m\n'
                current_patience -= 1

            print_str += '-'*130+'\n'
            print_str += '\t\tProgress summary:\n'
            if optimal_cost_estimation != float('inf'):
                distance_to_opt_estimation = best_cost - optimal_cost_estimation
                within_tolerance_optimum = distance_to_opt_estimation <= optimal_cost_estimation * optimal_cost_tol
            else:
                distance_to_opt_estimation = float('inf')
            
            print_str += f'\t\t\tDeadline  ={(deadline - timedelta(seconds=time()-start_of_ga_t)) if deadline is not None else "inf"} s\n'
            print_str += f'\t\t\tPatience  ={current_patience} iterations\n'
            print_str += f'\t\t\tIterations={iterno} / {iter_limit} ({(iterno / iter_limit)*100:.3f}%)\n'
            print_str += f'\t\t\tOptimum estimation: ~ {optimal_cost_estimation:.3f}; distance to estimation: {distance_to_opt_estimation}\n'

            
            current_gen_size = len(cl)
            print_str += '-'*130+'\n'
            print_str += '\t\tPopulation summary:\n'
            print_str += f'\t\t\tCurrent Population size: {len(cl)}; ({getsizeof(cl)/2048:.1f} MB)\n'

            history['population_size_mb'].append(getsizeof(cl)/2048)
            history['population_size'].append(len(cl))
            history['best_cost'].append(best_cost)
            history['best_k'].append(g.apply_coloring(best_coloring).current_coloring_number)

            if within_tolerance_optimum:
                break
            if current_patience < 1:
                break

            cl = next_generation(
                g,
                cl,
                symetrical_crossing_offspring,
                least_fitting_prune_enabled,
                least_fitting_prune_ratio,
                randomness_survival_pruning_enabled,
                randomness_survival_chance,
                graph_mutate_chance,
                introduce_n_random_individuals,
                min(g.apply_coloring(best_coloring).current_coloring_number, optimal_k_estimation),
                scheduled_pruning=scheduled_pruning, 
                pruning_schedule=prune_eta
            ) + [deepcopy(best_coloring)]


            t = time() - start_t
            print_str += f'\t\t\tNew Population size    : {len(cl)}; ({getsizeof(cl)/2048:.1f} MB)\n'
            print_str += f'\t\t\t\tPopulation Increase ratio: {len(cl)/current_gen_size:.3f}\n'
            print_str += '\n'
            print_str += f'Iteration took {timedelta(seconds=t)} seconds; Total time elapsed: {timedelta(seconds=(time()-start_of_ga_t))}\n'
            print_str += '='*130+'\n'

            if print_enabled:
                print(print_str)

            has_extinguished = cl == []
            if has_extinguished:
                break
            if deadline is not None and deadline < timedelta(seconds=time()-start_of_ga_t):
                break

        stop_reason = "Iter limit"
        if current_patience == 0:
            stop_reason = f'Ran out of patience (No improvement in {patience} rounds)'
        elif deadline is not None and deadline < timedelta(seconds=time()-start_of_ga_t):
            stop_reason = f'Hit time deadline'
        elif optimal_cost_estimation is not None and within_tolerance_optimum:
            stop_reason = f"Solution within {optimal_cost_tol} of estimated optimum"
        elif has_extinguished:
            stop_reason = "\033[91mExtinguished population\033[0m"

    except KeyboardInterrupt:
        stop_reason = "User interrupt"

    # Cover if there was maybe an early exit and we haven't added the stats yet
    history['time'] = timedelta(seconds=time()-start_of_ga_t)
    history['iterations'] = iterno
    history['best_cost'].append(best_cost)
    history['best_k'].append(g.apply_coloring(best_coloring).current_coloring_number)
    history['stop_reason'] = stop_reason
    
    if print_enabled:
        print('\n')
        print('*'*130)
        print(f'Stopped: Reason: {stop_reason}')
        print(f'Best found solution: {best_cost}')
        print(f'Best: {g.apply_coloring(best_coloring).summary(cost_map=cost_map)}')
        print(f'Best coloring: {best_coloring}')
        print(f'Took {history["time"]} s')
        print('*'*130)
        print()


    return best_coloring, history