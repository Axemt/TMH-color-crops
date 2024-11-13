from Graph import AdjacencyGraph, Color, Coloring, Vertex, Edge
from typing import List, Dict, Literal
import numpy as np
from datetime import timedelta
from copy import deepcopy
from time import time


CLI_BLUE_HIGHLIGHT = '\033[94m'
CLI_GREEN_HIGHLIGHT = '\033[92m'
CLI_RED_HIGHLIGHT = '\033[91m'


def fix_violation_or_decay_change(g: AdjacencyGraph) -> AdjacencyGraph:

    # Random choice of vertex decays to a lower coloring

    color_to_decay_to = np.random.randint(0, g.current_coloring_number)
    if g.coloring_violations > 0:
        vertex_to_decay = np.random.choice(g.coloring_violators)
    else:
        vertex_to_decay = np.random.choice(g.vertices)

    new_coloring = g.coloring
    new_coloring[vertex_to_decay] = color_to_decay_to

    return g.apply_coloring(new_coloring)

def whole_color_permute(g: AdjacencyGraph) -> AdjacencyGraph:

    origin_color = np.random.randint(0, g.current_coloring_number)
    destination_color = np.random.randint(0, g.current_coloring_number)

    c = g.coloring

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

    return g.apply_coloring(c)

def whole_color_change(g: AdjacencyGraph) -> AdjacencyGraph:

    color_that_decays = np.random.randint(0, g.current_coloring_number)
    color_to_decay_to = np.random.randint(0, g.current_coloring_number)

    new_coloring = { v:(c if c != color_that_decays else color_to_decay_to) for  v, c in g.coloring.items()}

    return g.apply_coloring(new_coloring)


def SA(
    g: AdjacencyGraph,
    cost_map: Dict[Color, float],
    iter_limit: int,
    temperature: float,
    temperature_reduction_method: Literal['linear', 'factor', 'decay'] = 'linear',
    temperature_k: float = 0.8,
    optimal_cost_estimation: float | None = None,
    optimal_cost_tol: float = 0,
    print_enabled: bool = True,
    deadline: timedelta | None = None        
) -> Coloring:

    assert temperature_reduction_method in ['linear', 'factor', 'decay'], 'Given temperature decay method is not valid'

    best_g = deepcopy(g)
    history = {
        'best_k': [],
        'best_cost': [],
    }
    print_str = ''


    if iter_limit is None:
        iter_limit = float('inf')

    if optimal_cost_estimation is None:
        optimal_cost_estimation = float('inf')

    start_of_sa_t = time()
    # 1.Siempre existen sucesores del estado actual de g (en este problema)
    try:
        iterno = 0
        while iterno < iter_limit:
            print_str = '='*100+'\n'
            predecessor_cost = g.coloring_cost(cost_map)
    
            
            neighbour_fn = np.random.choice(
                [
                    fix_violation_or_decay_change,
                    whole_color_change,
                    whole_color_permute
                ], 
                p=(
                    [1, 0, 0] if g.coloring_violations > 0 else None
                )
            )
            g_successor = neighbour_fn(deepcopy(g))
            successor_cost = g.coloring_cost(cost_map)
    
            print_str += f'It{iterno}: Current best={best_g.coloring_cost(cost_map):.3f}; Predecessor cost={predecessor_cost:.3f}; Successor cost={successor_cost:.3f}\n'
    
    
            cost_delta = successor_cost - predecessor_cost
    
    
            improves_current = cost_delta < 0
            improves_best_cost = successor_cost < best_g.coloring_cost(cost_map)
            improves_best_k = g_successor.current_coloring_number < best_g.current_coloring_number
    
            print_str += '-'*100+'\n'
            print_str += '\tSolution summary:\n'
            print_str += f'\t\tBest       : {best_g.summary(cost_map=cost_map)}\n'
            print_str += f'\t\tG          : {g.summary(cost_map=cost_map)}\n'
            print_str += f'\t\tG successor: {g_successor.summary(cost_map=cost_map)}\n'
            print_str += f'\n\tCost delta = {cost_delta};\n'
            print_str += '-'*100+'\n'
            print_str += '\tUpdate summary:\n\t\t'
            
            if (improves_best_cost and improves_best_k) and g_successor.is_proper_coloring:
            
                print_str += 'Improvement over best'
                
                best_g = deepcopy(g_successor)
                g = g_successor
            
            elif improves_current:
                g = deepcopy(g)
    
                print_str += CLI_BLUE_HIGHLIGHT+'Improvement over current\033[0m;'
            else:
                 
                # With chance = e^(Ac/T) and Ac < 0, chance in [0, 1]
                bern_chance = np.exp( -(cost_delta + np.finfo(float).eps)/(temperature+np.finfo(float).eps) )
                is_successor_accept = np.random.binomial(1, np.min([bern_chance, 1]))
                
                print_str += f'{CLI_RED_HIGHLIGHT}No improvement over current\033[0m; Cost delta = {cost_delta}; Accept chance = {bern_chance}; ' +  f'{CLI_GREEN_HIGHLIGHT if is_successor_accept else CLI_RED_HIGHLIGHT}{"Accepted" if is_successor_accept else "Rejected"}\033[0m;'
                
                if is_successor_accept:
                    g = g_successor
    
    
            print_str += '\n'
    
            if temperature_reduction_method == 'linear':
                temperature = np.max([temperature - temperature_k, 0])
            elif temperature_reduction_method == 'factor':
                assert temperature_k < 1 and temperature_k > 0, f'Temperature k {temperature_k} is not in the proper range for reduction method "Factor"'
                temperature *= temperature_k
            elif temperature_reduction_method == 'decay':
                temperature = temperature / (1+temperature_k*temperature)
            else:
                pass
            
            temperature = np.max([temperature, 0])
    
            print_str += f'\t\tCurrent temperature: {temperature}; Reduction method: {temperature_reduction_method};\n'
            print_str += '-'*100+'\n'
            print_str += '\tProgress summary:\n'
            print_str += f'\t\t\tDeadline  ={(deadline - timedelta(seconds=time()-start_of_sa_t)) if deadline is not None else "inf"} s\n'
            print_str += f'\t\t\tIterations={iterno} / {iter_limit} ({(iterno / iter_limit)*100:.3f}%)\n'

            distance_to_opt_estimation = float('inf')
            within_tolerance_optimum = False
            if optimal_cost_estimation != float('inf'):
                distance_to_opt_estimation = best_g.coloring_cost(cost_map) - optimal_cost_estimation
                within_tolerance_optimum = distance_to_opt_estimation <= optimal_cost_estimation * optimal_cost_tol

            print_str += f'\t\t\tOptimum estimation: ~ {optimal_cost_estimation:.3f}; distance to estimation: {distance_to_opt_estimation}\n'

            history['best_cost'].append(best_g.coloring_cost(cost_map))
            history['best_k'].append(best_g.current_coloring_number)
    
            print_str += '='*100+'\n'


            if print_enabled:
                print(print_str)
            if within_tolerance_optimum:
                break
            if deadline is not None and deadline < timedelta(seconds=time()-start_of_sa_t):
                break
            if np.isclose(temperature, 0):
                temperature = 0

            iterno += 1

    except KeyboardInterrupt:
        pass
       

    history['time'] = timedelta(seconds=time()-start_of_sa_t)
    history['iterations'] = iterno

    if print_enabled:
        stop_reason = "Iter limit"
        if deadline is not None and deadline < timedelta(seconds=time()-start_of_sa_t):
            stop_reason = f'Hit time deadline'
        if within_tolerance_optimum:
            stop_reason = f"Solution within {optimal_cost_tol} of estimated optimum"
        
        print('\n')
        print('*'*130)
        print(f'Stopped: Reason: {stop_reason}')
        print(f'Best found solution: {best_g.coloring_cost(cost_map)}')
        print(f'Best: {best_g.summary(cost_map=cost_map)}')
        print(f'Best coloring: {best_g.coloring}')
        print(f'Took {history["time"]} s')
        print('*'*130)
        print()


    return best_g.coloring, history

            
