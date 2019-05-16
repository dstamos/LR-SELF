import numpy as np
import random
import pylp
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

def create_toy_graph(size, degree):

    graph = lil_matrix((size,size))
    for i in range(size):
        for j in np.random.randint(size, size=(degree,)):
            graph[i,j] = -random.random()
    return graph

class FastSolver:

    def __init__(self, graph):

        assert(graph.shape[0] == graph.shape[1])
        self.num_nodes = graph.shape[0]

        # lookup tables from ILP vars to edges and vice versa
        self.var_to_edge = {}
        self.edge_to_var = {}
        var = 0
        for (u,v) in zip(graph.nonzero()[0], graph.nonzero()[1]):
            self.var_to_edge[var] = (u,v)
            self.edge_to_var[(u,v)] = var
            var += 1

        # initialze ILP solver and problem
        self.num_variables = var
        self.ilp_solver  = pylp.GurobiBackend()
        self.ilp_solver.initialize(self.num_variables, pylp.VariableType.Binary)
        self.objective   = pylp.LinearObjective(self.num_variables)
        self.constraints = pylp.LinearConstraints()
        self.graph = graph

        self.min_cost = 0
        for i in range(self.num_variables):
            (u,v) = self.var_to_edge[i]
            cost = graph[u,v]
            self.min_cost = min(cost, self.min_cost)
            self.objective.set_coefficient(i, cost)
        self.ilp_solver.set_objective(self.objective)

    def solve(self, max_iterations = None):

        i = 0
        while max_iterations == None or i < max_iterations:
            print("solving ILP, iteration " + str(i))
            solution = self.solve_ilp()
            if not self.add_violated_constraints(solution):
                print("global optimal solution found")
                break
            i += 1

        if i == max_iterations:
            print("maximal number of iterations reached, aborting")

        return solution

    def solve_ilp(self):
        """Solves the current ILP and returns the solution as a graph.

        The values of the edges in the returned graph are their original costs 
        plus a constant so that they are positiv (so that we can use it for 
        Dijkstra later).
        """

        ilp_solution = pylp.Solution()
        self.ilp_solver.set_constraints(self.constraints)
        message = self.ilp_solver.solve(ilp_solution)
        print("ILP solved with minimal value " + str(ilp_solution.get_value()) + " and status " + message)

        solution = lil_matrix(self.graph.shape)
        for i in range(self.num_variables):
            print("value of var " + str(i) + " is " + str(ilp_solution.get_vector()[i]))
            if ilp_solution.get_vector()[i] < 0.5:
                continue
            (u,v) = self.var_to_edge[i]
            solution[u,v] = self.graph[u,v] - self.min_cost + 1

        return solution

    def add_violated_constraints(self, solution):

        #print("testing for cycles in: " + str(solution))

        # get strongly connected components
        (num_components, labels) = connected_components(
                solution,
                directed=True,
                connection='strong')

        print("comp  : " + str(num_components))
        print("labels: " + str(labels))

        if num_components == self.num_nodes:
            print("there are no cycles :)")
            return False

        violated_constraint_found = False

        (components, sizes) = np.unique(labels, return_counts=True)
        for (component, size) in zip(components, sizes):

            if size == 1:
                continue

            violated_constraint_found = True

            component_graph = lil_matrix(solution.shape)
            for (u,v) in zip(solution.nonzero()[0], solution.nonzero()[1]):
                if labels[u] == component and labels[v] == component:
                    component_graph[u,v] = solution[u,v]

            self.add_component_constraints(component_graph)

        return violated_constraint_found

    def add_component_constraints(self, component_graph):

        marked_edges = set()

        for (u,v) in zip(component_graph.nonzero()[0], component_graph.nonzero()[1]):

            # only consider unvisited edges
            if (u,v) in marked_edges:
                continue

            # search for path from v to u
            (dist, predecessors) = dijkstra(
                    component_graph,
                    directed=True,
                    indices=[v],
                    return_predecessors=True)

            # walk back from u to v to get the cycle
            current_node = u
            cycle = [(u,v)]
            while current_node != v:
                predecessor = predecessors[0][current_node]
                cycle.append((predecessor, current_node))
                current_node = predecessor

            self.add_cycle_constraint(cycle)

            # mark cycle edges as visited
            for e in cycle:
                marked_edges.add(e)

            print("(v,u)       : " + str((v,u)))
            #print("distances   : " + str(dist))
            #print("predecessors: " + str(predecessors))
            print("cycle (u,v) : " + str(cycle))

    def add_cycle_constraint(self, cycle):

        constraint = pylp.LinearConstraint()
        for variable in [ self.edge_to_var[e] for e in cycle ]:
            constraint.set_coefficient(variable, 1)
        constraint.set_relation(pylp.Relation.LessEqual)
        constraint.set_value(len(cycle) - 1)

        self.constraints.add(constraint)

if __name__ == "__main__":

    # graph = create_toy_graph(100, 50)

    n_nods = 3
    graph = lil_matrix((n_nods, n_nods))

    graph[0, 1] = 2
    graph[1, 2] = 5
    graph[2, 0] = 1

    pylp.set_log_level(pylp.LogLevel.Debug)
    solver = FastSolver(graph)
    solution = solver.solve()

    print("original graph: " + str(graph))
    print("solution graph: " + str(solution))

    k = 1