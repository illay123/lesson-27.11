import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDATestsTravelDistToNearestLabHeuristic']


class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Max-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the distance of the remaining path of the ambulance,
         by calculating the maximum distance within the group of air distances between each two
         junctions in the remaining ambulance path. We don't consider laboratories here because we
         do not know what laboratories would be visited in an optimal solution.

        TODO [Ex.21]:
            Calculate the `total_distance_lower_bound` by taking the maximum over the group
                {airDistanceBetween(j1,j2) | j1,j2 in CertainJunctionsInRemainingAmbulancePath s.t. j1 != j2}
            Notice: The problem is accessible via the `self.problem` field.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
                distance calculations.
            Use python's built-in `max()` function. Note that `max()` can receive an *ITERATOR*
                and return the item with the maximum value within this iterator.
            That is, you can simply write something like this:
        >>> max(<some expression using item1 & item2>
        >>>     for item1 in some_items_collection
        >>>     for item2 in some_items_collection
        >>>     if <some condition over item1 & item2>)
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        return max(self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1=junction1, junction2=junction2)
             for junction1 in all_certain_junctions_in_remaining_ambulance_path
             for junction2 in all_certain_junctions_in_remaining_ambulance_path
             if junction1 != junction2)


class MDASumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining ambulance route in the following way:
        It builds a path that starts in the current ambulance's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all certain junctions (in `all_certain_junctions_in_remaining_ambulance_path`) that haven't
         been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like enforcing the #matoshim and free
         space in the ambulance's fridge). We only make sure to visit all certain junctions in
         `all_certain_junctions_in_remaining_ambulance_path`.
        TODO [Ex.24]:
            Complete the implementation of this method.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
             distance calculations.
            For determinism, while building the path, when searching for the next nearest junction,
             use the junction's index as a secondary grading factor. So that if there are 2 different
             junctions with the same distance to the last junction of the so-far-built path, the
             junction to be chosen is the one with the minimal index.
            You might want to use python's tuples comparing to that end.
             Example: (a1, a2) < (b1, b2) iff a1 < b1 or (a1 == b1 and a2 < b2).
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)

        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        junctions_to_visit = set(all_certain_junctions_in_remaining_ambulance_path) - {state.current_location} # TODO: check if needed
        current_junction_location = state.current_location
        distance_result = 0
        while len(junctions_to_visit) > 1:
            current_closest_junction = None
            index = 0
            min_distance = 0
            for current_junction in junctions_to_visit:
                current_distance = self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1=current_junction_location, junction2=current_junction)
                if index == 0:
                    min_distance = current_distance
                    current_closest_junction = current_junction
                elif current_distance < min_distance:
                    min_distance = current_distance
                    current_closest_junction = current_junction
                index += 1
            current_junction_location = current_closest_junction
            distance_result += min_distance
            junctions_to_visit = junctions_to_visit - {current_closest_junction}

        if len(junctions_to_visit) == 1:
            distance_result += self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1=current_junction_location, junction2=list(junctions_to_visit)[0])

        return distance_result


class MDAMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MST-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the distance of the remaining route of the ambulance.
        Here this remaining distance is bounded (from below) by the weight of the minimum-spanning-tree
         of the graph, in-which the vertices are the junctions in the remaining ambulance route, and the
         edges weights (edge between each junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        """
        TODO [Ex.27]: Implement this method.
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of distinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Google for how to use `networkx` package for this purpose.
              Use `nx.minimum_spanning_tree()` to get an MST. Calculate the MST size using the method
              `.size(weight='weight')`. Do not manually sum the edges' weights.
        """
        graph = nx.Graph()
        graph.add_nodes_from(nodes_for_adding=junctions)
        for first_junction in junctions:
            for second_junction in junctions:
                if first_junction == second_junction:
                    continue
                graph.add_edge(u_of_edge=first_junction,v_of_edge=second_junction,weight=self.cached_air_distance_calculator.get_air_distance_between_junctions(junction1=first_junction,junction2=second_junction))
        return nx.minimum_spanning_tree(G=graph).size(weight='weight')


class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound to the remained tests-travel-distance of the remained ambulance path.
        The main observation is that driving from a laboratory to a reported-apartment does not increase the
         tests-travel-distance cost. So the best case (lowest cost) is when we go to the closest laboratory right
         after visiting any reported-apartment.
        If the ambulance currently stores tests, this total remained cost includes the #tests_on_ambulance times
         the distance from the current ambulance location to the closest lab.
        The rest part of the total remained cost includes the distance between each non-visited reported-apartment
         and the closest lab (to this apartment) times the roommates in this apartment (as we take tests for all
         roommates).
        TODO [Ex.33]:
            Complete the implementation of this method.
            Use `self.problem.get_reported_apartments_waiting_to_visit(state)`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        problem = self.problem

        def air_dist_to_closest_lab(junction: Junction) -> float:
            """
            Returns the distance between `junction` and the laboratory that is closest to `junction`.
            """
            return min(junction.calc_air_distance_from(current_laboratory.location) for current_laboratory in problem.problem_input.laboratories)

        result_cost = 0
        tests_on_ambulance = state.get_total_nr_tests_taken_and_stored_on_ambulance()
        if tests_on_ambulance > 0:
            result_cost += tests_on_ambulance * air_dist_to_closest_lab(state.current_location)
        for current_apartment in problem.get_reported_apartments_waiting_to_visit(state):
            result_cost += current_apartment.nr_roommates * air_dist_to_closest_lab(current_apartment.location)
        return result_cost

