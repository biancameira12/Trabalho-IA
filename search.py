# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq



class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stack = util.Stack()
    trace = util.Stack()

    traveledSpace = []
    step_counter = 0


    start_state = problem.getStartState()
    stack.push((start_state, step_counter, 'START'))

    while not stack.isEmpty():
        
        # get current node
        current_state, _, action = stack.pop()
        traveledSpace.append(current_state)
        
        # record trace to the node
        if action != 'START':
            trace.push(action)
            step_counter += 1

        # goal test
        if problem.isGoalState(current_state):
            return trace.list

        successorActions = problem.getSuccessors(current_state)
        valid_children = 0

        for successor in successorActions:

            next_state = successor[0]
            next_action = successor[1]

            # do not expand repeated states
            if next_state not in traveledSpace:
                valid_children += 1
                stack.push((next_state, step_counter, next_action))

        # check if it is not successful and step backwards
        if valid_children == 0:
            while step_counter != stack.list[-1][1]:
                step_counter -= 1
                trace.pop()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    queue = util.Queue()
    trace = {}
    explored = []

    start_state = problem.getStartState()
    queue.push(start_state)
    explored.append(start_state)

    while not queue.isEmpty():
        
        # chooses the shallowest node in the frontier
        current_state = queue.pop()
        if problem.isGoalState(current_state):
            break

        # check the posible next states
        successors = problem.getSuccessors(current_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # check if the node was already explored
            if next_state not in explored:
                explored.append(next_state)
                queue.push(next_state)
                trace[next_state] = (current_state, next_action)

    actions = []
    backtrack_state = current_state # the goal state
    while backtrack_state != start_state:
        prev_state, action = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions

def updatePriorityQueue(self, item, priority):
        # update the priority queue with the nem priorities
        # differs of push function, because new item perhaps has divergent priority than the previous one
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
        return self


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()
    trace = {}
    explored = []

    start_state = problem.getStartState()
    prev_cost = 0
    trace[start_state] = [None, None, prev_cost]

    frontier = updatePriorityQueue(frontier,start_state, 0)
    explored.append(start_state)

    while not frontier.isEmpty():
        
        current_state = frontier.pop()
        if problem.isGoalState(current_state):
            break

        # check the posible next states
        successors = problem.getSuccessors(current_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]

            # avoid going back to previous states
            if next_state not in explored:
                prev_cost = trace[current_state][2]
                explored.append(next_state)
                frontier = updatePriorityQueue(frontier,next_state, next_cost + prev_cost)
                
            # update and trace with the best case
            if next_state in trace:
                if trace[next_state][2] > next_cost + prev_cost:
                    trace[next_state][2] = next_cost + prev_cost
                    trace[next_state][1] = next_action
                    trace[next_state][0] = current_state
            else:
                trace[next_state] = [current_state, next_action, next_cost + prev_cost]

    actions = []
    backtrack_state = current_state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()

    g = {}
    g[start_state] = 0
    def f(current_node): return float(g[current_node] + heuristic(current_node, problem))


    open_list = util.PriorityQueue()
    open_list.push(start_state, 0)
    open_seen = [start_state] # for 'in' operator, as PriorityQueueWithFunction records a tuple with priority
    close_list = []
    trace = {}
    trace[start_state] = [None, None, 0]

    while not open_list.isEmpty():

        current_state = open_list.pop()
        open_seen.remove(current_state)

        # check if state is goal
        if problem.isGoalState(current_state):
            break

        # get possible next states
        successors = problem.getSuccessors(current_state)
        
        for successor in successors:

            next_state = successor[0]
            next_cost = successor[2]
            next_action = successor[1]
            
            successor_cost = g[current_state] + next_cost
           
            UPDATE = False
            if next_state in open_seen:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    g[next_state] = successor_cost
                    open_list = updatePriorityQueue(open_list,item=next_state, priority=f(next_state))
            elif next_state in close_list:
                if g[next_state] <= successor_cost:
                    pass
                else: UPDATE = True
            else: UPDATE = True



            if UPDATE:
                g[next_state] = successor_cost
                open_list = updatePriorityQueue(open_list, item=next_state, priority=f(next_state))
                open_seen.append(next_state)

                if next_state in close_list:
                    close_list.remove(next_state)
                    open_seen.remove(next_state)

            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > successor_cost:
                    trace[next_state][0] = current_state
                    trace[next_state][2] = successor_cost
                    trace[next_state][1] = next_action
                    
            else:
                trace[next_state] = [current_state, next_action, successor_cost]

        close_list.append(current_state)

    # back track
    actions = []
    backtrack_state = current_state # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


def getStartNode(problem):
    return createNode(problem.getStartState())


def createNode(state, parent=None, direction=None, cost=0):
    return {'STATE': state, 'PARENT': parent, 'DIRECTION': direction, 'COST': cost}


def getActionSequence(node):

    actionSequence = []

    while node['PARENT'] != None:
        actionSequence.append(node['DIRECTION'])
        node = node['PARENT']

    if len(actionSequence) != 0:
        actionSequence.reverse()

    #actionSequence = ['West', 'West', 'West', 'West', 'South', 'South', 'East', 'South', 'South', 'West']
    return actionSequence


def getChildNode(sucessor, parentNode):
    # sucessor = [(int coordX, int coordY), string direcao, int custo]
    return createNode(sucessor[0], parentNode, sucessor[1], sucessor[2])

def genericSearch(strategy, problem, heuristic=nullHeuristic):
    node = getStartNode(problem)
    if problem.isGoalState(node['STATE']): return getActionSequence(node)

    if(strategy == 'dfs'):
        frontier = util.Stack()
        frontier.push(node)
    elif(strategy == 'bfs'):
        frontier = util.Queue()
        frontier.push(node)
    elif(strategy == 'ucs' or strategy == 'astar'):
        frontier = util.PriorityQueue()
        frontier.push(node, 0)
        heapItems = set()
    else: raise Exception, 'genericSearch receive an illegal strategy: %s!' % strategy

    explored = set()

    while frontier.isEmpty() == False:
        node = frontier.pop()
        if problem.isGoalState(node['STATE']):
            return getActionSequence(node)

        explored.add(node['STATE'])

        for sucessor in problem.getSuccessors(node['STATE']):
            if(strategy == 'dfs' or strategy == 'bfs'):
                if (sucessor[0] in explored or any(node['STATE'] == sucessor[0] for node in frontier.list)): continue
                child_node = getChildNode(sucessor, node)
                frontier.push(child_node)
            # ucs or astar
            else:
                if (sucessor[0] in heapItems or sucessor[0] in explored): continue
                child_node = getChildNode(sucessor, node)
                actions = getActionSequence(node)
                actions.append(sucessor[1])
                cost = problem.getCostOfActions(actions)
                if(strategy == 'ucs'):
                    priority = cost
                else:
                    priority = cost + heuristic(child_node['STATE'], problem)
                heapItems.add(sucessor[0])
                frontier.push(child_node, priority)

    return None






