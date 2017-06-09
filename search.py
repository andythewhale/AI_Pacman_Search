# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from game import *


class SearchProblem:
    """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """

    def getStartState(self):
        """
     Returns the start state for the search problem 
     """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
    "*** YOUR CODE HERE ***"

    # The best method I got hear was my method plus a few I researched on Stack Overflow.
    # I'm going to mix and match the methods to the one that works best.
    # I'm glad I found the Stack Overflow article on DFS. It taught me a lot.

    # At first I was defining the starting position (inside the loop) with the get start function
    # However, this was misguided. This resets the point of reference each time (duh?)
    # First thing we need to do when implementing a search is to grab the frontier states.
    # util.stack() gives us what we need on a LIFO basis. So we search "all the way left" on our graph first
    # Queue is the FIFO version.
    # Util has some useful functions:
    # push puts an item into the stack.
    # pop kicks it out
    # isEmpty tells you it's empty.

    frontier = util.Stack()

    # Set just eliminates duplicate values.
    # This is where our visited stuff goes.
    visited = set()

    # this pushes the start state onto our frontier to start us.
    # it assigns an empty list with the score zero as well.

    frontier.push((problem.getStartState(), [], 0))

    # This is the magic part.
    # I got this part right minus the above within the loop.
    # while we do not have an frontier ...
    while not frontier.isEmpty():
        stata, successors, costa = frontier.pop()

        # NOTE: this is for when stuff is added to the recursive function.
        # Initially this will not be implemented.
        # NOTE: that continue reissues the while loop beginning. I learned this (:
        if (stata in visited):
            continue;

        # so when we don't continue, we add the state to the visited set.
        visited.add(stata)

        # So if we somehow get the goal... (The dots are the goal :))
        if problem.isGoalState(stata):
            return successors

        # This part was extremely helpful and I took it from the stack article.
        for state, direction, cost in problem.getSuccessors(stata):
            frontier.push((state, successors + [direction], costa))

    return [];


"""
A recursive implementation of DFS:[5]
1  procedure DFS(G,v):
2      label v as discovered
3      for all edges from v to w in G.adjacentEdges(v) do
4          if vertex w is not labeled as discovered then
5              recursively call DFS(G,w)

A non-recursive implementation of DFS:[6]

1  procedure DFS-iterative(G,v):
2      let S be a stack
3      S.push(v)
4      while S is not empty
5          v = S.pop()
6          if v is not labeled as discovered:
7              label v as discovered
8              for all edges from v to w in G.adjacentEdges(v) do 
9                  S.push(w)

"""


def breadthFirstSearch(problem):
    """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
    "*** YOUR CODE HERE ***"

    # So the difference in BFS and DFS is just how we prioritize our frontier.
    # Luckyily, we are given a FIFO function. Queue. The rest has already been defined in DFS.

    frontier = util.Queue()

    # Set just eliminates duplicate values.
    visited = set()

    # this pushes the start state onto our frontier to start us.
    # it assigns an empty list with the score zero as well.
    frontier.push((problem.getStartState(), [], 0))

    # This is the magic part.
    # I got this part right minus the above within the loop.
    while not frontier.isEmpty():
        stata, successors, costa = frontier.pop()

        # NOTE: this is for when stuff is added to the recursive function.
        # Initially this will not be implemented.
        # NOTE: that continue reissues the while loop beginning. I learned this (:
        if (stata in visited):
            continue;

        # so when we don't continue, we add the state to the visited set.
        visited.add(stata)

        # So if we somehow get the goal... (The dots are the goal :))
        if problem.isGoalState(stata):
            return successors

        # This part was extremely helpful and I took it from the stack article.
        for state, direction, cost in problem.getSuccessors(stata):
            frontier.push((state, successors + [direction], costa))

    return [];


"""
Breadth-First-Search(Graph, root):
    
    create empty set S
    create empty queue Q      

    add root to S
    Q.enqueue(root)                      

    while Q is not empty:
        current = Q.dequeue()
        if current is the goal:
            return current
        for each node n that is adjacent to current:
            if n is not in S:
                add n to S
                n.parent = current
                Q.enqueue(n)
"""


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
