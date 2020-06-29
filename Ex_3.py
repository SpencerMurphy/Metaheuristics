import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl
from scipy import optimize       # to compare
import seaborn as sns
import time
from geneticalgorithm import geneticalgorithm as ga

################################################################
# This code is adapted from:
# https://perso.crans.org/besson/publis/notebooks/Simulated_annealing_in_Python.html


sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)

FIGSIZE = (19, 8)  #: Figure size, in inches!
mpl.rcParams['figure.figsize'] = FIGSIZE

def annealing(random_start,
              initTemp,
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps,
              debug):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = random_start()
    print("The start state was: ", state)
    cost = cost_function(state)
    print("The cost at start was: ", cost)
    states, costs  = [state], [cost]
    #T = initTemp
    BestSolution_cost = cost
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)*initTemp
        new_state = random_neighbour(state, fraction)
        new_cost = cost_function(new_state)
        if debug: print("Step #{:>2}/{:>2} : T = {:>5.4g}, stateX1 = {:>4.3g}, stateX2 = {:>4.3g}, cost = {:>4.3g}, new_stateX1 = {:>4.3g}, new_stateX2 = {:>4.3g}, new_cost = {:>4.3g} ...".format(step, maxsteps, T, state[0], state[1], cost, new_state[0], new_state[1], new_cost))
        if acceptance_probability(cost, new_cost, T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
        if BestSolution_cost > cost:
            BestSolution_cost = cost
            BestState = state
    print("The best state was: ", BestState)
    print("with a cost of: ", BestSolution_cost)
    print("The current state is:", state)
    print("with a cost of: ", cost_function(state))
    return state, cost_function(state), states, costs


def banana(x):
    return 100*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def Egg(x):
    """ Function to minimize."""
    return (x[0]**2 + x[1]**2 +25*(((np.sin(x[0]))**2)+(np.sin(x[1]))**2))

def clip(x):
    """ Force x to be in the interval."""
    a, b = interval

    return (max(min(x[0], b), a), max(min(x[1],b), a))

def random_start():
    """ Random point in the interval."""
    a, b = interval
    xStart = np.zeros(2)
    xStart[0] = (a + (b - a) * rn.random_sample())
    xStart[1] = (a + (b - a) * rn.random_sample())
    return xStart

def random_neighbour(x, fraction = 1):
    """Move a little bit x, from the left or the right."""
    amplitude = (max(interval) - min(interval)) * fraction / 10
    deltaX1 = (-amplitude/2.) + amplitude * rn.random_sample()
    deltaX2 = (-amplitude/2.) + amplitude * rn.random_sample()
    return clip((x[0] + deltaX1, x[1] + deltaX2))

def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p

def temperature(fraction):
    """ Example of temperature decreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))

# BANANA (ROSENBROCK) CALLER - in this script the print statements are throughout, no need to save the result
# don't forget to reset/uncomment the interval for different functions (To-Do: parameterize this)
print("First Up: Banana function with simulated annealing:")
interval = (-5,5)
ros_start = time.time()
annealing(random_start, 100, banana, random_neighbour, acceptance_probability, temperature, maxsteps=100000, debug=False);
ros_end = time.time()
print("compute time in seconds:",ros_end - ros_start)
print("")

# EGGCRATE CALLER - in this script the print statements are throughout, no need to save the result
# don't forget to reset the interval for different functions (To-Do: parameterize this)
print("Up Next: Eggcrate function with annealing function")
interval = (-2 * np.pi, 2 * np.pi)
egg_start = time.time()
annealing(random_start, 30, Egg, random_neighbour, acceptance_probability, temperature, maxsteps=100000, debug=False);
egg_end = time.time()
print("compute time in seconds:",egg_end - egg_start)
print("")

def see_annealing(states, costs):
    plt.figure()
    plt.suptitle("Evolution of states and costs of the simulated annealing")
    plt.subplot(121)
    plt.plot(states, 'r')
    plt.title("States")
    plt.subplot(122)
    plt.plot(costs, 'b')
    plt.title("Costs")
    plt.show()

def visualize_annealing(cost_function):
    state, c, states, costs = annealing(random_start, 80, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=1000, debug=False)
    see_annealing(states, costs)
    return state, c

### visualizations not yet updated to support x,y variables
#visualize_annealing(lambda x: x**3)
#visualize_annealing(lambda x: x**2)
#visualize_annealing(np.abs)
#visualize_annealing(np.cos)
#visualize_annealing(lambda x: np.sin(x) + np.cos(x))

# GOLINKSI METAHEURISTIC
# Using PyPi genetic algorithm with a penalty function
# since all variables are positively correlated, the max value is:
xMax = np.array([3.6,0.8,28,8.3,8.3,3.9,5.9])
# which has a cost of 7441.289351398401 found below:
"""
def GolinskiMax(x):
    GolMax = 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
    return GolMax
maxCost = GolinskiMax(xMax)
print(maxCost)
"""

# and so the cost function, with penatlies, is set as follows:
def Golinski(x):
    pen=0
    #g1
    if 27/(x[0]*x[1]**2*x[2])-1 > 0:
        # 8000 is comfortably above, but on the order of, the max value
        # and then we add a penalty for how far above zero the 'individual' is
        # in that constraint - this helps the algorithm learn the
        # constraints
        pen = 8000 + 1000*(27/(x[0]*x[1]**2*x[2])-1)
    #g2
    if (397.5/(x[0]*x[1]**2*x[2]**2)-1) > 0:
        pen = 8000 + 1000*(397.5/(x[0]*x[1]**2*x[2]**2)-1)
    #g3
    if (((1.93*x[3]**3)/(x[1]*x[2]*x[5]**4))-1) > 0:
        pen = 8000 + 1000*(((1.93*x[3]**3)/(x[1]*x[2]*x[5]**4))-1)
    #g4
    if (((1.93*x[4]**3)/(x[1]*x[2]*x[5]**4))-1) > 0:
        pen = 8000 + 1000*(((1.93*x[4]**3)/(x[1]*x[2]*x[5]**4))-1)
    #g5
    if (1/(110*x[5]**3))*np.sqrt(((745*x[3])/(x[1]*x[2]))**2 + 1.69*1.0e6) - 1 > 0:
        pen = 8000 + 1000*((1/(110*x[5]**3))*np.sqrt(((745*x[3])/(x[1]*x[2]))**2 + 1.69*1.0e6) - 1)
    #g6
    if (1/(85*x[6]**3))*np.sqrt(((745*x[4])/(x[1]*x[2]))**2 + 1.575*1.0e6) - 1 > 0 :
        pen = 8000 + 1000*((1/(85*x[6]**3))*np.sqrt(((745*x[4])/(x[1]*x[2]))**2 + 1.575*1.0e6) - 1)
    #g7
    if ((x[1]*x[2])/40) - 1 > 0 :
        pen = 8000 + 1000*(((x[1]*x[2])/40) - 1)
    #g8
    if ((5*x[1])/x[0]) - 1 > 0 :
        pen = 8000 + 1000*(((5*x[1])/x[0]) - 1)
    #g9
    if (x[0] / (12*x[1])) -1 > 0 :
        pen = 8000 + 1000*((x[0] / (12*x[1])) -1)
    #g10
    if ((1.5*x[5] + 1.9)/x[3]) -1 > 0 :
        pen = 8000 + 1000*(((1.5*x[5] + 1.9)/x[3]) -1)
    #g11
    if ((1.1*x[6] + 1.9)/ x[4]) - 1 > 0 :
        pen = 8000 + 1000*(((1.1*x[6] + 1.9)/ x[4]) - 1)

    GolCost = 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
    return GolCost+pen

# In this exercise we're given the start position x0
# with values based on Table 1 from Mehmood paper
x0_gol = np.array([2.87, 0.73, 18.73, 7.86, 7.76, 3.04, 5.18])

# the 'solved' solution from the paper
xSolved = np.array([3.5000, 0.7000, 17.0000, 7.3000, 7.7153, 3.3502, 5.2867])

varbound=np.array([[2.6,3.6],[0.7,0.8],[17,28],[7.3,8.3],
                   [7.3,8.3],[2.9,3.9],[5.0,5.9]])

#
algorithm_param = {'max_num_iteration': None,\
                   'population_size':1000,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.3,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=Golinski,\
            dimension=7,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

print("And Now: Golinski's Speed Reducer with an Elitist Genetic Algorithm")
gol_start = time.time()
model.run()
gol_end = time.time()
gol_time = gol_end - gol_start
print("Golinski ElitistGA compute time: ", gol_time)


# To-Do: run the algorithm multiple times, store all results and overlay convergence graphs
#convergence = model.report
#solution=model.output_dict

# To-Do: automatically pull the array from the model.output_dict and run it
# through GChecker

def GChecker(x):
    g_assessment = []
    #g1
    if (27/(x[0]*x[1]**2*x[2])-1) > 0:
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g2
    if (397.5/(x[0]*x[1]**2*x[2]**2)-1) > 0:
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g3
    if (((1.93*x[3]**3)/(x[1]*x[2]*x[5]**4))-1) > 0:
       g_assessment.append(False)
    else: g_assessment.append(True)
    #g4
    if (((1.93*x[4]**3)/(x[1]*x[2]*x[5]**4))-1) > 0:
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g5
    if (1/(110*x[5]**3))*np.sqrt(((745*x[3])/(x[1]*x[2]))**2 + 1.69*1.0e6) - 1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g6
    if (1/(85*x[6]**3))*np.sqrt(((745*x[4])/(x[1]*x[2]))**2 + 1.575*1.0e6) - 1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g7
    if ((x[1]*x[2])/40) - 1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g8
    if ((5*x[1])/x[0]) - 1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g9
    if (x[0] / (12*x[1])) -1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g10
    if ((1.5*x[5] + 1.9)/x[3]) -1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g11
    if ((1.1*x[6] + 1.9)/ x[4]) - 1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)

    return g_assessment
# workaround for now is to GChecker the previous solution
# embarassing, but it's late :)
prevSltn = [ 3.50033789,  0.70000465, 17.00594504,  7.30114005,  7.4040335,   2.90015523,\
  5.00078581]
print("Golinski EGA solution constraints: ")
print(GChecker(prevSltn))

