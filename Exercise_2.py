import numpy as np
import numpy.random as rn
from scipy.optimize import minimize
import math
import time

##############################################################################
#Banana (Rosenbrock) with Conjugate-Gradient

# interval for valid Rosenbrock vectors, given by assignment
interval_rosen = np.array([-5, 5])

# random_start() generates random points in the interval provided
def random_start(interval):
    # Random point in the interval.
    a, b = interval
    return (b - a) * rn.random_sample() + a


x0_rosen = np.array([random_start(interval_rosen),
                     random_start(interval_rosen)])

def banana(x):
    return 100*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def banana_jac(x):
    return np.array((-2*100*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1]- x[0]**2)))

def banana_checker(x_curr):
    checkX1 = False
    checkX2 = False
    inbounds = False
    if x_curr[0] < 2*np.pi:
        if x_curr[0] > -2*np.pi:
            checkX1 = True
    if x_curr[1] < 2*np.pi:
        if x_curr[1] > -2*np.pi:
            checkX2 = True
    if checkX1 and checkX2:
       inbounds = True
    return inbounds

banana_start = time.time()
res1 = minimize(banana, x0_rosen, method="CG", jac=banana_jac)
banana_end = time.time()
# To-Do: clean up these print statements, use .format
print("First Analysis: Rosenbrock with CG")
print("The starting point on this run is:")
print(x0_rosen)
print("The Rosenbrock with CG Analysis:")
print(res1)
print("The Rosenbrock function cost evaluated at this point")
print(banana(res1.x))
print("The known global solution for Rosenbrock is at (1,1) with a value of:")
print(banana(np.array([1,1])))
print("Total compute time for this analysis was:")
print(banana_end - banana_start)
print("The solution is inbounds:")
print(banana_checker(res1.x))
print("")
print("Now on to Eggcrate Function with Newton-CG:")

###############################################################################
# Eggcrate function
# Eggcrate() takes a tuple x, returns the Eggcrate cost function of those
# two variables
def Eggcrate(x):
    return(x[0]**2 + x[1]**2 +25*(((np.sin(x[0]))**2)+(np.sin(x[1]))**2))

# interval is given by assignment, set here as an np.array for consistency
interval_egg = np.array([-2*np.pi, 2*np.pi])

# random_vector takes the interval provided and uses random_start to
# return an array of length 2 to be used as the start state
def random_vector(interval):
    x0 = [random_start(interval), random_start(interval)]
    return x0

# random_vector caller
x0_egg = np.array(random_vector(interval_egg))

# egg_der takes an the current position and returns the derivative value
# at that point.
# really explicit with unnecessary variables, but it works
def egg_der(x_curr):
    der = np.zeros_like(x_curr)
    der[0] = 2*x_curr[0] + 25*(math.sin(2*x_curr[0]))
    der[1] = 2*x_curr[1] + 25*(math.sin(2*x_curr[1]))
    return der

# egg_hess takes the current vector position and returns
# a 2x2 matrix with the 2nd derivatives of the eggcrate function
# this will only work on the 2 variable function
# (not a generic hessian generator) but that's ok for this purpose
def egg_hess(x_curr):
    H_egg = np.zeros((2,2), dtype = float)
    H_egg[0,0] = 2*x_curr[0] + 25*math.sin(2*x_curr[0])
    H_egg[1,1] = 2*x_curr[1] + 25*math.sin(2*x_curr[1])
    return H_egg

def egg_checker(x_curr):
    checkX1 = False
    checkX2 = False
    inbounds = False
    if x_curr[0] < 2*np.pi:
        if x_curr[0] > -2*np.pi:
            checkX1 = True
    if x_curr[1] < 2*np.pi:
        if x_curr[1] > -2*np.pi:
            checkX2 = True
    if checkX1 and checkX2:
       inbounds = True
    return inbounds

egg_start = time.time()
res2 = minimize(Eggcrate, x0_egg, method='Newton-CG', jac=egg_der, 
                hess=egg_hess, options={'xtol':1e-08, 'disp':True})
egg_end = time.time()
print("The initial point for this test:")
print(x0_egg)
print("The solution with Newton-CG Analysis:")
print(res2.x)
print("Total compute time for Eggcrate with Newton-CG Analysis:")
print(egg_end - egg_start)
print("The solution point is inbounds:")
print(egg_checker(res2.x))
print("")
print("Finally Golinski with Newton-CG:")

###############################################################################
# GOLINSKI - Trying not to use GNU Octave :)

# Golinski() takes an array x of seven (7) variables, returns the
# Golinski cost function
# TO-DO: Golinski(x0) is not evaluating to 3087
def Golinski(x):
    GolCost = 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
    return GolCost

# Bounder() checks x1 - x7, brings them in bounds if they are outside the parameters
# is there a better (faster) way to do this?
def Bounder(x):
    x_bounded = x
    if x[0] < 2.6:
        x_bounded[0] = 2.6
    if x[0] > 3.6:
        x_bounded[0] = 3.6
    if x[1] < 0.7:
        x_bounded[1] = 0.7
    if x[1] > 0.8:
        x_bounded[1] = 0.8
    if x[2] < 17:
        x_bounded[2] = 17
    if x[2] > 28:
        x_bounded[2] = 28
    if x[3] < 7.3:
        x_bounded[3] = 7.3
    if x[3] > 8.3:
        x_bounded[3] = 8.3
    if x[4] < 7.3:
        x_bounded[4] = 7.3
    if x[4] > 8.3:
        x_bounded[5] = 8.3
    if x[5] < 2.9:
        x_bounded[5] = 2.9
    if x[5] > 3.9:
        x_bounded[5] = 3.9
    if x[6] < 5.0:
        x_bounded[6] = 5.0
    if x[6] > 5.9:
        x_bounded[6] = 5.9
    return x_bounded

# In this exercise we're given the start position x0
# with values based on Table 1 from Mehmood paper
x0_gol = np.array([2.87, 0.73, 18.73, 7.86, 7.76, 3.04, 5.18])

# the 'solved' solution from the paper
xSolved = np.array([3.5000, 0.7000, 17.0000, 7.3000, 7.7153, 3.3502, 5.2867])

# GChecker validates the simplified constraints and returns an array
# of booleans. The validation of the g1 constraint is at g_assessment[0],
# for example.
# simplified/rewritten constraint functions come from Lin Golinsky Paper:
# http://downloads.hindawi.com/journals/mpe/2013/419043.pdf
# NOTE: at initial point g8 constraint is not met

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
    if (1/(110*x[5]**3))*math.sqrt(((745*x[3])/(x[1]*x[2]))**2 + 1.69*1.0e6) - 1 > 0 :
        g_assessment.append(False)
    else: g_assessment.append(True)
    #g6
    if (1/(85*x[6]**3))*math.sqrt(((745*x[4])/(x[1]*x[2]))**2 + 1.575*1.0e6) - 1 > 0 :
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

# golinski_der() takes the current position vector and returns an array with
# the derivative of the cost function, with respect to each variable, 
# evalauted at the current point.
# e.g. der_gol[0] = the derivative of the cost function with respect to the x1
# variable, evaluated at the current position
# this and hess_gol were calculated and entered by hand,
# pray for them to be correct...
# (your prayers are in vain, no way there are no errors)
def golinski_der(x):
    der_gol = np.zeros_like(x)
    der_gol[0] = (2.61797382*x[1]**2*x[2]**2+11.72869236*x[1]**2*x[2]
                  -33.84555636*x[1]**2-1.508*x[5]**2-1.508*x[6]**2)
    der_gol[1] = (5.23594764*x[0]*x[1]*x[2]**2+23.45738472*x[0]*x[1]*x[2]
                  -67.69111272*x[0]*x[1])
    der_gol[2] = 5.23594764*x[0]*x[1]**2*x[2]+11.72869236*x[0]*x[1]**2
    der_gol[3] = 0.7854*x[5]**2
    der_gol[4] = 0.7854*x[6]**2
    der_gol[5] = -3.016*x[0]*x[5]+22.4331*x[5]**2
    der_gol[6] = -3.016*x[0]*x[6]+7.4777*x[6]**2+1.5708*x[4]*x[6]
    return der_gol

# golinski_hess() takes the current position vector and returns a 7x7 array
# with the second derivatives evaluated at that position.
# e.g. H_gol[1,2] = d^2/dx1dx2 (the second derivative with respect to x1,x2)
# evaluated at the current point
def golinski_hess(x):
    H_gol = np.zeros((7,7), dtype = float)
    H_gol[0,1] = (5.23594764*x[1]*x[2]**2+23.45738472*x[1]*x[2]
                  -67.69111272*x[1])
    H_gol[0,2] = (5.23594764*x[1]**2*x[2]+11.72869236*x[1]**2)
    H_gol[0,5] = -3.016*x[5]
    H_gol[0,6] = -3.016*x[6]
    H_gol[1,0] = H_gol[0,1]
    H_gol[1,1] = (5.23594764*x[0]*x[2]**2+23.45738472*x[0]*x[2]**2
                  -67.69111272*x[0])
    H_gol[1,2] = (10.47189528*x[0]*x[1]*x[2]+23.45738472*x[0]*x[2])
    H_gol[2,0] = H_gol[0,2]
    H_gol[2,1] = H_gol[1,2]
    H_gol[2,2] = 5.23594764*x[0]*x[1]**2
    H_gol[3,5] = 1.5708*x[5]
    H_gol[4,6] = 1.5708*x[6]
    H_gol[5,0] = -3.016*x[5]
    H_gol[5,3] = H_gol[3,5]
    H_gol[5,5] = -3.016*x[0]+44.8662*x[5]+1.5708*x[3]
    H_gol[6,0] = -3.016*x[6]
    H_gol[6,4] = H_gol[6,4]
    H_gol[6,6] = -3.016*x[0]+14.9554*x[6]+44.8662*x[6]+1.5708*x[4]
    return H_gol

# call the optimizer for Golinski
gol_start = time.time()
res3 = minimize(Golinski, x0_gol, method='Newton-CG', jac=golinski_der, 
                hess=golinski_hess, options={'xtol':1e-08, 'disp':True,'return_all':True})
gol_end = time.time()

# the minimized solution res3 is not even close to the parameters. We can bring it back
# in bounds (in a very brutal, rudimentary way) using the Bounder function on res3.x
# TO-DO: A better way would be to work this into the minimization loop, though in scipy
# .optimize.minimize this is only supported for COBYLA, SLSQP and trust-constr
# Since we know gradient search is not going to work well regardless, let's not waste
# time on that now



#print("x0_egg is a: ",type(x0_gol))
#print("re3.x is a: ",type(res3.x))
solution_point_test = GChecker(res3.x)
# To-Do: clean these statements up with .format
print("The initial point was:")
print(x0_gol)
print("Initial point has a cost of:")
print(Golinski(x0_gol))
print("The constraints at the initial point:")
print(GChecker(x0_gol))
print("The Golinksi problem with Newton-CG Analysis without constraints:")
print(res3.x)
print("The cost at this point is:")
print(Golinski(res3.x))
print("Golinski Newton-CG Analysis solution's contraints:")
print(GChecker(res3.x))
print("Total compute time for the Golinski Newton-CG Analysis was:")
print(gol_end - gol_start)
print("Let's crudely push this point back within the boundaries:")
rebounded_solution = Bounder(res3.x)
print(rebounded_solution)
print("One more time let's see if that soltion fits the constraints:")
print(GChecker(rebounded_solution))
print("OK, and what is the cost at that rebounded point? Answer:")
print(Golinski(rebounded_solution))
print("Ouch. Not a great approach.")
print("The Mehmood et al paper provides us with a feasible solution:")
print(xSolved)
print("The cost of this solution is:")
print(Golinski(xSolved))
print("and if we check the parameters of the Mehmood solution:")
print(GChecker(xSolved))
print("and what if we use gradient search to 'look around' that solution?")
res4 = minimize(Golinski, xSolved, method='Newton-CG', jac=golinski_der, 
                hess=golinski_hess, options={'xtol':1e-08, 'disp':True,'return_all':True})
sol_search_start = time.time()
print(res4.x)
sol_search_end = time.time()
print(Golinski(res4.x))
print(GChecker(res4.x))
print(sol_search_end - sol_search_start)
