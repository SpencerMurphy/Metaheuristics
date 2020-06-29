# Metaheuristics
Metaheuristic Analysis Projects
Metaheuristics Final Assignment
Spencer Murphy, Spring 2020 Cohort

To-Do: make this pretty :)

Exercise 2

The Banana (Rosenbrock) Function: (x, y) bounded at [-5, 5], global minimum at (1, 1) where f(x, y) = 0
Conjugate Gradient Search

Run	Starting Vector	Solution Vector	Cost at Solution	Compute Time (s)
1	[-0.08263047 -1.30948915]	[1.00000004, 1.00000231]	5.125386208250442e-12	0.00400090217590332
2	[4.83769742 4.82732319]	[1., 1.]	2.0925860881440125e-19	0.0029993057250976562
3	[ 0.2614314  -3.82967276]	[0.99999999, 0.99999971]	8.068292125466617e-14	0.0019991397857666016
4	[ 2.55141091 -2.10015311]	[0.99999992, 0.99999595]	1.5801463614247838e-11	0.003999948501586914
5	[-4.14511527 -4.19901386]	[1.00000007, 1.00000244]	5.757298231076208e-12	0.0029997825622558594
6	[-0.68575675 -1.84208405]	[1.00000008, 1.00000324]	1.0117700849518433e-11	0.0029997825622558594
7	[3.52908335 4.95113165]	[1.00000004, 1.00000204]	3.988693086738672e-12	0.002000570297241211
8	[-2.21297189 -2.67040145]	[1.00000001, 1.00000149]	2.155948133122679e-12	0.0030002593994140625
9	[ 2.24702506 -0.60440573]	[1., 1.]	4.547612546287085e-19	0.0020051002502441406
10	[-4.83248509  0.17551767]	[0.99999997, 0.99999834]	2.6596680359172107e-12	0.003000020980834961
minimize(banana, x0_rosen, method="CG", jac=banana_jac)

For the Rosenbrock function, a conjugate gradient method approach works well. While none of the solutions converge exactly to the minimum (conjugate gradient never will), it consistently points almost exactly to the minimum (0,0) and in less around 0.003 seconds run time. Since the randomized start point is already within the bounds, the shape of the banana function means we won’t have any solutions outside the parameters (which is confirmed by banana_checker() ). Parameters can be included in scipy.optimize.minimize but we can work without them in this example. With a bit of intuition from the person conducting the analysis (since we should never say a project is “fully automated” anyway!) we can easily see that the global minimum is at (0,0).

The Eggcrate Function: (x, y) bounded at [-2π, 2π], global minimum (0, 0) where f(x, y)=0
Newton-CG Search

Run	Starting Vector	Solution Vector	Cost at Solution	Compute Time (s)
1	[-5.37241228  3.88493547]	[-6.031424    3.01960188]	47.417669	
2	[-3.08136202  2.11400536]	[-3.01960188  3.01960188]	18.976395	
3	[-2.37920097 -0.52144656]	[-3.01960188e+00 -1.53097799e-09]	9.488197	
4	[ 1.89227844 -1.68340489]	[ 3.01960188 -3.01960188]	18.976395	
5	[-2.24766572  5.11526079]	[-3.01960185  6.03142398]	47.417669	
6	[ 3.48754359 -3.37094849]	[ 6.42953416e-09 -9.59621775e-09]	0.000000	
7	[-4.64052914  0.17779838]	[-3.01960188e+00  3.70363189e-08]	9.488197	
8	[ 0.47342953 -5.45510737]	[-1.55803114e-08 -6.03142399e+00]	37.929472	
9	[-0.49598649  1.49076466]	[ 1.42375651e-18 -6.67800776e-18]	0.000000	
10	[-4.51845265 -1.75767297]	[ 2.73032985e-08 -3.01960186e+00]	9.488197	
minimize(Eggcrate, x0_egg, method='Newton-CG', jac=egg_der, hess=egg_hess, options= {'xtol':1e-08, 'disp':True}

In the Eggcrate function we see the main drawback of gradient search functions: getting trapped in local minima. While the search function does sometimes get lucky and find the global minimum of (0, 0) as in runs 6 and 9, we can see from this small sample size that the success rate is around 20% and could be lower. All solutions were feasible but finding the global minimum is based on being fortunate enough to start very close to it. While a 20% success rate would work fine here, it is easy to see that with more parameters and a larger search space the success rate could quickly sink to virtually zero. An approach which allows escape from local minima is needed.

Golinksi’s Speed Reducer: multiple constraints with known feasible solutions
This problem has a known starting point provided in Table 1 of the Mehmood, et. al paper provided. Interestingly, the G8 constraint is not satisfied at the starting point. Newton-CG analysis will always lead us along the same path, although scipy.minimize.optimize() does not allow constraints on the Newton-CG method and so we quickly deviate outside the search space.  This solution does indeed have a very low cost because all the parameters are near zero. As 7 of the 11 constraints are not satisfied and all 7 parameters will be changed to their min/max by the bounder function we can surely say this is not a valuable answer. Here a crude method of pushing the results back into bounds is used as it still demonstrates that this gradient search method is not ideal. As you can see even after bringing the individual x1,…,x7 variables back within their bounds not all constraints are met and the overall cost of the function is actually higher than at the starting point. The multi-dimensionality of the problem has made the simple gradient search method unworkable and a better approach is needed. 

The feasible solution found by the Mehmood et al team (the ‘Mehmood solution’) is also analyzed. It does indeed have a lower cost and meets all but the last of the constraints (G11 is very nearly met and is likely only evaluated as false because of rounding errors).  

Golinski Newton-CG Search

Run	Vector Name	Vector	Cost	Constraint Analysis	Run Compute Time (s)
0	Starting Point	[ 2.87  0.73 18.73  7.86  7.76  3.04  5.18]	3002.6421078140806	[True, True, True, True, True, True, True, False, True, True, True]	N/A
1	Unconstrained Solution	[4.15616017e+00, 5.77090368e-07, 1.91623563e+01, 7.63787240e+00, 7.26336559e+00, 5.52438890e-01, 2.52331920e-01]	1.2630468339832406	[False, False, False, False, False, False, True, True, False, True, True]	0.010999441146850586
1	Solution Pushed Back in Bounds	[ 3.6,  0.7, 19.16235632,  7.6378724, 7.3, 2.9,  5. ]	3161.906523175484	[True, True, True, True, True, True, True, True, True, True, False]	N/A
	Mehmood solution	[3.5000, 0.7000, 17.0000, 7.3000, 7.7153, 3.3502, 5.2867]	2994.379775295455	[True, True, True, True, True, True, True, True, True, True, False]	N/A
minimize(Golinski, x0_gol, method='Newton-CG', jac=golinski_der, 
                hess=golinski_hess, options={'xtol':1e-08, 'disp':True,'return_all':True})



 
Exercise 3

The Banana (Rosenbrock) Function
Analyzed with Simulated Annealing

Run	Starting Vector	Solution Vector	Cost at Solution	Compute Time (s)
1	[0.23066668 3.3968423 ]	(1.0000520338184669, 1.0001440096699763)	2.72346976147522e-07	1.5589988231658936
2	[ 1.34047048 -1.59570355]	(1.0009017196239391, 1.0004821816692602)	8.305769889440997e-05	1.580343246459961
3	[ 3.88484532 -2.35213489]	(0.9991822416359635, 0.988600228799654)	0.00016222659931688753	1.5660171508789062
4	[-4.6728179   1.84896139]	(0.9993284989665365, 1.0036949225311147)	7.046750491531061e-05	1.620939016342163
5	[-1.84860781 -0.40860547]	(0.9989233526021779, 1.0071520723799106)	0.00020248524854142222	1.6286330223083496
6	[ 4.69988304 -0.33686598]	(0.9995821952416506, 0.9977066151481638)	1.958169950905126e-05	1.6343812942504883
7	[4.35878612 2.7774812 ]	(1.0013848258042928, 0.9772026311531758)	0.0008455448513924555	1.621711015701294
8	[ 1.32251741 -2.47124437]	(0.9995039896105423, 1.0154469832550004)	0.0002948353954777348	1.5925838947296143
9	[1.67707957 2.76344203]	(0.9992688168272451, 1.0000902899897997)	5.587196500370887e-05	1.8667020797729492
10	[-4.79165114 -2.55930333]	(1.001464086774174, 1.0218697358940345)	0.0005730565925464035	1.7149734497070312
annealing(random_start, 100, banana, random_neighbour, acceptance_probability, temperature, maxsteps=100000, debug=False) # note: use ‘debug=True’ to track iteration history

The main parameters to be “tuned” for the Simulated Annealing function annealing() are the initial temperature (initTemp), the maxsteps, and the minimum temp (found on the left side of the return(max(,)) statement in the function temperature()]. In the above test runs this minimum temp was set to 0.01 .

The initTemp parameter should be started high enough to allow escape from the local minima near the random start point, and the minimum temp should be low enough to get near the “bottom” of the area of interest. The parameter maxsteps should be as high as possible within reasonable compute time. 

Here we see that the Simulated Annealing metaheuristic approach will also work for the banana (Rosenbrock) function. As shown above the analysis returns (virtually) the global minimum every time, though this time the compute time is ~400 times what it was with the simple conjugate gradient. While computationally expensive, perhaps it will also work on the Eggcrate and Golinski problems!

The Eggcrate Function – bounded [-2π, 2π], global minimum (0, 0)
Analyzed with Simulated Annealing

Run #	Start Point	Solution Vector	Cost at Solution	Compute time (s)
1	[ 2.64321533 -5.29880933]	(-0.007062768856299362, 0.0023803874901685007)	0.0014442516584265669	1.761023998260498
2	[ 5.04315469 -1.20603131]	(-0.01152754786479343, 0.0026452009090229267)	0.0036367700807341734	1.7470314502716064
3	[5.1063472  0.59424432]	(0.008162737576026424, -0.0016323424330497716)	0.0018016284349391639	1.7687160968780518
4	[-2.0176742   4.06054316]	(-0.004879724349771464, 0.002623446297626564)	0.0007980435656627222	1.7210185527801514
5	[0.80739714 2.70992424]	(0.006789961894056118, 0.0036936350923753025)	0.0015533903268712547	1.7250161170959473
6	[ 2.96258731 -2.11744736]	(0.005146998027981242, -0.004162606250673606)	0.0011392825166520529	1.7711002826690674
7	[ 6.01402288 -3.54002732]	(0.004234089010265618, 0.006431839823911101)	0.0015416809654211078	1.7925593852996826
8	[0.77940778 4.24876765]	(-0.0015098747129296264, -0.00018384165045448597)	6.0151461111910726e-05	1.7542359828948975
9	[ 1.47269968 -5.97715986]	(0.0021151473694863143, 0.016590420962005137)	0.007271982356446239	1.7919261455535889
10	[-5.12322387  0.03321805]	(0.006293107360088257, -0.004115974698708114)	0.001470140185362612	1.9749054908752441
annealing(random_start, 30, Egg, random_neighbour, acceptance_probability, temperature, maxsteps=100000, debug=False)

The Eggcrate function is where Simulated Annealing shows its appeal – the analysis can (essentially) converge on the global minimum every run. Here a maxsteps value on the order of 100,000 is needed to be able to find (roughly) the global minimum (0, 0) consistently with an initTemp of 30 and a random start point anywhere within the boundaries. At maxsteps = 10,000 one order of magnitude is lost in the solution vector (solutions land around (±0.01, ±0.01)) and may be outside some confidence intervals. Although compute time is also considerably higher than before it is still within reason and, more importantly, we have located the global minimum (although there is never a guarantee that will happen). Still, with multiple trials it is reasonable to think we can analyze a more complicated function and have a reasonable chance of success.
 
Golinksi’s Speed Reducer: multiple constraints with known feasible solutions
Elitist Genetic Algorithm, population size 1,000, 

Run	Solution Vector	Cost	Constraint Assessment	Compute Time (s)
1	[ 3.50143818  0.70002552 17.0011846   7.3067162   7.40159272  2.90111998
  5.00089299]	2717.329790363609	[True, True, True, True, True, True, True, True, True, True, True]	91.2167911529541
2	[ 3.50060285  0.70007754 17.00393844  7.30016196  7.40122912  2.90341921
  5.00008683]	2717.63908132696	[True, True, True, True, True, True, True, True, True, True, True]	63.25876712799072
3	[ 3.50193767  0.70026113 17.01352334  7.30581511  7.40255406  2.90061339
  5.00055352]	2720.4549377416256	[True, True, True, True, True, True, True, True, True, True, True]	84.0391948223114
4	[ 3.50121049  0.70013716 17.00490514  7.30068045  7.40110067  2.90016302
  5.00070535]	2718.047495601196	[True, True, True, True, True, True, True, True, True, True, True]	79.20625877380371
5	[ 3.50033329  0.70003276 17.00581038  7.30005056  7.40211687  2.90187447
  5.00016872]	2717.4189347808156	[True, True, True, True, True, True, True, True, True, True, True]	68.89572834968567
6	[ 3.50123432  0.70000388 17.00033608  7.3017786   7.4004419   2.90032699
  5.00004979]	2716.319017779931	[True, True, True, True, True, True, True, True, True, True, True]	97.71618914604187
7	[ 3.50090462  0.70001926 17.00220823  7.30329927  7.40200237  2.90098221
  5.0000862 ]	2716.766429276701	[True, True, True, True, True, True, True, True, True, True, True]	63.56354904174805
8	[ 3.50066486  0.70002481 17.00963683  7.30018383  7.40097947  2.90065021
  5.00000038]	2717.8265594315285
	[True, True, True, True, True, True, True, True, True, True, True]	109.0639419555664
9	[ 3.50033789  0.70000465 17.00594504  7.30114005  7.4040335   2.90015523
  5.00078581]	2717.3821273899134	[True, True, True, True, True, True, True, True, True, True, True]	66.962575674057
10	[ 3.50046197  0.70005238 17.00083812  7.30008144  7.40303782  2.900523
  5.00097301]	2716.9149323894785	[True, True, True, True, True, True, True, True, True, True, True]	58.026984453201294
model=ga(function=Golinski, dimension=7, variable_type='real', variable_boundaries=varbound,     algorithm_parameters=algorithm_param)
algorithm_param = {'max_num_iteration': None,'population_size':1000,'mutation_probability':0.1,'elit_ratio': 0.3,                   'crossover_probability': 0.5,'parents_portion': 0.3,'crossover_type':'uniform',                  'max_iteration_without_improv':None} 

The elitist genetic algorithm does an excellent job searching the Golinski’s Speed Reducer problem-space, and in fact regularly returns feasible solutions better than the known feasible solution provided by Mehmood et al in all ten runs. The starting point, as before, is provided (and not compliant with constraint G8). Algorithm parameters were set as follows: population of 1,000, a mutation probability of 0.1, an ‘elite ratio’ of 0.3 (meaning the 300 best solutions are kept through each generation as “elites”), crossover probability of 0.5, parent’s portion of 0.3, and uniform crossover. The large population allows the ‘elite ratio’ be high at 30%, another 30% being parents and still have 400 children per generation. This results in convergence usually around 50-60 iterations (the run goes to 80 iterations, determined by the algorithm since it is not specified in parameters). Despite only running 80 iterations, compute time is still high because of managing such a large population. Still the convergence curves are slower and smoother than with smaller populations over more iterations, illustrating that more comparisons result in replacements. 

This problem exerts the highest computation time by far, averaging 78.195 seconds/run. Still, enough runs are possible within a narrow timeframe to demonstrate that the solution, while not necessarily the global minimum, is quite robust. Perhaps as importantly, all solutions fit all 11 constraints; the actual importance would be context-dependent but can be assumed to be of value. The Mehmood solution appears to be on the boundary of the feasible region, but all ten solutions here fit all constraints and weigh >270kg less, suggesting the global minimum is not on the boundary and proving it is not the Mehmood solution. Of course, with all ten solutions so close it is possible that they are all trapped in a local minimum as well. We can’t know if that is the case, but only that they are certainly falling into the same local minimum with the values of each variable all falling so close.
