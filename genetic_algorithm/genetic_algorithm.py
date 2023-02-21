import random                   #Used for random number generation
import statistics               #Used for mean function for code readability

#Some data visualization
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import pandas as pd


"""
randomGenome
	Generates a random individual genome of given length

Parameters:
	Name:		Description:
	-----		------------
	length		Desired length of genome

Returns:
	String of chars of specified length eg "10101011110111110100"
	for length 20

"""
def randomGenome(length):
	ret=""
	for j in range(length):
		intVal=round(random.random())
		if intVal==1: c='1'
		else: c='0'
		ret=ret+c
	return ret


"""
makePopulation
	Creates a list of length size chromosome structures, each with 
	a genome of length chrom_size

Parameters:
	Name:		Description:
	-----		------------
	size		Desired population size
	chrom_size	Desired chromosomze length
	debugging	(optional) Flag for extra debugging output

Returns:
	List of chromosome structures

"""
def makePopulation(size, chrom_size=20, debugging=False):
	ret = [] 
	for i in range(size):
		curr_individual = chromosome(randomGenome(chrom_size))
		ret.append(curr_individual)
	if debugging:
		for i in ret: print(i)
	return ret


"""
Parent Selection Function
	Implements fitness proportionate roulette selection

Parameters:
	Name:		Description:
	-----		------------
	p			list of chromosome structures
	rate        Probability of mutation for a given char
	debugging	(optional) Flag for extra debugging output

Returns:
	Does not return anything. The input list parameter p is directly manipulated


Notes:
	While sorting is possible, the sequence of candidates does not matter since
	Roulette Wheel selection is performed, so sorting has been taken out to
	optimize runtime. 
	In scenarios where initial sorting is desired, this could be implemented 
	by something similar to below:
		initial_candidates=sorted(p, key=lambda x: x.get_fitness(), reverse=True)
"""
def parent_selection(p, s, debugging=False):
	ret = []		#Parent group
	new_fitness=0	#Parent group cumulative fitness 
	"""
	The size must remain constant, however, a single parent can produce multiple offspring
	and the second generation's size will remain constant (elitism/culling)
	"""
	while len(ret)<len(p):
		r=round(random.random()*s)
		tempSum=0
		i=0
		curr_fitness=0

		#The below few lines illustrate Roulette Wheel Selection.
		while tempSum<r: 
			curr_fitness=p[i].get_fitness()
			tempSum=tempSum+curr_fitness
			i=i+1

		ret.append(chromosome(p[i-1].get_value()))
		new_fitness=new_fitness+curr_fitness

	if debugging:
		print() 
		print("-------------------------------------------------------------")
		print("Parent Selection: initial cumulative fitness: ",s)
		print("Parent Selection: repopulation group cumulative fitness: ",new_fitness)
		print("-------------------------------------------------------------")
	return ret, new_fitness


"""
Recombination Function

Parameters:
	Name:		Description:
	-----		------------
	pairs		list of chromosome structures. Assumption is made that they
	            are sequentially paired eg [{0,1},{2,3},{4,5}] since the 
				roulette algorithm selects parents with fitness weights it does
				not matter the sequence for which parents are partnered
	prob        Probability of crossover
	chrom_size  (optional) Length of each chromosome
	debugging	(optional) Flag for extra debugging output

Returns: 
	The recombined list (ret) of chromosomes calculated based on crossover probability. 
	A random number is generated, if it is within the crossover probability then 
	crossover is performed. Here another random number is generated to select the
	splitting index between the parents, and the children are generated. 


Notes:
	"Replacement" is performed here as the output here. The children may be mutated 
	afterwards before one would consider the generation officially replaced. 

	The debugging does not show a difference in sums because there will be no
	difference in sums
"""
def recombine(pairs, prob, chrom_size=20, debugging=False):
	#The selected parents are sufficient if crossover is not possible
	if prob==0: return pairs
	numCrossovers=0
	numPairs=0
	ret = [] 
	#We step by 2 since our pairs are in a list data structure
	for i in range(0, len(pairs)-1, 2):
		#eg x < .7, we perform crossover
		if random.random()<=prob: 
			numCrossovers=numCrossovers+1
			#We're directly accessing a character so indexed at 0
			#But we want it to be between 1 and 19 so that actual change occurs
			#For a string that can be indexed between 0 and 19 (slicing accesses [start,finish) )
			crossover_point = random.randint(1,chrom_size-1)
			s1=pairs[i].get_value()[0:crossover_point]+pairs[i+1].get_value()[crossover_point:]
			s2=pairs[i+1].get_value()[0:crossover_point]+pairs[i].get_value()[crossover_point:]
			#The below two lines as partial replacement and partial evaluation. 
			#The chromosome data structure evaluates itself as part of the constructor. 
			ret.append(chromosome(s1))
			ret.append(chromosome(s2))
		#Crossover is not performed, the parents replicate asexually/live on to the next gen
		else:
			numPairs=numPairs+1
			#The below two lines serve as partial replacement.
			#The chromosome data structures have already evaluated themselves
			ret.append(pairs[i])
			ret.append(pairs[i+1])
	
	if debugging: 
		print()
		print("-------------------------------------------------------------")
		print("Recombination: Number of crossovers: ",numCrossovers*2)
		print("Recombination: Number of chromosomes untouched: ", numPairs*2)
		print("-------------------------------------------------------------")

	#The below line is "replacement"
	return ret


"""
Mutation Function

Parameters:
	Name:		Description:
	-----		------------
	p			(out)list of chromosome structures
	rate        Probability of mutation for a given char
	debugging	(optional) Flag for extra debugging output

Returns:
	Does not return anything. The input list parameter p is directly manipulated.


Notes: 
	We consider "replacement" performed by recombination, and directly manipulate
	the next generation here. 

	We consider "evaluation" performed here by directly manipulating the data of the
	next generation they will already internally compute their own fitness. 

"""
def mutate(p, rate, debugging=False):
	num_mutations=0 

	if debugging: 
		sum=0
		for i in p: sum=sum+i.get_fitness()
		print("Initial sum in mutation: ",sum)
		print()

	#Iterates for each char and reverses the bit if the random chance of mutation occurs
	for i in p: 
		str = i.get_value()
		for x, c in enumerate(str):
			if random.random()<=rate: 
				num_mutations=num_mutations+1
				temp=str[0:x]
				if c=='1': temp=temp+'0'+str[x+1:]
				else: temp=temp+'1'+str[x+1:]
				i.set_value(temp)
	if debugging: 
		sum=0
		for i in p: sum=sum+i.get_fitness()
		print("Final sum in mutation: ",sum)
		print("Number of mutations: ",num_mutations)
		print() 


"""
Genetic Algorithm 

Parameters:
	Name:					Description:
	-----					------------
	max_iteration			Maximum number of iterations to call
	population_size			Population size (remains constant)
	crossover_probability	Probability for crossover (Pc)
	bitwise_mutation_rate	Probability for mutation (Pm)
	fitness					Fitness function, all_ones
	chrom_size				Length of Genome (defaults to 20)
	debugging				In-depth data for each sub-function
	convergence delta		Difference at which the population remains "static"
	logging					Printer display flag
	initial_population		A pre-initialized p. Will break if parameters are different
	output_file				Name of output file if desired

Returns:
	Convergence, step t at which the population has produced a perfect specimen (or -1 if not)
	initial_fitness, the cumulative fitness of the initial population (for data analysis)


Notes: 
	We consider "replacement" performed by recombination, and directly manipulate
	the next generation here. 

	We consider "evaluation" performed here by directly manipulating the data of the
	next generation they will already internally compute their own fitness. 

"""
def genetic_algorithm(max_iteration, population_size, crossover_probability, bitwise_mutation_rate, fitness, chrom_size=20, debugging=False, convergence_delta=.011, logging=True, initial_population=None, outputFile=None):
	#Some variables not necessarily part of genetic algorithms to keep track of convergence
	convergence = -1
	average_fitnesses=[]
	infinite_until_convergence=False 
	initial_fitness=-1

	#Output file creation and initial parameter writing 
	if not (outputFile == None): 
		fh=open(outputFile, "w+")
	if logging: 
		print("Population size: "+str(population_size))
		print("Genome length: "+str(chrom_size)) 

	#For data analysis we sometimes want to tell the algorithm to continue until fitness is not changing
	if max_iteration==-1: 
		if convergence_delta==0: convergence_delta=.001
		infinite_until_convergence=True 

	#Step and population variables, respecively
	t = 0
	if initial_population == None: 
		p=makePopulation(population_size, chrom_size, debugging)
	else:
		p=[]
		for i in initial_population: 
			p.append(chromosome(i.get_value()))

	"""
	Here is where "evaluate_population" would take place. In this particular algorithm,
	evaluation is done as part of initialization by the chromosome data structures
	"""
	while t < max_iteration or infinite_until_convergence:
		optimal_string, max_string, max_ones, avg_ones, s = fitness(p, chrom_size,debugging=debugging)
		if t==0: initial_fitness=s
		#If we are collecting large datasets, we don't actually necessarily want all of this. 
		if logging: print(t+1, avg_ones, max_ones)
		#Or if we have specified an output file then we can open it and write to it as well. 
		if not (outputFile == None): 
			line = str(t+1)+" "+str(avg_ones)+" "+str(max_ones)+"\n"
			fh.write(line) 
		average_fitnesses.append(avg_ones) 
		if max_ones==chrom_size and convergence ==-1: 
			convergence=t+1
			break
		if not convergence_delta==0 and len(average_fitnesses)>200 and abs(statistics.mean(average_fitnesses[:t])-statistics.mean(average_fitnesses))<=convergence_delta:
			if logging: print("Convergence reached at: ",t)
			break
		t = t+1
		#Gets a list of successive pairs by roulette selection
		pairs, parent_cumulative_fitness = parent_selection(p, s, debugging=debugging)
		pairs = recombine(pairs, crossover_probability, debugging=debugging)

		p=pairs
		mutate(p, bitwise_mutation_rate, debugging=debugging)
		"""
		Here is where "evaluate_population" and "replace_population" would take place. 
		In this particular algorithm evaluation is done as part of initialization by the
		chromosome data structures, which are replaced as part of recombination
		"""
	if logging: 
		if convergence == -1: 
			print("No perfect specimens found")
		else: 
			print("Optimal fitness found at iteration #",convergence)
	#Write to the file as well
	if not (outputFile == None): 
		if logging: print("results saved in file ",outputFile)
		if convergence == -1: fh.write("Failed to produce perfect specimen")
		else: fh.write(str(convergence)+"\n") 

	return convergence, initial_fitness



"""
Fitness Function

Parameters:
	Name:		Description:
	-----		------------
	list		list of chromosome structures
	debugging	flag for extra debugging output

Returns: 
	A boolean identifying if a string of all ones is found
	The string with the most ones in the list
	The amount of ones in the string above
	The average ones
"""
def all_ones(list, chrom_size=20, debugging=False):
	#There must be a population to first evaluate
	if len(list)<1: return False, "", 0, 0
	max_ones=0
	sum_ones=0
	optimal_string=False
	max_string=list[0]
	perfect_string = "" 
	for chrom in list:
		val = chrom.get_value()
		fitness = chrom.get_fitness()
		sum_ones = sum_ones + fitness
		if fitness > max_ones: 
			max_ones=fitness
			max_string=chrom
		if fitness == len(chrom.get_value()): 
			optimal_string = True
			perfect_string = chrom

	avg_ones = sum_ones/len(list)
	if debugging: 
		print(max_ones, sum_ones, optimal_string, perfect_string)
	return optimal_string, max_string, max_ones, avg_ones, sum_ones 


#Data structure that holds a string and a value, and calculates its own fitness
class chromosome: 
	#Gets a chromosome string and automatically evaluates its own fitness
	def __init__(self, _value):
		self.__value = _value
		self.__fitness = self.calculate_fitness()

	def __str__(self):
		return str(self.__value)+", "+str(self.__fitness)

	#Simply calculates the numbers of '1's in the chromosome
	def calculate_fitness(self):
		ret = 0
		for i in self.__value: 
			if i=='1': ret=ret+1
		return ret 

	def set_value(self,val):
		self.__value=val
		self.__fitness = self.calculate_fitness() 

	def get_value(self):
		return self.__value 

	def get_fitness(self):
		return self.__fitness 


#Dataset of first part with doubled timesteps so that an optimal specimen is always found
def analyze_dataset(t=50, n=100, crossover=.7, mutation=.001, funct=all_ones, chrom_size=20, sample_size=35, debugging=False, convergence_delta=0, logging=False, initial_population=None, points=False):
	dataset = []
	fitnesses = [] 
	for i in range(sample_size):
		ret, initial_fitness = genetic_algorithm(t, n, crossover, mutation, funct, chrom_size, debugging=debugging,convergence_delta=convergence_delta, logging=logging, initial_population=initial_population)
		if ret == -1: ret=t #Was continue 
		if logging: print("Current speed: ",ret, "Initial fitness: ", initial_fitness)
		dataset.append(ret)
		fitnesses.append(initial_fitness/n)

	if points: 
		return dataset, fitnesses

	mean=statistics.mean(dataset)
	stdev=statistics.stdev(dataset)
	print()
	print("---------------------------------------------------------------------------------")
	print("Testing population of",n," ", chrom_size, "chromosome size ", t," iterations, ", crossover," crossover and ",mutation," mutation rate")
	print("Mean: ",mean)
	print("Standard Deviation: ",stdev)
	print("---------------------------------------------------------------------------------")
	print()

	return mean, stdev

#Generates a basline population that has an initial average fitness mu-delta
#Only used for debugging as this leads to poor population distributions
def generate_baseline_population(size, chrom_size=20, delta=.01): 
	p=[]
	population_average=0
	while 10-delta > population_average or population_average > 10+delta: 
		population_average=0
		p=makePopulation(size, chrom_size)
		for i in p: 
			population_average=population_average+i.get_fitness()
		population_average=population_average/size
	return p, population_average

#Simply calls the algorithm with the assignment-specified parameters
def main():
	genetic_algorithm(50, 100, .7, .001, all_ones, outputFile = "run1.txt")


if __name__ == "__main__":
	main()
	exit(0)