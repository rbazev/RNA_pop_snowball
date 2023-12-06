"""
An indivdual-based implementation of the Russian roulette model (Gavrilets, 1997) 
to simulate the accumulation of Dobzhansky-Muller incompatibilities (DMIs) in 
diverging populations composed of binary sequences.
"""


__author__ = 'Ata Kalirad, Ricardo B. R. Azevedo'

__version__ = '2.0'

import time
from copy import deepcopy
from itertools import *

import numpy as np
import random as rnd
import pandas as pd

# dict storing the fitness of sequences
fitness_dict = {}

# reset fitness dict
def reset_fitness_dict():
    fitness_dict.clear()

#  set random number generator seed
rnd.seed()

class Population(object):
    
    def __init__(self, pop_size, L, p_net, U_rate=0.1, r_prob="Free"):
        """Initialize Population objec

        Args:
            pop_size (int): N
            L (int): length of the sequences
            p_net (float): probability of a genotype being viable.
            U_rate (flort, optional): genomic muation rate. Defaults to 0.1.
            r_prob (float or str, optional): recombination probability. Defaults to Free.
        """
        self.pop_size = pop_size
        self.p_net = p_net
        self.seq_L = L
        mut_rate = U_rate/L
        self.mut_rate = mut_rate
        self.r_prob = r_prob
        start_seq = [rnd.choice([0, 1]) for _ in range(L)]
        self.population = np.array([np.array(start_seq) for _ in range(pop_size)])
        key = ''.join(map(str, start_seq))
        assert key not in fitness_dict
        fitness_dict[key] = 1.0
        neut_loc = np.zeros(pop_size)
        self.population = np.column_stack((self.population, neut_loc))
        self.start_seq = start_seq

    def introgress(self, seqA, seqB):
        """Construct all possible single introgressions

        Args:
            seqA (array)
            seqB (array)

        Returns:
            int : number of inviable single introgressions.
        """
        single_introg = []
        for i in range(len(seqA)):
            if seqA[i] != seqB[i]:
                new_array = seqA.copy()
                new_array[i] = seqB[i]
                single_introg.append(new_array)
        single_introg = np.array(single_introg)
        w_single_introg = self.cal_pop_fitness(single_introg, with_nl=False)
        n_II = len(single_introg) - np.sum(w_single_introg)
        return n_II
        
    def cal_fitness(self, array):
        """Calculate the fintess of a sequence.

        Args:
            array (array): a sequence represented as an array.

        Returns:
            float : fitness 
        """
        key = ''.join(map(str, array.astype(int)))
        if key in fitness_dict:
            return fitness_dict[key]
        else:
            w = np.random.binomial(1, self.p_net)
            fitness_dict[key] = w
            return w
        
    def update_neut_loci(self):
        """Update the value of the unlinked neutral loci in the population.
        """
        new_vals = np.random.normal(size=self.pop_size)
        self.population[:, -1] += new_vals
        
    def mutate_pop(self, pop): 
        """Mutate the sequences in the population

        Args:
            pop (numpy.ndarray): the population matrix.

        Returns:
            numpy.ndarray :  the mutated population matrix.
        """
        pop_wo_nl = pop[:, :-1]
        neut_loc = pop[:,-1]
        mut = np.random.binomial(1, self.mut_rate, size=(pop_wo_nl.shape))
        masked_pop_wo_nl = np.ma.masked_array(pop_wo_nl, mask=1-mut)
        mutated_pop = (pop_wo_nl * masked_pop_wo_nl.mask)  + (1 - masked_pop_wo_nl.data)*(1 - masked_pop_wo_nl.mask)
        mutated_pop = np.column_stack((mutated_pop, neut_loc))
        return mutated_pop
    
    def recombine_pop_single_cross(self, pop1, pop2, max_cross=False):
        """"Recombine between two populations. Recombination 
        probability (r_prob) determines if a randomly drawn sequence will recombine 
        with another randomly drawn sequences. Only a single cross-over is allowed.

        Args:
            pop1 (numpy.ndarray) 
            pop2 (numpy.ndarray) 
            max_cross (bool, optional): If true, recombination frequency will be set at 1.0. Defaults to False.

        Returns:
            numpy.ndarray :  the recombined population matrix.
        """
        np.random.shuffle(pop1)
        np.random.shuffle(pop2)
        pop_wo_nl1 = pop1[:, :-1]
        neut_loc1 = pop1[:,-1]
        pop_wo_nl2 = pop2[:, :-1]
        neut_loc2 = pop2[:,-1]
        if max_cross:
            num_mut = len(pop1)
        else:
            num_mut = np.random.binomial(self.pop_size, self.r_prob)
        locs = np.random.randint(0, self.seq_L - 1, size=num_mut)
        pop_wo_nl1_rec = pop_wo_nl1[:num_mut, :]
        pop_wo_nl2_rec = pop_wo_nl2[:num_mut, :]
        neut_loc1_rec = neut_loc1[:num_mut]
        neut_loc2_rec = neut_loc2[:num_mut]
        temp1 = np.arange(pop_wo_nl1_rec.shape[1]) <= locs[:, None]
        temp2 = np.arange(pop_wo_nl2_rec.shape[1]) > locs[:, None]
        recombinant = temp1*pop_wo_nl1_rec + temp2*pop_wo_nl2_rec
        new_neut = np.where(np.random.randint(2, size=len(neut_loc1_rec)), neut_loc1_rec, neut_loc2_rec)
        recs_w_nl = np.column_stack((recombinant, new_neut))
        new_pop1 = np.row_stack((recs_w_nl, pop1[num_mut:, :]))
        return new_pop1
    
    def recombine_free(self, pop1, pop2):
        """Recombine with multiple cross overs.

        Args:
            pop1 (numpy.ndarray) 
            pop2 (numpy.ndarray) 

        Returns:
            numpy.ndarray :  the recombined population matrix.
        """
        np.random.shuffle(pop1)
        np.random.shuffle(pop2)
        pop_wo_nl1 = pop1[:, :-1]
        neut_loc1 = pop1[:,-1]
        pop_wo_nl2 = pop2[:, :-1]
        neut_loc2 = pop2[:,-1]
        multi_rec = np.random.binomial(1, 0.5, size=(pop_wo_nl1.shape))
        mask = np.ma.masked_array(pop_wo_nl1, mask= 1 - multi_rec).mask
        recombinants = (pop_wo_nl1 * mask) + (pop_wo_nl2 * (1 - mask))
        new_neut = np.where(np.random.randint(2, size=len(neut_loc1)), neut_loc1, neut_loc2)
        recs_w_nl = np.column_stack((recombinants, new_neut))
        return recs_w_nl
    
    def cal_pop_fitness(self, pop, with_nl=True):
        """Calculate the fitness of a population.

        Args:
            pop (numpy.ndarray): The population matrix
            with_nl (bool, optional): If true, the last column of the matrix is removed. Defaults to True.

        Returns:
            numpy array: an array of fitness values.
        """
        if with_nl:
            pop_wo_nl = pop[:, :-1]
        else:
            pop_wo_nl = pop
        unique_rows, counts = np.unique(pop_wo_nl, axis=0, return_counts=True)
        w_unique = np.array([self.cal_fitness(i) for i in unique_rows])
        w_viable = np.where(w_unique==1)[0]
        w_pop = np.zeros(len(pop))
        if len(w_viable) == 0:
            return w_pop
        else:
            index = [list(np.where(np.all(pop_wo_nl == unique_row, axis=1))[0]) for unique_row in unique_rows[w_viable]]
            flattened_index = np.array([item for sublist in index for item in sublist])
            w_pop = np.zeros(len(pop))
            w_pop[flattened_index] = 1
        return w_pop
    
    def generate_offspring(self):
        """Generate offspring.

        Returns:
            numpy.ndarray :  the offspring population matrix.
        """
        current_pop = deepcopy(self.population)
        if self.r_prob == 'Free':
            current_pop2 = deepcopy(current_pop)
            rec_pop = self.recombine_free(current_pop, current_pop2)
        elif self.r_prob == 0.0:
            rec_pop = current_pop
        else:
            current_pop2 = deepcopy(current_pop)
            rec_pop = self.recombine_pop_single_cross(current_pop, current_pop2)
        mutants = self.mutate_pop(rec_pop)
        return mutants
        
    def get_next_generation(self):
        """Generate the next generation population matrix.
        """
        w_sum = 0
        while w_sum == 0:
            next_gen = self.generate_offspring()
            w_pop = self.cal_pop_fitness(next_gen)
            w_sum = np.sum(w_pop)
        next_gen_viable = next_gen[np.where(w_pop==1)]
        self.prop_live = len(next_gen_viable)/self.pop_size
        sampled_indices = np.random.choice(next_gen_viable.shape[0], size=self.pop_size, replace=True)
        self.population = next_gen_viable[sampled_indices]
        self.update_neut_loci()
        
    def get_allele_freqs(self, locus):
        """Calculate the allele frequency for a given locus.
        
        Arguments:
            locus (int) : the site for which the allele frequency is to be calculated.
        
        Returns:
            list : the allele frequency locus.
        """
        pop = self.population[:, :-1]
        unique, counts = np.unique(pop[:,locus], return_counts=True)
        return np.divide(counts, float(len(pop)))
    
    @staticmethod
    def generate_neighbours(input_array):
        """Generate all the mutations needed to specify all single mutation
        neighbors of a sequence.

        Args:
            input_array (numpy.ndarray

        Returns:
            numpy.ndarray : single-mutation neighbors
        """
        length = len(input_array)
        results = []
        for i in range(length):
            # Create a copy of the input array
            diff_array = np.copy(input_array)
            # Flip the current element (0 to 1 or 1 to 0)
            diff_array[i] = 1 - diff_array[i]
            results.append(diff_array)
        return np.array(results)
    
    @property
    def gene_diversity(self):
        """Calculate gene diversity for the entire sequence.
        
        Returns:
            float : heterozygosity for the entire sequence.
        """
        H = np.array([(1 - np.power(self.get_allele_freqs(i), 2).sum()) for i in range(self.seq_L)])
        return H
    
    @property
    def pop_robustness(self):
        """Calculate the genetic robustness of the population

        Returns:
            float : the genetic robustness.
        """
        if self.pop_size > 100:
            sample = deepcopy(self.population[:, :-1])
            np.random.shuffle(sample)
            sample = sample[:100]
        else:
            sample = deepcopy(self.population[:, :-1])
        unique_rows, counts = np.unique(sample, axis=0, return_counts=True)
        nu = 0
        for i, j in zip(unique_rows, counts):
            nei_i = self.generate_neighbours(i)
            nei_w = self.cal_pop_fitness(nei_i, with_nl=False)
            rob = np.divide(np.sum(nei_w), self.seq_L)
            freq = j/len(sample)
            nu += rob*freq
        return nu
    
    def get_IIs(self, pop1, pop2):
        """Find incompatible introgressions

        Args:
            pop1 (numpy.ndarray) 
            pop2 (numpy.ndarray) 

        Returns:
            float : the avergae number of incompatible introgressions.
        """
        if self.pop_size > 100:
            sample1 = deepcopy(pop1[:, :-1])
            np.random.shuffle(sample1)
            sample1 = sample1[:100]
            sample2 = deepcopy(pop2[:, :-1])
            np.random.shuffle(sample2)
            sample2 = sample2[:100]
        else:
            sample1 = deepcopy(pop1[:, :-1])
            np.random.shuffle(sample1)
            sample2 = deepcopy(pop2[:, :-1])
            np.random.shuffle(sample2)
        n_II = []
        for i,j in zip(sample1, sample2):
            n_II.append(self.introgress(i,j))
        return np.mean(n_II)

class TwoPops(object):

    def __init__(self, pop, burnin_t):
        """Initialize TwoPops object from Population.

        Args:
            pop (Population): An instance of Population object.
            burnin_t (int): The length of burnin phase.
        """
        self.init_pop = deepcopy(pop)
        self.pop1 = deepcopy(self.init_pop)
        self.pop2 = deepcopy(self.init_pop)
        self.burnin_t = burnin_t

    def get_RI(self):
        """Calculate reproductive isolation between the two populations.

        Returns:
            Dictionary : RI: The level of reproductive isolation based on 
                        the inviability of the single cross-over recombinants 
                        according to recombination rate (r_prob).
                         RI_max: The max level of reproductive isolation based 
                                on the inviability of all the single cross-over 
                                recombinants.
                         RI_max_free:  The max level of reproductive isolation 
                                    based on free recombination.
        """
        RI = 0
        RI_max = 0
        RI_max_free = 0
        if isinstance(self.pop1.r_prob, float) and  0 < self.pop1.r_prob < 1.0:
            pop1 = deepcopy(self.pop1)
            pop2 = deepcopy(self.pop2)
            rec_pop = self.pop1.recombine_pop_single_cross(pop1.population, pop2.population)
            rec_pop_w = pop1.cal_pop_fitness(rec_pop)
            RI = 1. - np.divide(np.sum(rec_pop_w), len(rec_pop_w))
        # RI with r=1.0
        pop1 = deepcopy(self.pop1)
        pop2 = deepcopy(self.pop2)
        rec_pop = self.pop1.recombine_pop_single_cross(pop1.population, pop2.population, max_cross=True)
        rec_pop_w = pop1.cal_pop_fitness(rec_pop)
        RI_max = 1. - np.divide(np.sum(rec_pop_w), len(rec_pop_w))
        # Free rec RI
        pop1 = deepcopy(self.pop1)
        pop2 = deepcopy(self.pop2)
        rec_pop = self.pop1.recombine_free(pop1.population, pop2.population)
        rec_pop_w = pop1.cal_pop_fitness(rec_pop)
        RI_max_free = 1. - np.divide(np.sum(rec_pop_w), len(rec_pop_w))
        return {
            'RI':  np.round(RI, decimals=3), 
            'RI_max':  np.round(RI_max, decimals=3), 
            'RI_max_free':  np.round(RI_max_free, decimals=3)
            }
    

    def get_genetic_variation(self, pop1, pop2):
        """Measure genetic variation in population.

        Args:
            pop1 (numpy.ndarray) 
            pop2 (numpy.ndarray) 

        Returns:
            Dictionary: HS: mean gene diversity within demes
                        HT: total gene diversity (pooling all demes)
                        GST: Nei's GST
                        D: Jost's D
        """

        H_within = []
        H_within.append(pop1.gene_diversity.mean())
        H_within.append(pop2.gene_diversity.mean())
        pooled = np.concatenate((pop1.population, pop2.population))
        pooled_pop = deepcopy(self.pop1)
        pooled_pop.population = deepcopy(pooled)
        HS = np.mean(H_within)
        HT = pooled_pop.gene_diversity.mean()
        if HT == 0:
            GST = 0.
        else:
            GST = 1. - (HS / HT)
        D = (HT - HS) / (1 - HS) * 2
        return {
            'GST': GST, 
            'HT': HT, 
            'HS': HS, 
            'D':D
            }
    
    def evolve_burnin(self):
        """Evolve the population.
        """
        var_n = []
        rob = []
        het = []
        viables = []
        start_index = self.burnin_t - self.burnin_t // 5
        stp = self.burnin_t//10
        for i in np.arange(0, self.burnin_t+1, 1):
            self.pop1.get_next_generation()
            if not i%stp:
                rob.append(self.pop1.pop_robustness)
                het.append(np.round(self.pop1.gene_diversity.mean(), decimals=3))
                viables.append(self.pop1.prop_live)
            if i >= start_index:
                var_n.append(np.var(self.pop1.population[:,-1]))
        self.ne_estimate = np.round(np.mean(var_n), decimals=3)
        self.pop2 = deepcopy(self.pop1)
        self.burnin_output = pd.DataFrame.from_dict({'nu': rob, 'H': het, 'prop_viab': viables})

    def diverge(self, target_D = 0.3, tot_sample=11, verbose=False):
        """Evolve two divergent populations after equilibrium.

        Args:
            target_D (float, optional): The target level of divergence between two populations. Defaults to 0.3.
            tot_sample (int, optional): The number of assays during divergence. Defaults to 11.
            verbose (bool, optional): Print the save points. Defaults to False.
        """
        output = pd.DataFrame()
        div = self.get_genetic_variation(self.pop1, self.pop2)
        curr_D = div['D']
        nu_1 = self.pop1.pop_robustness
        nu_2 = self.pop2.pop_robustness
        nu_mean = np.round(np.mean([nu_1, nu_2]), decimals=4)
        sin_II = self.pop1.get_IIs(self.pop1.population, self.pop2.population)
        seg_II_1 = self.pop1.get_IIs(self.pop1.population, self.pop1.population)
        seg_II_2 = self.pop2.get_IIs(self.pop2.population, self.pop2.population)
        seg_II = np.round(np.mean([seg_II_1, seg_II_2]),decimals=3)
        RI_assay = self.get_RI()
        div = {key: np.round(value, decimals=3) for key, value in div.items()}
        net_size = len(fitness_dict)
        output_t0 = pd.DataFrame.from_dict({'gen': [0], 'dt':[0.0], 'net_size': [net_size], 'D':div['D'], 'II': [sin_II], 'segII': [seg_II], 'nu':[nu_mean], 'HS': div['HS'], 'HT': div['HT'], 'GST':div['GST'], 'RI': RI_assay['RI'], 'RI_max':RI_assay['RI_max'], 'RI_max_free':RI_assay['RI_max_free']})
        output = pd.concat([output, output_t0], ignore_index=True)
        stp = np.linspace(0, target_D, tot_sample)[1]
        curr_target = stp
        count = 0
        n_sample = 0
        t0 = time.time()
        self.D_series = []
        while (curr_D < target_D) or (n_sample < tot_sample):
            self.pop1.get_next_generation()
            self.pop2.get_next_generation()
            div = self.get_genetic_variation(self.pop1, self.pop2)
            curr_D = div['D']
            self.D_series.append(curr_D)
            if curr_target <= curr_D < curr_target + stp:
                t1= time.time()
                delta_t = t1-t0
                RI_assay = self.get_RI()
                sin_II = self.pop1.get_IIs(self.pop1.population, self.pop2.population)
                seg_II_1 = self.pop1.get_IIs(self.pop1.population, self.pop1.population)
                seg_II_2 = self.pop2.get_IIs(self.pop2.population, self.pop2.population)
                seg_II = np.round(np.mean([seg_II_1, seg_II_2]),decimals=3)
                nu = np.round(np.mean([self.pop1.pop_robustness, self.pop2.pop_robustness]), decimals=3)
                div = {key: np.round(value, decimals=3) for key, value in div.items()}
                net_size = len(fitness_dict)
                output_t = pd.DataFrame.from_dict({'gen': [count+1], 'dt': [delta_t], 'net_size': [net_size], 'D':div['D'],'II': [sin_II], 'segII': [seg_II], 'nu':[nu_mean], 'HS': div['HS'], 'HT': div['HT'], 'GST':div['GST'], 'RI': RI_assay['RI'], 'RI_max':RI_assay['RI_max'], 'RI_max_free':RI_assay['RI_max_free']})
                output = pd.concat([output, output_t], ignore_index=True)
                n_sample += 1
                curr_target += stp
            if verbose:
                print(curr_D)
            count += 1
        return output
        
