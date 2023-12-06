__author__ = 'Ata Kalirad'

__version__ = '1.0'

import time
from copy import *
from itertools import *

import numpy as np
import random as rnd
import pandas as pd
import RNA

# dict storing the fitness of sequences
RNA_nucl = ['A', 'C', 'G', 'U']

fitness_dict = {}

# reset fitness dict
def reset_fitness_dict():
    fitness_dict.clear()

#  set random number generator seed
rnd.seed()

class Population(object):
    
    def __init__(self, ref_seq, pop_size, U_rate=1e-4, r_prob=0., alpha=12):
        self.pop_size = pop_size
        self.seq_L = len(ref_seq)
        mut_rate = U_rate/len(ref_seq)
        self.mut_rate = mut_rate
        self.r_prob = r_prob
        self.alpha = alpha
        self.alpha = alpha
        self.ref_seq = ref_seq
        self.ref_seq_struct, self.ref_seq_mfe = RNA.fold(ref_seq)
        self.start_seq = ref_seq
        self.start_seq_int = self.convertor(self.start_seq)
        self.population = np.array([np.array(self.start_seq_int) for _ in range(pop_size)])
        key = ''.join(map(str, ref_seq))
        fitness_dict[key] = 1.0
        neut_loc = np.zeros(pop_size)
        self.population = np.column_stack((self.population, neut_loc))
    
    @staticmethod
    def mutate_random(seq):
        """Mutate RNA sequence at a single randomly chosen site to a randomly
        chosen nucleotide.

        Note
        ----
        Does not calculate structure of mutant RNA sequence.

        Parameters
        ----------
        seq : str
            Sequence.

        Returns
        -------
        str
            Mutant RNA sequence.
        """
        site = np.random.randint(0, len(seq))
        nucl = [i for i in RNA_nucl if i != seq[site]]
        np.random.shuffle(nucl)
        return seq[:site] + nucl[0] + seq[site+1:]
        
    @staticmethod
    def convertor(seq, inv=False):
        if inv:
            dic = {0:'A', 1:'U', 2:'C', 3:'G'}
        else:
            dic = {'A':0, 'U':1, 'C': 2, 'G': 3}
        temp = []
        for i in seq:
            temp.append(dic[i])
        if type(temp[0]) == str:
            return ''.join(temp)
        else:
            return temp

    def introgress(self, seqA, seqB):
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
        key = self.convertor(array, inv=True)
        if key in fitness_dict:
            w = fitness_dict[key]
        else:
            w = 0.0 
            struct, mfe = RNA.fold(key)
            bp = struct.count('(')
            if bp > self.alpha:
                bp_dist = RNA.bp_distance(self.ref_seq_struct, struct) 
                if bp_dist <= self.alpha:
                    w = 1.0
            fitness_dict[key] = w
        return w
        
    def update_neut_loci(self):
        new_vals = np.random.normal(size=self.pop_size)
        self.population[:, -1] += new_vals
        
    def mutate_pop(self, pop): 
        pop_wo_nl = pop[:, :-1]
        neut_loc = pop[:,-1]
        mutation_mask = np.random.binomial(1, self.mut_rate, size=(pop_wo_nl.shape))
        new_values = np.random.randint(4, size=pop_wo_nl.shape)
        replace_condition = (mutation_mask == 1) & (new_values == pop_wo_nl)
        # Replace these entries with new random values until they are different
        while replace_condition.any():
            new_values[replace_condition] = np.random.randint(4, size=np.sum(replace_condition))
            replace_condition = (mutation_mask == 1) & (new_values == pop_wo_nl)
        # Mutate the array
        pop_wo_nl[mutation_mask == 1] = new_values[mutation_mask == 1]
        mutated_pop = np.column_stack((pop_wo_nl, neut_loc))
        return mutated_pop
    
    def recombine_pop_single_cross(self, pop1, pop2, max_cross=False):
        # ensure that pop1 and pop2 has already been shuffled, if not shuffle within this method
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
        if with_nl:
            pop_wo_nl = pop[:, :-1]
        else:
            pop_wo_nl = pop
        unique_rows, counts = np.unique(pop_wo_nl, axis=0, return_counts=True)
        w_unique = np.array([self.cal_fitness(i) for i in unique_rows])
        w_viable = np.where(w_unique==1.0)[0]
        w_pop = np.zeros(len(pop_wo_nl))
        if len(w_viable) == 0:
            return w_pop
        else:
            index = [list(np.where(np.all(pop_wo_nl == unique_row, axis=1))[0]) for unique_row in unique_rows[w_viable]]
            flattened_index = np.array([item for sublist in index for item in sublist])
            w_pop[flattened_index] = 1.0
        return w_pop
    
    def generate_offspring(self):
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
        w_sum = 0
        while w_sum == 0:
            next_gen = self.generate_offspring()
            w_pop = self.cal_pop_fitness(next_gen)
            w_sum = np.sum(w_pop)
        next_gen_viable = next_gen[np.where(w_pop==1.0)]
        sampled_indices = np.random.choice(next_gen_viable.shape[0], size=self.pop_size, replace=True)
        self.population = next_gen_viable[sampled_indices]
        self.update_neut_loci()
        
    def get_allele_freqs(self, locus):
        """Calculate the allele frequency for a given locus.
        
        Arguments:
            locus {int} -- the site for which the allele frequency is to be calculated.
        
        Returns:
            list -- the allele frequency locus.
        """
        pop = self.population[:, :-1]
        unique, counts = np.unique(pop[:,locus], return_counts=True)
        return np.divide(counts, float(len(pop)))
    
    @staticmethod
    def generate_neighbours(input_array):
        array_length = len(input_array)
        differing_arrays = []
        for i in range(array_length):
            for replacement_value in [0, 1, 2, 3]:
                if replacement_value != input_array[i]:
                    new_array = input_array.copy()
                    new_array[i] = replacement_value
                    differing_arrays.append(new_array)
        return differing_arrays
    
    @property
    def gene_diversity(self):
        """Calculate gene diversity for the entire sequence.
        
        Returns:
            float -- heterozygosity for the entire sequence.
        """
        H = np.array([(1 - np.power(self.get_allele_freqs(i), 2).sum()) for i in range(self.seq_L)])
        return H
    
    @property
    def pop_robustness(self):
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
            rob = np.divide(np.sum(nei_w), self.seq_L*3)
            freq = j/len(sample)
            nu += rob*freq
        return np.round(nu, decimals=3)
    
    def get_IIs(self, pop1, pop2):
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
        
        Parameters
        ----------
        pop : Population
            An instance of Population object.
        """
        self.init_pop = deepcopy(pop)
        self.pop1 = deepcopy(self.init_pop)
        self.pop2 = deepcopy(self.init_pop)
        self.burnin_t = burnin_t

    def get_RI(self):
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
        
        Returns
        -------
        Dictionary
            HS: mean gene diversity within demes
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
    
    def evolve_burnin(self, verbose=False):
        var_n = []
        rob = []
        het = []
        start_index = self.burnin_t - self.burnin_t // 5
        stp = self.burnin_t//10
        for i in np.arange(0, self.burnin_t+1, 1):
            if verbose:
                print(i , end=" ")
            self.pop1.get_next_generation()
            if not i%stp:
                rob.append(self.pop1.pop_robustness)
                het.append(np.round(self.pop1.gene_diversity.mean(), decimals=3))
            if i >= start_index:
                var_n.append(np.var(self.pop1.population[:,-1]))
        self.ne_estimate = np.round(np.mean(var_n), decimals=3)
        self.pop2 = deepcopy(self.pop1)
        self.burnin_output = pd.DataFrame.from_dict({'nu': rob, 'H': het})
        
    def diverge(self, target_D = 0.1, tot_sample=11, verbose=False):
        output = pd.DataFrame()
        div = self.get_genetic_variation(self.pop1, self.pop2)
        curr_D = div['D']
        nu = np.round(np.mean([self.pop1.pop_robustness, self.pop2.pop_robustness]), decimals=3)
        sin_II = self.pop1.get_IIs(self.pop1.population, self.pop2.population)
        seg_II_1 = self.pop1.get_IIs(self.pop1.population, self.pop1.population)
        seg_II_2 = self.pop2.get_IIs(self.pop2.population, self.pop2.population)
        seg_II = np.round(np.mean([seg_II_1, seg_II_2]),decimals=3)
        RI_assay = self.get_RI()
        div = {key: np.round(value, decimals=3) for key, value in div.items()}
        net_size = len(fitness_dict)
        output_t0 = pd.DataFrame.from_dict({'gen': [0], 'dt': [0.0], 'net_size': [net_size], 'D':div['D'], 'II': sin_II, 'segII': seg_II, 'nu':nu, 'HS': div['HS'], 'HT': div['HT'], 'GST':div['GST'], 'RI': RI_assay['RI'], 'RI_max':RI_assay['RI_max'], 'RI_max_free':RI_assay['RI_max_free']})
        output = pd.concat([output, output_t0], ignore_index=True)
        stp = np.linspace(0, target_D, tot_sample)[1]
        curr_target = stp
        count = 1
        t0 = time.time()
        n_sample = 0
        while (curr_D < target_D) or (n_sample < tot_sample):
            self.pop1.get_next_generation()
            self.pop2.get_next_generation()
            div = self.get_genetic_variation(self.pop1, self.pop2)
            curr_D = div['D']
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
                output_t = pd.DataFrame.from_dict({'gen': [count], 'dt': [delta_t], 'net_size': [net_size], 'D':div['D'],'II': sin_II, 'segII': seg_II, 'nu':nu, 'HS': div['HS'], 'HT': div['HT'], 'GST':div['GST'], 'RI': RI_assay['RI'], 'RI_max':RI_assay['RI_max'], 'RI_max_free':RI_assay['RI_max_free']})
                output = pd.concat([output, output_t], ignore_index=True)
                curr_target += stp
                n_sample += 1
                if verbose:
                    print(curr_D)
            count += 1
        return output