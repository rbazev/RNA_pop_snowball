"""
A individual-based model of sequence divergence on a holey fitness landscape 
based on RNA folding to simulate the accumulation of Dobzhansky-Muller 
incompatibilities (DMIs) in diverging populations.
"""

__author__ = 'Ata Kalirad, Ricardo B. R. Azevedo'

__version__ = '2.0'

import os
import uuid
import pickle
from copy import *
from itertools import *

import numpy as np
import pandas as pd
import random as rnd

# ViennaRNA package python bindings
import RNA as RNA

# global
RNA_nucl = ['A', 'C', 'G', 'U']

# dict storing the secondary structures of RNA sequences during divergence
RNA_folding_dict = {}

# set
#  random number generator seed
np.random.set_state(('MT19937', np.array([
    3691495208, 2881111814, 3977485953,  126022579, 1276930617,
     355103692, 3248493791, 3009808844,  612188080,  248004424,
    1489588601,  173474438, 4039752635, 2508845774, 2622234337,
    2700397831, 1811893199, 2190136060, 2315726008, 1162460778,
    2341168633,  236659960, 3175264097, 3400454537,  427729918,
    4066770621,  567157494, 4014767970, 2930740323,  378300123,
    2705662117, 3891078126, 1960649845, 3044656210,  882045208,
    1570375463, 2086686192,  407452463, 2030931525, 2734889467,
    3712254193, 3949803070,  764947052, 2833180084, 2612938943,
    3513858645, 1012338082, 1723965053,   40253333, 3097240011,
    3472905330,  563287754,  704858225,  610145833, 2824639775,
    3671030693,  225662685, 4093017874,  488496843, 3011853058,
    3141429748, 2892388748, 1752852512, 1097583623, 3335701968,
    2741138771, 2366687650, 2909722827, 3896701472, 2855844360,
      14740992,  126288255,  556395335, 3606698449, 1990092369,
    1892289888, 1025326265, 3335170268, 2955298765, 2086040311,
    2644433388, 1986237624,  831065590, 2567078834, 3535829239,
    1597256603,  781977323, 2945733169, 3479378352, 3652557111,
    1100223342,  235212556, 2599186570,  899620665,  675417868,
    1297279698, 3980368873, 1671894382, 3219957975,  129492647,
     369423255, 1887390651,  536695139, 3467326731,  577893063,
    3628585169, 2772043849,  369219244, 1271097627, 1346409244,
    2331891903,   39930497, 2068899034,  539572370, 4195007861,
    3495378688, 3377756157, 2835342219, 3699793011, 3321615441,
    2211559076, 2398792755, 2796307031,  818646352,  355446500,
    2946711801, 1049957619,  561188288, 2829760282,   55894884,
    1568204679, 1764468784, 1959965565, 4065967902, 3887804509,
    3833073650, 3717783102, 1837449653,  528963116, 4121548680,
    2402147957, 2202929313,  747086954, 3205182257, 1631864764,
     858833100,  148465241,   17458708, 2148761251, 3002919548,
    3773743659, 2611894356, 2275521209, 3027905006, 2234470309,
    2709870512, 1052969526, 3035329785,  110428213, 2893701759,
    2512125031, 3045322315, 2322452091, 3576747394, 2006737455,
     124047895, 3870223050, 3757797920,  698743578,  701653240,
    3561309206,   39541368, 2659965257, 3356207001,  698671102,
    1967130233, 3584965340, 3302789650,  104792115,  989737788,
    1289315250, 2742066874,  943135962, 2610987463, 4156696495,
    1957093316, 1880989243,  211024555, 1594171485, 2646518040,
    1391570537, 2982210346, 3225750783, 1452478140, 1063288625,
    2782363442,  333182057, 2864780704, 3890295634, 1022925971,
     226535384, 2132360150,   74977604, 4208008791, 1697651592,
    4029637378,  397828762, 2954491996, 1120498466, 3197759375,
    2646537589, 2903140119,  580234113, 2324229766, 1485090247,
    3173462698, 1441000100, 3212564317,  598271368, 1052134622,
    2751284206, 4040281713, 2630844601, 1921303308,  861775468,
    3522939180, 2855935558, 3227004083, 4121725263,  805407916,
    1207185676,  785322196, 3104463214, 3070205549, 1984686779,
       5199855, 2585264490, 3703002136, 3352578045,  257641487,
    1613285168, 3845545412, 2884412656, 3795140597, 2864082431,
    1708426814,  661272124, 3359489670, 2989690080, 1120054048,
    3029239860, 2037244341, 3411962036, 3468887812, 1294329307,
    1967939294, 1668712931, 1560596708, 2986374405, 3266952874,
    1758277657, 3876598642, 1149698899, 1548677880, 2464327872,
     466262570, 2573332645, 3577605405, 3511489634, 3001210402,
    4047160993, 1096981688, 1365437714,  967187969, 2651685599,
    4258218418,  618336653, 1813338507, 4161534170, 1206855048,
    3766692676, 1984622584, 1256641952, 2293866774, 2566572107,
    1296931689,  202959755, 3331103372, 3095866549, 1832670718,
    3588629070,  533366259,  301078755, 1299816886, 2612908898,
    1142385071, 4044229138,  392786907, 1473264101,  171872184,
    2873022820, 1878820461,   88690985, 3019565333, 2121461097,
    1522107992, 1733374438, 2311932879,  556408593, 1461835210,
    1423528436,  819211315,  889069790, 3086689727, 1730639543,
    1216615289, 2492159266, 1809961698, 1659780200, 3125102201,
    1711752707, 2723337471, 2521518355, 3884672928, 1313721188,
    1901655237, 3962083231,  757934816, 2196008247, 2111842931,
    2965600004, 1312840433, 3455017541,  545137641, 2279641585,
    2939005091, 1537081838, 2463922331, 1996015762, 1196027276,
     906621855, 1704400250,   76236737,  136244169,  619138087,
      98595120,  719278264, 1334390246, 3171154143, 1280182795,
    2215843496, 2676742417, 2197843524, 1396698993,  609335212,
     723295525, 3605167513, 4155694342, 3017089897, 1955520678,
    4067049686, 3239743094, 1221155545, 4095319239,  425400349,
    1806147353, 3671105575,  627163234, 1861707767,  274296576,
     638507216, 1649469686,  608691281, 4232809768,  611030651,
     853789168, 1733062866,  540453354,   11996619, 2695864391,
    2050310856,  141509199,  252149019, 3547463915,  329855083,
    2856249739, 3735981321, 2875626876, 2379144635,   13062386,
    1562227109, 1191505353, 3203340427, 2778675184, 2770557127,
    3644383877, 1790071106, 2240228460, 1676798968,  863141840,
    1175886689, 1178806726,  358678487, 3328835908, 2633561969,
    4074930335,  772447425, 3430950121, 3352113867,  701629620,
      25420967, 3791888554, 1412926413,  791735289,  161600651,
     506627594, 4220683170,  539553216,  176491711,  870303302,
    2405928427,  673609577,  616683903, 2009922698, 2088461621,
     631204850,  495792565, 1105952597, 1332646700,   23124919,
    2539330986, 1231655942, 1860851071, 3651186456, 2775290123,
    3681258608,  637100105, 4220549846, 3186875083, 3856908269,
    3867761132, 3985657986, 4173577631,  552539584, 2204479092,
    4165177831, 2396591349, 3474222162, 2920321345, 3906718099,
     515536637,  991766590, 2116510279,  482084635, 4005496942,
     374235227, 1711760850, 3750465691,  101652558, 3589303631,
    1360138030, 1382922742,  340163774, 2692240084, 2626346609,
    3041178492, 3616792294,  699158099, 1180482576, 3504356230,
    1897868877,  464615571, 3149754153, 2219112250, 2421357980,
    3182082688, 3145015709, 2579307737, 3490881071, 2970802492,
    3235037551, 1994684505,  355293861, 2682386071, 1408942224,
    3272168205, 3715571520,  476379336, 3644917929,  666542692,
    2680727545,  560661664, 1022989241,  806139402,  495605276,
     462775794, 2795097035, 1348129402, 4137368209, 2768709750,
    2129930451,  422284347, 1297682726, 1252742143, 3031031382,
      75134366, 3411139976, 1654986716,  532012083, 1253013106,
    1814002341,  584805750, 4151151859,  279516416, 2068669679,
    1452548111,  255585988, 2731877417,  805942443, 3209104026,
    1105115396, 1929339947, 3829736722, 2980275336, 2169476831,
     784792828, 3572862771, 1057808935, 1774004947, 3086076921,
     969435958, 4291618669,  892653473, 2713995907, 2137887400,
    2565641007, 1417836736,  415508859, 1624683723,   23763112,
     518111653, 2355447857, 2023934715,  934168085, 2250448450,
     450387908, 1069332538, 4170085337, 2145735300, 2298032455,
    1437026749, 2863147795, 3273446986, 1979692197, 3208629490,
    2080357079,  584771674, 1802076639, 2018580439, 4261231470,
    1708636029, 3602321445,   18885205, 1940272685, 4187271341,
    1647123067, 1450487947, 3463781280, 3759557524,  493883757,
    3901885447, 3190687437,  742916954, 3176758487, 3010187255,
     936898923, 1805555016, 1981187811, 1196213096, 3067885662,
    2550095824, 3396199635, 3614915928, 1977375679, 2173583078,
    2643789240, 2587955166, 2158941995, 2347766906, 1711205114,
      66633020, 3977356823, 1510661526, 3048960083,   51672689,
    3596587592, 4038438382, 4019922490, 2146383929, 1692948176,
    1233895739, 3938222851, 2698966080, 2950467396, 1878048591,
    3547155317, 3627364723,  906814924, 1075129814, 3302437944,
    2756803960, 2719380291, 1774084191, 2789415893, 4095723844,
    1297221824, 1938199324, 4112704123, 1741415251, 1105144176,
    1259977468,  131064353, 4036118418,  311279014], dtype=np.uint32),
    624, 0, 0.0))

class RNASeq(object):
    """RNA sequence and corresponding minimum free energy (MFE) secondary
    structure.

    Attributes
    ----------
    L : int
        Sequence length.
    bp : int
        Number of base pairs.
    mfe : float
        MFE.
    seq : str
        Sequence.
    struct : str
        MFE secondary structure.
    """

    def __init__(self, seq):
        """Initialize RNASeq object from RNA sequence.

        Note
        ----
        Does not check whether sequence contains invalid nucleotide(s).

        Parameters
        ----------
        seq : str
            Sequence.

        Examples
        --------
        >>> seq = RNASeq('UAAAUCGGCGUUCUGCCACAGCAGCGAA')
        >>> seq.struct
        '........(((((((...))).))))..'
        >>> seq.mfe
        -4.199999809265137
        >>> seq.bp
        7
        """
        self.seq = seq
        self.L = len(seq)
        self.update_struct()
        self.bp = self.struct.count('(')

    def update_struct(self): 
        """Calculate MFE and secondary structure of current RNA sequence.
        Check if MFE and secondary structure have already been calculated.

        Note
        ----
        Uses ViennaRNA package.
        """
        if self.seq in RNA_folding_dict:
            self.struct, self.mfe = RNA_folding_dict[self.seq]
        else:
            mfe_struct = RNA.fold(self.seq)
            self.struct, self.mfe = mfe_struct
            RNA_folding_dict[self.seq] = mfe_struct

    @staticmethod
    def random_sequence(L):
        """Generate a random RNA sequence of a certain length.

        Parameters
        ----------
        L : int
            Sequence length.

        Returns
        -------
        RNASeq
            Random sequence.

        Examples
        --------
        >>> L = 100
        >>> unfolded = RNASeq('.' * L)
        >>> seq = RNASeq.random_sequence(L)
        >>> seq.bp == seq.get_bp_distance(unfolded)
        True
        """
        seq_array = np.random.choice(RNA_nucl, size=L)
        return RNASeq(''.join(seq_array))

    @staticmethod
    def convertor(seq, inv=False):
        """Convert a sting to int or vice versa.

        Keyword Arguments:
            inv {bool} -- if True takes a list of digits and converts it into a 
            strings, else converts a string into a list of digits 
            (default: {False}).
        
        Returns:
            list or str 
        """
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

    @staticmethod
    def mutate(seq, site, nucl):
        """Mutate RNA sequence at specific site(s) to specific nucleotide(s).

        Note
        ----
        Does not calculate structure of mutated sequence.

        Parameters
        ----------
        seq : str
            RNA sequence.
        site : int or list
            Site(s).
        nucl : str or list
            Nucleotide(s).

        Returns
        -------
        str
            Mutated sequence

        Examples
        --------
        >>> RNASeq.mutate('AAAAAAA', [0, 1, 4], ['C', 'G', 'U'])
        'CGAAUAA'
        """
        if type(site) is list:
            assert len(site) == len(nucl)
            for i, j in zip(site, nucl):
                seq = seq[:i] + j + seq[i + 1:]
            return seq
        else:
            return seq[:site] + nucl + seq[site + 1:]

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
        return RNASeq.mutate(seq, site, nucl[0])

    @staticmethod
    def get_hamdist(seqA, seqB):
        """Calculate Hamming distance between two sequences.

        Parameters
        ----------
        seqA : str
            Sequence.
        seqB : str
            Sequence.

        Returns
        -------
        int
            Hamming distance.

        Examples
        --------
        >>> seqA = 'UAAAUCGGCGUCCUGCCACAGCAGCGAA'
        >>> seqB = 'UAAAUCGGCGUUCUGCCACAGCAGCGAA'
        >>> RNASeq.get_hamdist(seqA, seqB)
        1
        """
        assert type(seqA) == type(seqB)
        diffs = 0
        for ch1, ch2 in zip(seqA, seqB):
            if ch1 != ch2:
                diffs += 1
        return diffs

    @staticmethod
    def get_diverged_sites(seq1, seq2):
        """Find diverged sites between two sequences.

        Parameters
        ----------
        seq1 : str
            Sequence.
        seq2 : str
            Sequence.

        Returns
        -------
        tuple
            Three elements:
            seq1_nucls: list
                Diverged nucleotides on seq1.
            seq2_nucls: list
                Diverged nucleotides on seq2.
            sites: list
                sites of diverged sites.

        Examples
        --------
        >>> seqA = 'UAAAUCGGCGUCCUGCCACAGCAGCGAA'
        >>> seqB = 'UAAAUCGGCGUUCUGCCACAGCAGCGAA'
        >>> RNASeq.get_diverged_sites(seqA, seqB)
        (['C'], ['U'], [11])
        """
        assert len(seq1) == len(seq2)
        seq1_nucls = []
        seq2_nucls = []
        sites = []
        for i in range(len(seq1)):
            if seq1[i] != seq2[i]:
                seq1_nucls.append(seq1[i])
                seq2_nucls.append(seq2[i])
                sites.append(i)
        return seq1_nucls, seq2_nucls, sites

    @staticmethod
    def introgress(recipient, donor, n_introgr):
        """Construct all possible introgressions of between 1 and 4 diverged
        nucleotides from a donor sequence to a recipient sequence.

        Parameters
        ----------
        recipient : str
            Sequence.
        donor : str
            Sequence.
        n_introgr : int
            Number of nucleotides to introgress.

        Returns
        -------
        list
            Introgressed genotypes.

        Examples
        --------
        >>> RNASeq.introgress('AAAAAAA', 'AAUAUAU', 2)
        ['AAUAUAA', 'AAUAAAU', 'AAAAUAU']
        """
        assert len(recipient) == len(donor)
        assert 1 <= n_introgr <= 4
        indices = []
        temp = RNASeq.get_diverged_sites(recipient, donor)
        if temp != 0:
            indices = temp[2]
        introgress = []
        if n_introgr == 1:
            for i in range(len(recipient)):
                if recipient[i] != donor[i]:
                    introgress.append(recipient[:i] + donor[i] +
                        recipient[i + 1:])
        else:
            indices_introg = list(combinations(indices, n_introgr))
            for i in indices_introg:
                if n_introgr == 2:
                    introgress.append(recipient[:i[0]] + donor[i[0]] +
                        recipient[i[0] + 1:i[1]] + donor[i[1]] +
                        recipient[i[1] + 1:])
                elif n_introgr == 3:
                    introgress.append(recipient[:i[0]] + donor[i[0]] +
                        recipient[i[0] + 1:i[1]] + donor[i[1]] +
                        recipient[i[1] + 1:i[2]] + donor[i[2]] +
                        recipient[i[2] + 1:])
                elif n_introgr == 4:
                    introgress.append(recipient[:i[0]] + donor[i[0]] +
                        recipient[i[0] + 1:i[1]] + donor[i[1]] +
                        recipient[i[1] + 1:i[2]] + donor[i[2]] +
                        recipient[i[2] + 1:i[3]] + donor[i[3]] +
                        recipient[i[3] + 1:])
        return introgress

    @staticmethod
    def get_all_recombinants(seq1, seq2):
        """Generate all recombinants of two RNA sequences resulting from a
        single crossover event.

        For a sequence of length L, there will be 2 * (L - 1) recombinants.

        Parameters
        ----------
        seq1 : str
            Sequence.
        seq2 : str
            Sequence.

        Returns
        -------
        list
            Recombinants.

        Examples
        --------
        >>> RNASeq.get_all_recombinants('AAAA', 'UUUU')
        ['AUUU', 'UAAA', 'AAUU', 'UUAA', 'AAAU', 'UUUA']
        """
        assert len(seq1) == len(seq2)
        recs = []
        for i in np.arange(1, len(seq1), 1):
            recs.append(seq1[:i] + seq2[i:])
            recs.append(seq2[:i] + seq1[i:])
        return recs
    
    @staticmethod
    def creat_multi_recs(seq1, seq2):
        """Generate all recombinants of two RNA sequences resulting from free
        recombination.

        By default 100 recombinants are generated.

        Parameters
        ----------
        seq1 : str
            Sequence.
        seq2 : str
            Sequence.

        Returns
        -------
        list
            Recombinants.
        """
        rec_probs = np.random.binomial(1, 0.5, (100, 100))
        recs = []
        for i in rec_probs:
            recombinant = []
            site = 0
            for j in i:
                if j:
                    recombinant.append(seq2[site])
                else:
                    recombinant.append(seq1[site])
                site += 1
            temp = ''.join(recombinant)
            recs.append(temp)
        return np.array(recs)

    @staticmethod
    def get_neighbors(seq):
        """Generate all the mutations needed to specify all single mutation
        neighbors of a sequence.

        Parameters
        ----------
        seq : str
            Sequence.

        Returns
        -------
        seqs : list
            A list of tuples of the form (site, nucleotide).

        Examples
        --------
        >>> RNASeq.get_neighbors('AAA')
        [(0, 'C'), (0, 'G'), (0, 'U'), (1, 'C'), (1, 'G'), (1, 'U'), (2, 'C'), (2, 'G'), (2, 'U')]
        """
        seqs = []
        for i in range(len(seq)):
            for j in RNA_nucl:
                if j != seq[i]:
                    seqs.append((i, j))
        return seqs

    def get_bp_distance(self, other):
        """Calculate base-pair distance between the secondary structures of
        two RNA sequences.

        Note
        ----
        Uses ViennaRNA package.

        Parameters
        ----------
        other : RNASeq

        Returns
        -------
        int
            Base-pair distance.

        Examples
        --------
        >>> seq1 = RNASeq('UAAAUCGGCGUCCUGCCACAGCAGCGAA')
        >>> seq2 = RNASeq('UAAAUCGGCGUUCUGCCACAGCAGCGAA')
        >>> seq1.get_bp_distance(seq2)
        13
        """
        assert type(self) == type(other)
        return RNA.bp_distance(self.struct, other.struct)

class Population(object):
  
    def __init__(self, seq, pop_size, mut_rate=1e-4, rec_rate=0., alpha=12, t_burnin=200):
        """Initialize Population object from reference RNA sequence and alpha.

        The reference sequence and alpha define the fitness landscape.  A
        sequence is viable if its secondary structure (1) has more than alpha
        base pairs and (2) is at most alpha base pairs away from the structure
        of the reference sequence; a sequence is inviable otherwise.

        Set ancestor to reference sequence (no burn-in).

        Parameters
        ----------
        seq : str
            Reference sequence.
        alpha : int, optional
            alpha.
        """
        assert type(pop_size) == int and pop_size > 0
        assert type(mut_rate) == float and 0 <= mut_rate <= 1.
        assert type(rec_rate) == float and 0 <= rec_rate <= 1. or rec_rate == 'Free'
        self.ref_seq = RNASeq(seq)
        assert self.ref_seq.bp > alpha
        self.ancestor = RNASeq(seq)
        self.alpha = alpha
        self.u = mut_rate
        self.r = rec_rate
        self.burnin(t_burnin)
        self.population = np.array([RNASeq.convertor(self.ancestor.seq) for i in range(pop_size)])
        if rec_rate == 'Free':
            self.free_rec = True
        else:
            self.free_rec = False

    @property
    def N(self):
        return len(self.population)

    def burnin(self, t):
        """Evolve ancestor (initially set to equal the reference sequence) for
        t (viable) substitutions allowing multiple hits.  Update ancestor at
        the end.  Set lineages A and B to ancestor.

        Parameters
        ----------
        t : int
            Length of burn-in period.
        """
        count = 0
        while count < t:
            fix = False
            while not fix:
                mut = deepcopy(self.ancestor)
                mut = RNASeq(RNASeq.mutate_random(mut.seq))
                fix = self.is_viable(mut)
            self.ancestor = mut
            count += 1

    def change_pop(self, new_pop):
        self.population = deepcopy(new_pop)

    @property
    def w_pop(self):
        w_pop = [self.is_viable(RNASeq(RNASeq.convertor(i, inv=True))) for i in self.population]
        return np.sum(w_pop)/float(len(w_pop))

    def get_allele_freqs(self, locus):
        """Calculate the allele frequency for a given locus.
        
        Arguments:
            locus {int} -- the site for which the allele frequency is to be calculated.
        
        Returns:
            list -- the allele frequency locus.
        """
        allele_freqs = []
        unique, counts = np.unique(self.population[:,locus], return_counts=True)
        allele_freqs = np.divide(counts, float(self.N))
        return allele_freqs
    
    @property
    def gene_diversity(self):
        """Calculate gene diversity for the entire sequence.
        
        Returns:
            float -- heterozygosity for the entire sequence.
        """
        H = np.array([(1 - np.power(self.get_allele_freqs(i), 2).sum()) for i in range(self.ancestor.L)])
        return H

    @property
    def genotypes_dic(self): 
        """Return a dictionary of genotypes with their respective frequencies.
        
        Returns:
            dic -- the dictionary of genotypes with sequences as keys and frequencies as values.
        """
        unique, counts = np.unique(self.population, return_counts=True, axis=0)
        return dict(zip([RNASeq.convertor(i, inv=True) for i in unique], np.divide(counts, float(self.N)))) #add a small random numbers 0 < <1/(N*N) to uniques. 
 
    @property
    def wt_seq(self):
        """Get the most common sequence in the population.
        
        Returns:
            string -- the most common sequence.
        """
        return max(self.genotypes_dic, key=self.genotypes_dic.get) 

    def is_viable(self, seq):
        """Evaluate whether a sequence is viable (More extensive docs)

        See Evolve.__init__() for more details.

        Parameters
        ----------
        seq : RNASeq
            Sequence.

        Returns
        -------
        bool
            Viability.
        """
        assert type(self.ref_seq) == type(seq)
        if seq.bp <= self.alpha:
            return False
        else:
            bp = self.ref_seq.get_bp_distance(seq)
            if bp <= self.alpha:
                return True
            else:
                return False

    def recombine_pop(self, other=False):
        """Recombine within population or between two populations. Recombination 
        probability (r) determines if a randomly drawn sequence will recombine 
        with another randomly drawn sequences. Only a single cross-over is allowed.
        
        Parameters
        ----------
        other : bool, optional
            Recombine two samples drawn form the population, 
            otherwise recombine two samples drawn from self.population and 
            other.population, by default False

        rec_max : bool, optional
            If true, recombination frequency will be set at 1.0, by default False
        
        Returns
        -------
        2d numpy array
            The recombined population matrix.
        """
        if self.free_rec:
            rec_freq = 1.0
        else:
            rec_freq = self.r            
        if not other:
            sample_1 = self.population[np.random.randint(self.population.shape[0], size= self.population.shape[0]), :]
            sample_2 = self.population[np.random.randint(self.population.shape[0], size= self.population.shape[0]), :]
        else:
            sample_1 = self.population[np.random.randint(self.population.shape[0], size= self.population.shape[0]), :]
            sample_2 = other.population[np.random.randint(other.population.shape[0], size= other.population.shape[0]), :]
        rec_prob_num = np.random.binomial(1, rec_freq, size=self.N).sum()
        recs_1 = sample_1[:rec_prob_num]
        recs_2 = sample_2[:rec_prob_num]
        rec_pos = np.random.randint(1, len(sample_1[0]), size=len(recs_1))
        for i,j in zip(range(len(recs_1)), rec_pos):
                recs_1[i][j:,] = 0
                recs_2[i][:j] = 0
        recs = recs_1 + recs_2
        return np.concatenate((recs, sample_1[rec_prob_num:]))

    def recombine_multi(self, other=False):
        """Recombine with multiple cross overs.
        
        Parameters
        ----------
        other : bool, optional
            Population if recombination between populations is needed, by default False
        
        Returns
        -------
        2d numpy array
            The recombined population matrix.
        """
        if other:
            mat_1 = self.population[np.random.randint(self.population.shape[0], size= self.population.shape[0]), :]
            mat_2 = other.population[np.random.randint(other.population.shape[0], size= other.population.shape[0]), :]
        else:
            mat_1 = self.population[np.random.randint(self.population.shape[0], size= self.population.shape[0]), :]
            mat_2 = self.population[np.random.randint(self.population.shape[0], size= self.population.shape[0]), :]
        rec_probs = np.random.binomial(1, 0.5, (mat_1.shape[0], mat_1.shape[1]))
        recs = []
        for i,j,k in zip(mat_1, mat_2, rec_probs):
                recombinant = []
                site = 0
                for z in k:
                    if z:
                        recombinant.append(j[site])
                    else:
                        recombinant.append(i[site])
                    site += 1
                recs.append(recombinant)
        return np.array(recs)

    def mutate_pop(self, pop): 
        """Mutate the population.
        
        Arguments:
            pop {2d numpy array} -- The matrix representation of the population.        
        
        Returns:
            2d numpy array -- The mutated population matrix.
        """

        mut = np.random.binomial(1, self.u, size=(pop.shape)) # mutations per site for the entire population
        x, y = np.where(mut == 1)
        for i,j in zip(x,y):
            new_allele = rnd.choice([k for k in range(0, 4) if k != pop[i,j]]) # number of alleles = 4
            pop[i,j] = new_allele
        return pop

    def get_viable_offspring(self, offspring):
        """Generate viable offspring
        
        Arguments:
            offspring {2d numpy array} -- The offspring, viable and inviable.
        
        Returns:
            2d numpy array -- An array containing the viable offspring.
        """
        viable_offspring = []
        for i in offspring:
            if self.is_viable(RNASeq(RNASeq.convertor(i, inv=True))):
                viable_offspring.append(i)
        if len(viable_offspring) == 0:
            return self.population
        else:
            return np.array([rnd.choice(viable_offspring) for i in range(self.N)])

    def get_next_generation(self):
        """Generate the next generation population matrix.
        """
        if self.free_rec:
            recombinants = self.recombine_multi()
        else:
            if self.r == 0:
                recombinants = deepcopy(self.population)
            else:
                recombinants = self.recombine_pop()
        mutants = self.mutate_pop(recombinants)
        next_gen = self.get_viable_offspring(mutants)
        self.change_pop(next_gen)

    def mixis(self):
        """Reproduce without selection.
        
        Returns
        -------
        float
            The mean fitness of offspring.
        """
        if self.free_rec:
            recombinants = self.recombine_multi()
        else:
            if self.r == 0:
                recombinants = deepcopy(self.population)
            else:
                recombinants = self.recombine_pop()
        mutants = self.mutate_pop(recombinants)
        num_mut = len(mutants)
        w_progeny = np.zeros(num_mut)
        for i in range(num_mut):
            if self.is_viable(RNASeq(RNASeq.convertor(mutants[i], inv=True))):
                w_progeny[i] = 1.
        return np.sum(w_progeny)/float(len(w_progeny))

    @property
    def rec_load_max(self):
        """Calculate max recombination load.
        
        Returns
        -------
        float
            Max recombination load.
        """
        w_recs = np.zeros(self.N)
        recombinants = self.recombine_multi()
        for i in range(len(recombinants)):
            if self.is_viable(RNASeq(RNASeq.convertor(recombinants[i], inv=True))):
                w_recs[i] = 1.
        return 1. - np.sum(w_recs)/float(len(w_recs))

    @property
    def rec_load_realized(self):
        """Calculate realized recombination load.
        
        Returns
        -------
        float
            Realized recombination load.
        """
        w_recs = np.zeros(self.N)
        if self.free_rec:
            recombinants = self.recombine_multi()
        else:
            recombinants = self.recombine_pop()
        for i in range(len(recombinants)):
            if self.is_viable(RNASeq(RNASeq.convertor(recombinants[i], inv=True))):
                w_recs[i] = 1.
        return 1. - np.sum(w_recs)/float(len(w_recs))

    def get_inviable_introgressions(self, recipient, donor, n_introgr):
        """Find inviable introgressions from donor sequence to recipient
        sequence.

        Parameters
        ----------
        recipient : str
            Sequence.
        donor : str
            Sequence.
        n_introgr : int
            Number of nucleotides to introgress.

        Returns
        -------
        dict
            Inviable introgressions; keys are tuples containing site numbers.
        """
        introgressions = RNASeq.introgress(recipient, donor, n_introgr)
        inviable = {}
        for i in introgressions:
            seq = RNASeq(i)
            if not self.is_viable(seq):
                sites = RNASeq.get_diverged_sites(recipient, i)[2]
                key = tuple(sites)
                inviable[key] = ([donor[site] for site in sites], self.ref_seq.get_bp_distance(seq))
        return inviable, len(introgressions)

    def introgression_assay(self, seqA, seqB):
        """Find single, double, and triple introgressions in which result in an inviable genotype
        
        Parameters
        ----------
        dir : int, optional
            Direction of introgression. '1' tests 2 -> 1 DMIs whereas '2' tests 1 -> 2 DMIs.
        
        Returns
        -------
        dict
            A dictionary of single, double, and triple introgressions in one direction which result in inviable genotypes.
        
        Raises
        ------
        ValueError
            dir can only be '1' or '2'.
        """
        sin, n_sin = self.get_inviable_introgressions(seqA, seqB, 1)
        dou, n_dou = self.get_inviable_introgressions(seqA, seqB, 2)
        tri, n_tri = self.get_inviable_introgressions(seqA, seqB, 3)
        # remove inviable double introgressions that are caused by inviable
        # Single introgressions 
        sin_pairs = list(combinations(sin, 2))
        sin_pairs_seqs = [RNASeq.mutate(seqA, [i[0][0], i[1][0]], [sin[i[0]][0][0], sin[i[1]][0][0]]) for i in sin_pairs]
        sin_pairs_seqs_w = [self.is_viable(RNASeq(i)) for i in sin_pairs_seqs]
        # single introgressions
        trimmed_dou = {}
        for i in dou:
            found = False
            for j in sin:
                if j[0] in i:
                    found = True
                    break
            if not found:
                trimmed_dou[i] = dou[i]
        trimmed_dou
        # remove inviable triple introgressions that are caused by inviable
        # single or double introgressions
        trimmed_tri = {}
        for i in tri:
            found = False
            for j in sin:
                if j[0] in i:
                    found = True
                    break
            if not found:
                for j in trimmed_dou:
                    if sum([k in j for k in i]) == 2:
                        found = True
                        break
            if not found:
                trimmed_tri[i] = tri[i]

        if n_sin > 0:
            p1 = len(sin) / float(n_sin)
        else:
            p1 = 0
        if n_dou > 0:
            p2 = len(dou) / float(n_dou)
        else:
            p2 = 0
        if n_tri > 0:
            p3 = len(tri) / float(n_tri)
        else:
            p3 = 0
        return {
            'single': sin,
            'double': trimmed_dou,
            'triple': trimmed_tri,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'double_untrimmed_n' : n_dou,
            'triple_untrimmed_n' : n_tri,
            'viable_sin_pair': np.sum(sin_pairs_seqs_w),
        }

    def get_inviable_neighbors(self, seq):
        """Get inviable sequences that are a single mutation away from a
        sequence.

        Parameters
        ----------
        seq : str
            Sequence

        Returns
        -------
        int
            Number of inviable neighbors.
        """
        nei = RNASeq.get_neighbors(seq)
        inviable = []
        for site, nucl in nei:
            mut = RNASeq.mutate(seq, site, nucl)
            if not self.is_viable(RNASeq(mut)):
                inviable.append((site, nucl))
        return inviable

class TwoPops(object):

    def __init__(self, pop):
        """Initialize TwoPops object from Population.
        
        Parameters
        ----------
        pop : Population
            An instance of Population object.
        """
        self.init_pop = deepcopy(pop)
        self.pop1 = deepcopy(self.init_pop)
        self.pop2 = deepcopy(self.init_pop)

    def init_history(self):
        """Initialize attributes that will be used to keep a record of the
        evolutionary history during evolution.
        """
        self.divergence = []
        self.lin1 = []
        self.lin2 = []
        self.pop_lin1 = [] 
        self.pop_lin2 = []
        self.lin1_bp = []
        self.lin2_bp = []
        self.avg_mfe = []
        self.holeyness = []
        self.pop1_samples = []
        self.pop2_samples = []
        self.pop1_sample_1 = []
        self.pop1_sample_2 = []
        self.pop2_sample_1 = []
        self.pop2_sample_2 = []
        self.diverged_per_site = []
        self.dmi_per_site = []
        self.introg_per_sub = []
        self.introg_results = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.single = []
        self.double = []
        self.triple = []
        self.single_loc = []
        self.double_loc = []
        self.triple_loc = []
        self.double_untrimmed = []
        self.triple_untrimmed = []
        self.single_seg = []
        self.single_seg_norm = []
        self.double_seg = []
        self.triple_seg = []
        self.substitutions = []
        self.RI_max = []
        self.RI = []
        self.GST = []
        self.HT = []
        self.HS = []
        self.D = []
        self.rec_load = []
        self.rec_load_max = []
        self.pop_size = []

    def update_history(self, extended=True, single_lin=False, sample_n=10):
        """Update history during simulation.
        
        Parameters
        ----------
        sample_n : int, optional
            The number of samples used to do assays, by default 10
        """
        if single_lin:
            self.lin1.append(self.pop1.wt_seq)
            self.lin1_bp.append(RNASeq(self.pop1.wt_seq).get_bp_distance(self.pop1.ref_seq))
            pop1_seqs = [RNASeq.convertor(i, inv=True) for i in self.pop1.population]
            pop1_sample = list(np.random.choice(pop1_seqs, size=sample_n))
            self.pop_lin1.append(pop1_seqs)
            self.pop1_samples.append(pop1_sample)
            # Calculate holeyness
            holeyness = []   
            for i in pop1_sample:
                holeyness.append(len(self.pop1.get_inviable_neighbors(i)))
            self.holeyness.append(np.mean(holeyness))
            # population genetic measures
            self.HS.append(self.pop1.gene_diversity.mean())
            # Calculate mean recombination load
            self.rec_load.append(self.pop1.rec_load_realized)
            self.rec_load_max.append(self.pop1.rec_load_max)
            # calculate the segregating IIs
            pop1_sample1 = list(np.random.choice(pop1_seqs, size=sample_n))
            pop1_sample2 = list(np.random.choice(pop1_seqs, size=sample_n))
            single_seg = []
            single_seg_norm = []
            double_seg = []
            triple_seg = []
            d_sites = []
            for i,j in zip(pop1_sample1, pop1_sample2):
                temp = self.pop1.introgression_assay(i,j)
                single_seg.append(len(temp['single']))
                double_seg.append(len(temp['double']))
                triple_seg.append(len(temp['triple']))
                diff = RNASeq.get_hamdist(i, j)
                d_sites.append(diff)
            # Mean Pop 1 segregating IIs
            single_seg = np.mean(single_seg)
            if np.mean(d_sites) != 0:
                single_seg_norm = single_seg/np.mean(d_sites)
            else:
                single_seg_norm = 0.
            double_seg = np.mean(double_seg)
            triple_seg = np.mean(triple_seg)
        else:
            self.lin1.append(self.pop1.wt_seq)
            self.lin2.append(self.pop2.wt_seq)
            self.lin1_bp.append(RNASeq(self.pop1.wt_seq).get_bp_distance(self.pop1.ref_seq))
            self.lin2_bp.append(RNASeq(self.pop2.wt_seq).get_bp_distance(self.pop1.ref_seq))
            pop1_seqs = [RNASeq.convertor(i, inv=True) for i in self.pop1.population]
            pop2_seqs = [RNASeq.convertor(i, inv=True) for i in self.pop2.population]
            if self.pop1.N > 1:
                pop1_sample = list(np.random.choice(pop1_seqs, size=sample_n))
                pop2_sample = list(np.random.choice(pop2_seqs, size=sample_n))
            else:
                pop1_sample = pop1_seqs
                pop2_sample = pop2_seqs
            self.pop_lin1.append(pop1_seqs)
            self.pop_lin2.append(pop2_seqs)
            self.pop1_samples.append(pop1_sample)
            self.pop2_samples.append(pop2_sample)
            #Calculate divergence
            divergence = []   
            for i,j in zip(pop1_sample, pop2_sample):
                divergence.append(RNASeq.get_hamdist(i,j))
            self.divergence.append(np.mean(divergence))
            # Calculate holeyness
            holeyness = []   
            for i,j in zip(pop1_sample, pop2_sample):
                holeyness.append(len(self.pop1.get_inviable_neighbors(i)))
                holeyness.append(len(self.pop2.get_inviable_neighbors(j)))
            self.holeyness.append(np.mean(holeyness))
            # RI
            temp = self.get_RI()
            self.RI_max.append(temp['RI_max'])
            self.RI.append(temp['RI'])
            # population genetic measures
            temp = self.get_genetic_variation()
            self.GST.append(temp['GST'])
            self.HT.append(temp['HT'])
            self.HS.append(temp['HS'])
            self.D.append(temp['D'])
            # Calculate mean recombination load
            if self.pop1.N >1:
                pop1_realized_load = self.pop1.rec_load_realized
                pop2_realized_load = self.pop2.rec_load_realized
                self.rec_load.append(np.mean([pop1_realized_load, pop2_realized_load]))
                pop1_max_load = self.pop1.rec_load_max
                pop2_max_load = self.pop2.rec_load_max
                self.rec_load_max.append(np.mean([pop1_max_load, pop2_max_load]))
            # Check pop size
            self.pop_size.append(np.mean([self.pop1.N, self.pop2.N]))

        if extended:
            # Introgression 2 -> 1
            introg_results = []
            p1 = []
            p2 = []
            p3 = []
            single = []
            double = []
            triple = []
            double_untrimmed = []
            triple_untrimmed = []
            single_loc = []
            double_loc = []
            triple_loc = []
            diverged_per_site = []
            dmi_per_site = []
            introg_per_sub = []
            for i,j in zip(pop1_sample, pop2_sample):
                temp = self.pop1.introgression_assay(i,j)
                introg_results.append(temp)
                p1.append(temp['p1'])
                p2.append(temp['p2'])
                p3.append(temp['p3'])
                single.append(len(temp['single']))
                double.append(len(temp['double']))
                triple.append(len(temp['triple']))
                double_untrimmed.append(temp['double_untrimmed_n'])
                triple_untrimmed.append(temp['triple_untrimmed_n'])
                single_loc += [k[0] for k in temp['single'].keys()]
                double_loc += [k[0] for k in temp['double'].keys()] + [k[1] for k in temp['double'].keys()]
                triple_loc += [k[0] for k in temp['triple'].keys()] + [k[1] for k in temp['triple'].keys()] + [k[2] for k in temp['triple'].keys()]
                diff = RNASeq.get_diverged_sites(i, j)
                subs = np.zeros(100)
                loci = [k[0] for k in temp['single'].keys()]
                for k in diff[2]:
                    subs[k] = 1
                dmi = np.zeros(100)
                for l in loci:
                    dmi[l] = 1
                diverged_per_site.append(subs)
                dmi_per_site.append(dmi)
                introg_per_sub.append(np.divide(dmi, subs, out=np.zeros_like(dmi), where=subs!=0))
            self.diverged_per_site.append(diverged_per_site)
            self.dmi_per_site.append(dmi_per_site)
            self.introg_per_sub.append(np.mean(introg_per_sub, axis=0))
            self.single_loc.append(single_loc)
            self.double_loc.append(double_loc)
            self.triple_loc.append(triple_loc)
            self.introg_results.append(introg_results)
            self.p1.append(np.mean(np.array(p1)))
            self.p2.append(np.mean(np.array(p2)))
            self.p3.append(np.mean(np.array(p3)))
            self.single.append(np.mean(np.array(single)))
            self.double.append(np.mean(np.array(double)))
            self.triple.append(np.mean(np.array(triple)))
            self.double_untrimmed.append(np.mean(np.array(double_untrimmed)))
            self.triple_untrimmed.append(np.mean(np.array(triple_untrimmed)))
            # Find segregating IIs within populations
            pop1_sample1 = list(np.random.choice(pop1_seqs, size=sample_n))
            pop1_sample2 = list(np.random.choice(pop1_seqs, size=sample_n))
            pop2_sample1 = list(np.random.choice(pop2_seqs, size=sample_n))
            pop2_sample2 = list(np.random.choice(pop2_seqs, size=sample_n))
            self.pop1_sample_1.append(pop1_sample1)
            self.pop1_sample_2.append(pop1_sample2)
            self.pop2_sample_1.append(pop2_sample1)
            self.pop2_sample_2.append(pop2_sample2)
            # Pop 1 segregating IIs
            single_seg_1 = []
            single_seg_1_norm = []
            double_seg_1 = []
            triple_seg_1 = []
            d_sites = []
            for i,j in zip(pop1_sample1, pop1_sample2):
                temp = self.pop1.introgression_assay(i,j)
                single_seg_1.append(len(temp['single']))
                double_seg_1.append(len(temp['double']))
                triple_seg_1.append(len(temp['triple']))
                diff = RNASeq.get_hamdist(i, j)
                d_sites.append(diff)
            # Mean Pop 1 segregating IIs
            single_seg_1 = np.mean(single_seg_1)
            if np.mean(d_sites) != 0:
                single_seg_1_norm = single_seg_1/np.mean(d_sites)
            else:
                single_seg_1_norm = 0.
            double_seg_1 = np.mean(double_seg_1)
            triple_seg_1 = np.mean(triple_seg_1)
            # Pop 2 segregating IIs
            single_seg_2 = []
            single_seg_2_norm = []
            double_seg_2 = []
            triple_seg_2 = []
            d_sites = []
            for i,j in zip(pop2_sample1, pop2_sample2):
                temp = self.pop1.introgression_assay(i,j)
                single_seg_2.append(len(temp['single']))
                double_seg_2.append(len(temp['double']))
                triple_seg_2.append(len(temp['triple']))
                diff = RNASeq.get_hamdist(i, j)
                d_sites.append(diff)
            # Mean Pop 2 segregating IIs
            single_seg_2 = np.mean(single_seg_2)
            if np.mean(d_sites) != 0:
                single_seg_2_norm = single_seg_2/np.mean(d_sites)
            else:
                single_seg_2_norm = 0.
            double_seg_2 = np.mean(double_seg_2)
            triple_seg_2 = np.mean(triple_seg_2)
            # Averaging segregating IIs across the two diverged sites
            self.single_seg.append(np.mean([single_seg_1, single_seg_2]))
            self.single_seg_norm.append(np.mean([single_seg_1_norm, single_seg_2_norm]))
            self.double_seg.append(np.mean([double_seg_1, double_seg_2]))
            self.triple_seg.append(np.mean([triple_seg_1, triple_seg_2]))

    @property
    def stats(self):
        """Generate a dictionary of evolutionary history attributes.

        Used by save_stats().

        Returns
        -------
        dict
            Evolutionary history.
        """
        stats = {}
        stats['ancestor'] = self.pop1.ancestor.seq
        stats['ref_seq'] = self.pop1.ref_seq.seq
        stats['divergence'] = self.divergence
        stats['seqs_lin1'] = self.lin1
        stats['seqs_lin2'] = self.lin2
        stats['pop_lin1'] = self.pop_lin1
        stats['pop_lin2'] = self.pop_lin2
        stats['lin1_bp'] = self.lin1_bp
        stats['lin2_bp'] = self.lin2_bp
        stats['mfe_avg'] = self.avg_mfe
        stats['holeyness'] = self.holeyness
        stats['pop1_samples'] = self.pop1_samples
        stats['pop2_samples'] = self.pop2_samples
        stats['pop1_sample_1'] = self.pop1_sample_1
        stats['pop1_sample_2'] = self.pop1_sample_2
        stats['pop2_sample_1'] = self.pop2_sample_1
        stats['pop2_sample_2'] = self.pop2_sample_2
        stats['d_per_site'] = self.diverged_per_site
        stats['dmi_per_site'] = self.dmi_per_site
        stats['dmi_per_d'] = self.introg_per_sub
        stats['introg_results'] = self.introg_results
        stats['p1'] = self.p1
        stats['p2'] = self.p2
        stats['p3'] = self.p3
        stats['single'] = self.single
        stats['double'] = self.double
        stats['triple'] = self.triple
        stats['double_untrimmed'] = self.double_untrimmed
        stats['triple_untrimmed'] = self.triple_untrimmed
        stats['single_seg'] = self.single_seg
        stats['single_seg_norm'] = self.single_seg_norm 
        stats['double_seg'] = self.double_seg
        stats['triple_seg'] = self.triple_seg
        stats['substitutions'] = self.substitutions
        stats['RI'] = self.RI
        stats['RI_max'] = self.RI_max
        stats['GST'] = self.GST
        stats['HT'] = self.HT
        stats['HS'] = self.HS
        stats['D'] = self.D
        stats['rec_load'] = self.rec_load
        stats['rec_load_max'] = self.rec_load_max
        stats['N'] = self.pop_size
        stats['single_loci'] = self.single_loc
        stats['double_loci'] = self.double_loc
        stats['triple_loci'] = self.triple_loc
        temp = self.get_panmictic_fitness()
        stats['pooled_w'] = temp['w_pooled']
        stats['pooled_w_max'] = temp['w_pooled_max']
        return stats

    def save_stats(self, directory):
        """Save evolutionary history in pickle format.
        
        Parameters
        ----------
        directory : int
            A number added to the file name to avoid overwriting.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = open(directory + "/stats_" + str(uuid.uuid4()), 'w')
        pickle.dump(self.stats, file)
        file.close()

    def get_genetic_variation(self):
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
        H_within.append(self.pop1.gene_diversity.mean())
        H_within.append(self.pop2.gene_diversity.mean())
        pooled = np.concatenate((self.pop1.population, self.pop2.population))
        pooled_pop = deepcopy(self.pop1)
        pooled_pop.change_pop(pooled)
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

    def get_RI(self):
        """Calculate reproductive isolation between the two populations
        
        Returns
        -------
        Dictionary
            RI_max: The max level of reproductive isolation based on the inviability 
            of all the single cross-over recombinants.
            RI: The level of reproductive isolation based on the inviability of the 
            single cross-over recombinants according to recombination rate (r).
        """
        RI_r = 0
        if self.pop1.N == 1: #SSWM
            recs = RNASeq.creat_multi_recs(self.pop1.wt_seq, self.pop2.wt_seq)
            recs_w = [self.pop1.is_viable(RNASeq(i)) for i in recs]
            RI_max = 1. -  np.sum(recs_w)/float(len(recs_w))
        else:
            recs = self.pop1.recombine_multi(other=self.pop2)
            recs_w = [self.pop1.is_viable(RNASeq(RNASeq.convertor(i, inv=True))) for i in recs]
            RI_max = 1. - np.sum(recs_w)/float(len(recs_w))
            if self.pop1.r:
                recs = self.pop1.recombine_pop(other=self.pop2)
                recs = [RNASeq.convertor(i, inv=True) for i in recs]
                recs_w = [self.pop1.is_viable(RNASeq(i)) for i in recs]
                WS = np.sum(recs_w)/float(len(recs_w))
                RI_r = 1. - WS
        return {
            'RI_max': RI_max, 
            'RI': RI_r
            }

    def get_panmictic_fitness(self):
        """Calculate the fitness of a panmictic population after divergence.
        
        Returns
        -------
        Dictionary
            Contains the max and the realized fitness of the pooled population.
        """
        pooled = np.concatenate((self.pop1.population, self.pop2.population))
        pooled_pop = deepcopy(self.pop1)
        pooled_pop.change_pop(pooled)
        pooled_pop_free_rec = deepcopy(pooled_pop)
        pooled_pop_free_rec.free_rec = True
        assert pooled_pop.ref_seq.seq == pooled_pop_free_rec.ref_seq.seq == self.pop1.ref_seq.seq
        pooled_w = pooled_pop.mixis()
        pooled_free_w = pooled_pop_free_rec.mixis()
        pop1 = self.pop1.population
        pop2 = self.pop2.population
        return {
            'w_pooled' : pooled_w,
            'w_pooled_max' : pooled_free_w
            }

    def evolve_from_eq(self, gen, step=100, eq_t=1000, verbose=False):
        """Evolve two divergent populations after equilibrium.

        Parameters
        ----------
        gen : int
            The number generations after divergence.
        step : int, optional
            The intervals at which the relevant statistics are to be saved, by default 100
        eq_t : int, optional
            The number of generations during which the ancestral population evolves, by default 1000
        verbose : bool, optional
            Print the save points, by default False, by default False
        """
        self.init_history()
        self.update_history(extended=False, single_lin=True)
        for i in np.arange(1, eq_t + 1, 1):
            self.pop1.get_next_generation()
            if not i%step: 
                self.update_history(extended=False, single_lin=True)
        if verbose:
            print('Reached eq.')
        self.pop2 = deepcopy(self.pop1)
        self.update_history(extended=True)
        for i in np.arange(1, gen + 1, 1):
            self.pop1.get_next_generation()
            self.pop2.get_next_generation()
            if not i%step:
                self.update_history(extended=True)
                if verbose:
                    print i, 

    def evolve(self, gen, step=100, ext_hist=True,verbose=False):
        """Evolve two divergent populations.
        
        Parameters
        ----------
        gen : int
            The number of generation.
        step : int, optional
            The intervals at which the relevant statistics are to be saved, by default 100
        verbose : bool, optional
            Print the save points, by default False
        """
        self.init_history()
        self.update_history(extended=ext_hist)
        for i in np.arange(1, gen + 1, 1):
            self.pop1.get_next_generation()
            self.pop2.get_next_generation()
            if not i%step:
                self.update_history(extended=ext_hist)
                if verbose:
                    print i,

if __name__ == "__main__":
    import doctest
    doctest.testmod()