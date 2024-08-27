"""Ibm_model.py contains three classes: IbmModel, Model1 and Model2
"""
import os
from abc import ABC, abstractmethod
from alignment import Alignment


class IbmModel(ABC):
    """
    Abstract base class providing a framework for implementing IBM Model aligners which are 
    used for aligning words in bilingual sentence pairs during machine translation.

    Attributes:
        sentence_pairs (list[tuple[list[str], list[str]]]): Bilingual sentence pairs.
        n_iter (int): Number of iterations for training the model.
        f_vocab (set[str]): Vocabulary of words in the foreign language.
        e_vocab (set[str]): Vocabulary of words in the English language.
        translation_probs (dict[str, dict[str, float]]): Dictionary storing translation probabilities.
    """
    NAME = "Abstract IBM Model"
    
    def __init__(self, sentence_pairs: list[tuple[list[str], list[str]]], n_iter=20, use_null=True):
        self.sentence_pairs = sentence_pairs
        self.n_iter = n_iter
        self.use_null = use_null
        self.f_vocab = set(word for pair in sentence_pairs for word in pair[0])
        if self.use_null:
            self.f_vocab.add(None)  # Add NULL token to the foreign vocabulary
        self.e_vocab = set(word for pair in sentence_pairs for word in pair[1])
        self.translation_probs = self._initialize_translation_probabilities()


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_alignment(self, continuous=False)->list[Alignment]:
        pass

    def _initialize_translation_probabilities(self):
        translation_probs = {}
        total_words = len(self.e_vocab) + (1 if self.use_null else 0)
        uniform_probability = 1.0 / total_words
        for f_word in self.f_vocab:
            translation_probs[f_word] = {e_word: uniform_probability for e_word in self.e_vocab}
        return translation_probs


    def _collect_counts(self):
        counts = {f_word: {e_word: 0 for e_word in self.e_vocab} for f_word in self.f_vocab}
        totals = {e_word: 0 for e_word in self.e_vocab}

        for f_sentence, e_sentence in self.sentence_pairs:
            if self.use_null:
                f_sentence = [None] + f_sentence  # Include NULL token in the f sentence

            for f_word in f_sentence:
                s_total = sum(self.translation_probs[f_word].get(e_word, 0) for e_word in e_sentence)
                if s_total > 0:
                    for e_word in e_sentence:
                        translation_prob = self.translation_probs[f_word].get(e_word, 0)
                        count_increment = translation_prob / s_total
                        counts[f_word][e_word] += count_increment
                        totals[e_word] += count_increment
        return counts, totals


    def _estimate_translation_probabilities(self, counts, totals):
        """ Estimates new translation probabilities based on collected counts """
        for f_word in counts:
            for e_word, count_fe in counts[f_word].items():
                total_e = totals[e_word]
                if total_e != 0:
                    self.translation_probs[f_word][e_word] = count_fe / total_e

    def find_highest_prob(self, f_word):
        """ Finds the English word with the highest probability for a given foreign word """
        e_word, max_prob = max(self.translation_probs[f_word].items(), key=lambda item: item[1])
        return e_word, max_prob

class Model1(IbmModel):
    """
    Implements the IBM Model 1 training algorithm.
    """
    NAME = "Model1"

    def __init__(self, sentence_pairs, n_iter=20, use_null=False):
        super().__init__(sentence_pairs, n_iter, use_null)

    def train(self, convergence_threshold=0.001):
        converged = False
        n = 0
        prev_translation_probs = None

        while not converged and n < self.n_iter:
            counts, totals = self._collect_counts()
            self._estimate_translation_probabilities(counts, totals)

            # Convergence check (simplified)
            if prev_translation_probs:
                changes = [abs(self.translation_probs[f_word][e_word] - prev_translation_probs[f_word][e_word])
                           for f_word in self.f_vocab for e_word in self.e_vocab if e_word in prev_translation_probs[f_word]]
                max_change = max(changes, default=0)
                if max_change < convergence_threshold:
                    converged = True

            prev_translation_probs = {f_word: {e_word: prob for e_word, prob in e_words.items()}
                                      for f_word, e_words in self.translation_probs.items()}
            n += 1

        print(f"Training completed after {n} iterations with convergence status: {converged}")

    def get_alignment(self, continuous=False) -> list[Alignment]:
        total_f_index = 0  # Initialize total index count for foreign sentences
        total_e_index = 0  # Initialize total index count for English sentences
        alignments = []
        print("ALIGNING using Model1 aligner...")

        for f_sentence, e_sentence in self.sentence_pairs:
            l = len(f_sentence)
            m = len(e_sentence)
            alignment_sequence = []

            for i, f_word in enumerate(f_sentence):
                max_prob = 0.0
                aligned_index = -1
                for j, e_word in enumerate(e_sentence):
                    transl_prob = self.translation_probs[f_word].get(e_word, 0.0)
                    if transl_prob > max_prob:
                        max_prob = transl_prob
                        aligned_index = j

                if continuous:
                    alignment_sequence.append((total_f_index + i, total_e_index + aligned_index))
                else:
                    alignment_sequence.append((i, aligned_index))

            if continuous:
                total_f_index += l
                total_e_index += m

            alignments.append(Alignment(alignment_sequence))
        return alignments


class Model2(IbmModel):
    """
    A concrete implementation of the IbmModel, representing IBM Model 2.

    This class implements the specific training algorithm and word alignment method as
    described in IBM Model 2, which extends Model 1 by incorporating alignment
    probabilities.

    Attributes:
        NAME (str): Name of the model.
        alignment_probs (dict): Dictionary to store alignment probabilities.

    Methods:
        train:
        initialize_alignment_counts:
        initialize_alignment_probabilities:
        estimate_alignment_probabilities:
        get_alignment: 
    """

    NAME = "Model2"


    def __init__(self, sentence_pairs, n_iter=20, use_null=True):
        super().__init__(sentence_pairs, n_iter, use_null)
        self.alignment_probs = self._initialize_alignment_probabilities()

    def _initialize_alignment_probabilities(self):
        alignment_probs = {}
        for e_sentence, f_sentence in self.sentence_pairs:
            l = len(e_sentence)
            m = len(f_sentence) + (1 if self.use_null else 0)
            for i in range(l):
                for j in range(m):
                    alignment_probs[(i, j, l, m)] = 1 / (m if m else 1)  # Ensure non-zero division
        return alignment_probs


    def _estimate_alignment_probabilities(self):
        alignment_counts, alignment_totals = self._initialize_alignment_counts()
        for e_sentence, f_sentence in self.sentence_pairs:
            l = len(e_sentence)
            m = len(f_sentence) + (1 if self.use_null else 0)
            for j, f_word in enumerate([None] + f_sentence if self.use_null else f_sentence):
                total_s = 0
                for i, e_word in enumerate(e_sentence):
                    # Safe access to probabilities with default value of 0.0 if keys don't exist
                    transl_prob = self.translation_probs.get(f_word, {}).get(e_word, 0.0)
                    align_prob = self.alignment_probs.get((i, j, l, m), 0.0)
                    total_s += transl_prob * align_prob

                if total_s > 0:  # To avoid division by zero
                    for i, e_word in enumerate(e_sentence):
                        transl_prob = self.translation_probs.get(f_word, {}).get(e_word, 0.0)
                        align_prob = self.alignment_probs.get((i, j, l, m), 0.0)
                        count = transl_prob * align_prob / total_s
                        alignment_counts[(i, j, l, m)] += count
                        alignment_totals[(j, l, m)] += count

        # Update the alignment probabilities based on the counts
        # for key in alignment_counts:
        #     i, j, l, m = key
        #     self.alignment_probs[key] = alignment_counts[key] / alignment_totals[(j, l, m)] if alignment_totals[(j, l, m)] else 0

        for key, count in alignment_counts.items():
            i, j, l, m = key
            total = alignment_totals[(j, l, m)]
            if total > 0:  # Prevent division by zero
                self.alignment_probs[key] = count / total
            else:
                self.alignment_probs[key] = 0


    def train(self):
        for n in range(self.n_iter):
            counts, totals = self._collect_counts()
            self._estimate_translation_probabilities(counts, totals)
            if isinstance(self, Model2):
                self._estimate_alignment_probabilities()
            print(f"\nTraining {self.NAME}. Iteration No: {n}\n")

    def _initialize_alignment_counts(self) -> tuple[dict[tuple[int, int, int, int], float], dict[tuple[int, int, int] : float]]:
        """Initializes alignment counts."""
        alignment_counts = {}
        alignment_totals = {}
        for e_sentence, f_sentence in self.sentence_pairs:
            l = len(e_sentence)
            m = len(f_sentence)
            for i in range(l):
                for j in range(m):
                    alignment_counts[(i, j, l, m)] = 0 #counta(i|j,le,lf) = 0 for all i,j,le,lf
                    alignment_totals[(j, l, m)] = 0
        return alignment_counts, alignment_totals

    def get_alignment(self, continuous=False):
        total_f_index = 0
        total_e_index = 0
        alignments = []
        for e_sentence, f_sentence in self.sentence_pairs:
            l = len(e_sentence)
            m = len(f_sentence)
            alignment_sequence = []

            for i, e_word in enumerate(e_sentence):
                max_prob = 0.0
                aligned_index = -1  # Use -1 to indicate no alignment if no positive probability found
                for j, f_word in enumerate(f_sentence):
                    align_prob = self.alignment_probs.get((i, j, l, m), 0.0)
                    if align_prob > max_prob:
                        max_prob = align_prob
                        aligned_index = j

                if continuous:
                    alignment_sequence.append((total_f_index + i, total_e_index + aligned_index))
                else:
                    alignment_sequence.append((i, aligned_index))

            if continuous:
                total_f_index += l
                total_e_index += m
            alignments.append(Alignment(alignment_sequence))

        return alignments

@staticmethod
def write_probs_to_file(translation_probs: dict[str, dict[str, float]]) -> None:
    """Writes translation probabilities to file mainly for testing and verification purposes.
    Parameters:
        translation_probs (dict of dict of str to float): The translation probabilities to write to the file.

    Raises:
        IOError: If there is an error in writing to the file.
    """
    print("WRITING PROBS_DICT TO FILE...")
    try:
        if not os.path.exists("alignments"):
            os.makedirs("alignments")
        with open("alignments/00probs.txt", "w", encoding="UTF-8") as file_1,\
                open("alignments/01probs.txt", "w", encoding="UTF-8") as file_2:##!!!!FILE1
            for e_word, e_probs in translation_probs.items():
                max_f_word, max_prob = None, float('-inf')
                for word, prob in e_probs.items():
                    if prob > max_prob:
                        max_f_word, max_prob = word, prob
                # Now using max_f_word in the output
                file_2.write(f"{e_word} - {max_f_word}: {max_prob}\n")
    except (IOError, OSError) as e:
        print(f"Error writing probabilities to file: {e}")


def get_alignment(self, continuous=False):
        alignments = []
        for e_sentence, f_sentence in self.sentence_pairs:
            l = len(e_sentence)
            m = len(f_sentence)
            alignment_sequence = []
            for i, e_word in enumerate(e_sentence):
                max_prob = 0.0
                aligned_index = 0
                for j, f_word in enumerate(f_sentence):
                    transl_prob = self.translation_probs[e_word].get(f_word, 0.0)
                    align_prob = self.alignment_probs.get((i, j, l, m), 0.0)
                    combined_prob = transl_prob * align_prob
                    if combined_prob > max_prob:
                        max_prob = combined_prob
                        aligned_index = j
                alignment_sequence.append((i, aligned_index))
            alignments.append(Alignment(alignment_sequence))
        return alignments