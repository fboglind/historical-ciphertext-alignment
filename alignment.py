"""alignment.py contains the Alignment class"""
from typing import Iterator
from nltk.translate import alignment_error_rate
from Levenshtein import distance as levenshtein_distance

class Alignment:
    """Represents the word alignment of a single sentence pair in a bilingual corpus.

        This class is used to handle the alignment data structure which consists of pairs of integers,
        where each pair represents a link between a word in the source sentence and a word in the target sentence.

        Attributes:
            alignment (list[tuple[int, int]]): A list of tuples, each containing two integers representing aligned word indices.
            model_name (str): The name of the model or method used to generate the alignment.
        Methods:
            __str__: Returns a string representation of the alignment.
            __len__: Returns the number of alignments.
            __iter__: Returns an iterator for the alignment list.
            __repr__: Returns a string representation of the alignment.
            calculate_aer: Calculates the Alignment Error Rate (AER) comparing to a reference alignment.
            calculate_levenshtein: Calculates the Levenshtein distance between the reference alignment and the model alignment.
            get_precision_recall_fscore: Calculates precision, recall, and F1 score comparing to a reference alignment.
        """
    def __init__(self, alignment: list[tuple[int, int]], model_name=None) -> None:
        self.alignment = alignment
        self.model_name = model_name

    # def __str__(self) -> str:
    #     return " ".join(f"{a}-{b}" for a, b in self.alignment)
    def __str__(self) -> str:
        return " ".join(f"{idx}: {a}-{b}" for idx, (a, b) in enumerate(self.alignment))

    def __len__(self) -> int:
        return len(self.alignment)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.alignment)

    def __repr__(self) -> str:
        return self.__str__()

    def calculate_aer(self, ref_alignment: list[tuple[int, int]]) -> float:
        """Calculate the Alignment Error Rate (AER) comparing to a reference alignment."""
        sure_set = set(ref_alignment)  # Assuming ref_alignment is the sure set
        possible_set = set(ref_alignment)  # Assuming it's the same in this context
        proposed_set = set(self.alignment)
        return alignment_error_rate(sure_set, proposed_set, possible_set)

    def calculate_levenshtein(self, ref_alignment: list[tuple[int, int]]) -> int:
        """Calculate the Levenshtein distance between the reference alignment and the model alignment."""
        ref_seq = ','.join(f'{src}-{tgt}' for src, tgt in ref_alignment)
        model_seq = ','.join(f'{src}-{tgt}' for src, tgt in self.alignment)
        return levenshtein_distance(ref_seq, model_seq)

    def get_precision_recall_fscore(self, ref_alignment: list[tuple[int, int]]) -> tuple[float, float, float]:
        """Calculate precision, recall, and F1 score comparing to a reference alignment."""
        true_positives = set(self.alignment) & set(ref_alignment)
        false_positives = set(self.alignment) - set(ref_alignment)
        false_negatives = set(ref_alignment) - set(self.alignment)

        precision = len(true_positives) / len(self.alignment) if self.alignment else 0
        recall = len(true_positives) / len(ref_alignment) if ref_alignment else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        return precision, recall, f1_score