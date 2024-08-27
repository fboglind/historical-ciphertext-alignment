"""cipher_alignment_model.py contains the CipherAlignmentModel, an abstract adapter facilitating using IBM Model
implementations for the task of aligning plaintext and ciphertext and recreating cipherkeys by calculating
translation probabilities.
This version is an OPTIMIZED version which disallows the use of NULL alignments in the alignment process
"""
from abc import ABC, abstractmethod
from nltk.translate import AlignedSent, IBMModel1, IBMModel2, IBMModel3
from ibm_model import IbmModel, Model1, Model2
from alignment import Alignment

class CipherAlignmentModel(ABC):
    """Initialize with the specified model name and training corpus.

    :param bitext: A list of tuples, each containing two lists of strings (source, target)
    :param iterations: Number of iterations for model training
    """
    NAME="CipherAlignmentModel"

    def __init__(self, bitext, n_iter=20):
        self.bitext = bitext
        self.n_iter = n_iter
        self.model = None
        self.translation_probs = {}
        self.alignments = []
        

    @abstractmethod
    def train(self):
        """Train the model on the given bitext."""
        pass

    @abstractmethod
    def get_translation_probs(self):
        """Retrieve the translation probabilities in a structured dictionary."""
        pass

    @abstractmethod
    def align_sentences(self):
        """Align sentences and return the alignments."""
        pass

class NLTKIBMModelAdapter(CipherAlignmentModel):
    def __init__(self, bitext, n_iter=20, model_type='model1', use_null=True):
        super().__init__(bitext, n_iter)
        self.bitext = [AlignedSent(src, trg) for src, trg in bitext]
        self.NAME = f"NLTK_{model_type}"
        self.use_null = use_null  # Store the use_null parameter
        self.model = self.init_model(model_type)
        self.translation_probs = self.get_translation_probs()
        self.alignments = self.align_sentences()

    def init_model(self, model_type):
        """Initialize the IBM Model based on the specified type and adjust NAME based on use_null."""
        if model_type == 'model1':
            model = IBMModel1(self.bitext, self.n_iter)
        elif model_type == 'model2':
            model = IBMModel2(self.bitext, self.n_iter)
        elif model_type == 'model3':
            model = IBMModel3(self.bitext, self.n_iter)
        else:
            raise ValueError(f"Unsupported model type {model_type}. Only 'model1', 'model2', and 'model3' are supported.")

        # Adjust the NAME attribute based on use_null
        if not self.use_null:
            self.NAME = f"NLTK_{model_type}_no_null"
        else:
            self.NAME = f"NLTK_{model_type}"

        return model


    def train(self):
        """Optionally re-train the model if needed."""
        if self.model is None:
            self.model = self.init_model(self.model_type)
            self.translation_probs = self.get_translation_probs()
            self.alignments = self.align_sentences()

    def get_translation_probs(self):
        """Converts the model's translation table to a dictionary."""
        source_to_target_prob = {}
        for src_word, trg_dict in self.model.translation_table.items():
            for trg_word, prob in trg_dict.items():
                if src_word not in source_to_target_prob:
                    source_to_target_prob[src_word] = {}
                source_to_target_prob[src_word][trg_word] = prob
        return source_to_target_prob

    def align_sentences(self, continuous=False):
        """Align sentences and return the alignments."""
        alignment_objects = []
        total_source_index = 0
        total_target_index = 0

        for sentence_pair in self.bitext:
            adjusted_alignment = []
            if continuous:
                for src_index, trg_index in sentence_pair.alignment:
                    if not self.use_null and trg_index is None:
                        continue  # Skip NULL alignments if not allowed
                    adjusted_src_index = total_source_index + src_index
                    adjusted_trg_index = total_target_index + trg_index if trg_index is not None else -1
                    adjusted_alignment.append((adjusted_src_index, adjusted_trg_index))
                total_source_index += len(sentence_pair.words)
                total_target_index += len(sentence_pair.mots)
            else:
                for src, trg in sentence_pair.alignment:
                    if not self.use_null and trg is None:
                        continue  # Skip NULL alignments if not allowed
                    adjusted_src = src
                    adjusted_trg = trg if trg is not None else -1
                    adjusted_alignment.append((adjusted_src, adjusted_trg))

            alignment_objects.append(Alignment(adjusted_alignment, self.NAME))

        return alignment_objects


 
class GenericIBMModelAdapter(CipherAlignmentModel):
    """Adapter class to use a Generic IBM Model implementation from the ibm_model.py file."""
    def __init__(self, bitext, n_iter=20, model_type='model1', use_null=True):
        super().__init__(bitext, n_iter)
        self.bitext = bitext
        self.n_iter = n_iter
        self.model_type = model_type
        self.use_null = use_null
        
        # Determine which model to instantiate and set the appropriate name
        if model_type == 'model1':
            self.model = Model1(self.bitext, self.n_iter)
            if use_null:
                self.NAME = "generic_model1"
            elif not use_null:
                self.NAME = "generic_model1_no_null"  # Update the NAME attribute based on the model
        elif model_type == 'model2':
            self.model = Model2(self.bitext, self.n_iter)
            self.NAME = "model2"  # Update the NAME attribute based on the model
        else:
            raise ValueError("Invalid model type specified. Choose 'model1' or 'model2'.")

        self.model.train()  # Train the model immediately if desired
        self.translation_probs = self.get_translation_probs()
        self.alignments = self.align_sentences()

    def train(self):
        """Train the model using the implemented logic in the Generic IBM Model."""
        self.model.train()

    def get_translation_probs(self):
        """Retrieves the translation probabilities from the model."""
        return self.model.translation_probs

    def align_sentences(self, continuous=False):
        """Uses the model's method to get alignments."""
        return self.model.get_alignment(continuous)
