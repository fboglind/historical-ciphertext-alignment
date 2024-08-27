"""cipher.py contains the Cipher class, the CipherKey class and the ErrorSequence class"""
import random
from error_sequence import ErrorSequence

class Cipher:
    """Class representing a cipher"""
    def __init__(self, id_tag: str, plaintext:str, str_key: str, ciphertext: str, filename: str, homophonicity: str, csv_filename=None, error: ErrorSequence=None, seq_size=None):
        self.id_tag = id_tag
        self.plaintext = plaintext
        self.original_reference_key = CipherKey(str_key=str_key)
        self.seq_size = len(plaintext) if seq_size is None else seq_size  # Set once, cleanly
        self.homophonicity = homophonicity
        self.error = error
        self.ciphertext = ciphertext if error is None else error.new_sequence
        self.reference_alignment_data = [(i, i) for i in range(len(plaintext))]
        self.filename = filename
        self.csv_filename = csv_filename
        self.bitext = self.make_bitext(self.seq_size)  # Use already set self.seq_size
  
    def __repr__(self):
        try:
            return f"Plaintext: {self.plaintext}\nCiphertext: {self.ciphertext}\nKey: {self.original_reference_key}\nFilename:\nError: {self.error}\ncsv-filename: {self.csv_filename}"
        except RecursionError:
            return f"Cipher object at {id(self)}"
    def __str__(self):
        return f"Plaintext: {self.plaintext}\nCiphertext: {self.ciphertext}\nKey: {self.original_reference_key}\nFilename: {self.filename}\nError: {self.error}\ncsv-filename: {self.csv_filename}"
    
    def make_bitext(self, seg_size: int) -> list[tuple[list[str],list[str]]]:
        """Creates a list of bisegments of n-length from plaintext and ciphertext"""
        cipher_list=self.ciphertext.split("_")
        plaintext_segments = [list(self.plaintext[i:i+seg_size]) for i in range(0, len(self.plaintext), seg_size)]
        ciphertext_segments = [cipher_list[i:i+seg_size] for i in range(0, len(cipher_list), seg_size)]
        return list(zip(ciphertext_segments, plaintext_segments))

class CipherKey:
    """Class representing a cipher key. A cipher key can be made from a Cipher-objects string-representation of a cipherkey,
    from translation probabilities or from a plaintext-ciphertext bitext."""
    def __init__(self, translation_probs: dict[str, dict[str, float]] = None, str_key: str = None, bitext: list[tuple[list[str],list[str]]] = None):
        self.translation_probs = translation_probs
        self.str_key = str_key
        self.bitext = bitext
        self.key = None
        self.type=None
        #self.homophonicity=self.check_homopohonicity() if self.key is not None else None

        if self.translation_probs is not None:
            self.make_key_from_probs(self.translation_probs)
        elif self.str_key is not None:
            self.make_key_from_string()
        elif self.bitext is not None:
            self.make_key_from_bitext(bitext)
    
    def __str__(self) -> str:
        return f"{self.key}"

    def make_key_from_bitext(self, bitext: list[tuple[list[str],list[str]]]) -> dict[str, set[str]]:
        """Creates a cipherkey from bitext, where the key is a plaintext element (a character) and the value is a list of possible ciphertext elements (numerical values).
        One plaintext element can map to multiple possible ciphertext elements."""
        key_dict = {}
        plaintext, ciphertext = bitext
        for i in range(len(plaintext)):
            if plaintext[i] not in key_dict:
                key_dict[plaintext[i]] = {ciphertext[i]}
            else:
                cipherkey[plaintext[i]].add(ciphertext[i])
        self.type="From bitext"
        self.key = dict(sorted(key_dict.items()))

    def make_key_from_probs(self, translation_probs: dict[str, dict[str, float]]) -> None:
        """
        Creates a cipherkey from translation probabilities, where the key is a plaintext element (a character) and the
        value is a list of possible ciphertext elements (numerical values). One plaintext element can map to multiple
        possible ciphertext elements, but one ciphertext element can only have one corresponding plaintext element.
        """
        key_dict = {}
        for target_word, source_probs in translation_probs.items():
            max_source_word, max_prob = None, float('-inf')
            for word, prob in source_probs.items():
                if prob > max_prob:
                    max_source_word, max_prob = word, prob
            # Ensure max_source_word is not None before adding to the dictionary
            if max_source_word is not None:
                if max_source_word not in key_dict:
                    key_dict[max_source_word] = {target_word}
                else:
                    key_dict[max_source_word].add(target_word)
        self.key = dict(sorted(key_dict.items()))
        self.type = "From translation probabilities"


    def make_key_from_string(self) -> None:
        """Creates a cipherkey from a string representation of a cipherkey which has been created synthetically
        and has a specific format: 'A:[64|41|66],B:[00],C:[38|65],D:[84|89|09]...'"""
        key_dict = {}
        key_list = self.str_key.split(",")
        for item in key_list:
            key, values = item.split(":")
            key_dict[key] = set(values[1:-1].split("|"))
        self.key = dict(sorted(key_dict.items()))  # Update this line
        self.type="Original. From string"


    def compare_keys(self, ref_key: "CipherKey") -> tuple[float, float, float]:
        """Compares the current cipher key with a reference key and returns the precision, recall, and F1 score.
        Note: The reference key should be an original key, a key that is recreated from a fully aligned bitext
        with no errors."""
        hypothesis_key = self.key
        reference_key = ref_key
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for k, v in hypothesis_key.items():  # Use self.cipherkey instead of hypothesis_key.key
            if k in reference_key.key:
                intersection = v.intersection(reference_key.key[k])
                true_positives += len(intersection)
                false_positives += len(v) - len(intersection)
                false_negatives += len(reference_key.key[k]) - len(intersection)
            else:
                false_positives += len(v)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return (precision, recall, f1_score)

    # def check_hompohonicity(self) -> int:
    #     """Checks the homophonicity of a cipherkey. If the cipherkey is homophonic,
    #     it returns the number of homophonic elements, otherwise it returns 0(?)."""
    #     homophonicity = 0
    #     evenness = True
    #     previous_value = 0
    #     for k, v in self.key.items():
    #         homophonicity = len(v)
    #         if evenness:
    #             if len(v) != previous_value:
    #                 evenness = False
    #             else:
    #                 evenness = True
    #         previous_value = len(v)
    #     print(f"homophonicity: {homophonicity}") # Debugging
    #     return homophonicity #if evenness else 0
    
    def check_unique_values(self) -> bool:
        """Checks if all values in the key are unique"""
        values = []
        for v in self.key.values():
            values.extend(v)
        return len(values) == len(set(values))

    def check_missing_keys(self, ref_key: "CipherKey") -> bool:
        """Checks if the key has missing keys that is present in the reference key."""
        missing_keys = []
        for k in ref_key.key:
            if k not in self.key:
                missing_keys.append(k)
        return missing_keys


class ErrorSequence:
    """Class representing a sequence with errors. Can add addition, deletion, substitution, 
    transposition, and duplication errors to a sequence."""

    def __init__(self, sequence:str, error_type:str, error_rate:float=0.05):

        self.sequence = sequence
        self.is_cipher: bool = sequence[0].isnumeric()
        self.error_type = error_type
        self.error_rate = error_rate
        self.new_sequence = self.add_error()
    
    def __str__(self):
        return f"Error_type={self.error_type}, Error_rate={self.error_rate})"

    def add_error(self):
        """Add errors to a sequence with a given error rate."""
        if self.error_type == "addition":
            return self.add_addition_error()
        elif self.error_type == "deletion":
            return self.add_deletion_error()
        elif self.error_type == "substitution":
            return self.add_substitution_error()
        elif self.error_type == "transposition":
            return self.add_transposition_error()
        elif self.error_type == "duplication":
            return self.add_duplication_error()
        elif self.error_type == "all":
            self.sequence = self.apply_all_errors()
            return self.sequence

        else:
            raise ValueError(f"Unknown error type: {self.error_type}")

    def add_addition_error(self):
        """Add addition errors to a sequence with a given error rate."""
        delimitor = "_" if self.is_cipher else ""
        temp_sequence = self.sequence.split(delimitor)
        for i in range(len(temp_sequence)):
            if random.random() < self.error_rate:
                temp_sequence.insert(i, str(random.randint(0, 99)).zfill(2))  # Insert a random number between 00 and 99 (zfill for double digits)
        return "_".join(temp_sequence) if self.is_cipher else "".join(temp_sequence)

    def add_deletion_error(self):
        """Add deletion errors to a sequence with a given error_rate."""
        delimitor = "_" if self.is_cipher else ""
        temp_sequence = self.sequence.split(delimitor)
        i = 0
        while i < len(temp_sequence):
            if random.random() < self.error_rate:
                del temp_sequence[i]
            else:
                i += 1
        return "_".join(temp_sequence) if self.is_cipher else "".join(temp_sequence)

    def add_substitution_error(self):
        """Add substitution errors to a sequence with a given error_rate."""
        delimitor = "_" if self.is_cipher else ""
        temp_sequence = self.sequence.split(delimitor)
        for i in range(len(temp_sequence)):
            if random.random() < self.error_rate:
                temp_sequence[i] = str(random.randint(0, 99)).zfill(2)# Replace the number with a random number between 00 and 99
        return "_".join(temp_sequence) if self.is_cipher else "".join(temp_sequence)

    def add_transposition_error(self):
        """Add transposition errors to a sequence with a given error_rate."""
        temp_sequence = self.sequence.split('_')
        for i in range(len(temp_sequence) - 1):
            if random.random() < self.error_rate:
                temp_sequence[i], temp_sequence[i + 1] = temp_sequence[i + 1], temp_sequence[i]  # Swap the numbers
        return "_".join(temp_sequence) if self.is_cipher else "".join(temp_sequence)

    def add_duplication_error(self):
        """Add duplication errors to a sequence with a given error_rate."""
        temp_sequence = self.sequence.split('_')
        i = 0
        while i < len(temp_sequence):
            if random.random() < self.error_rate:
                temp_sequence.insert(i, temp_sequence[i])  # Duplicate the number
                i += 1
            i += 1
        return "_".join(temp_sequence) if self.is_cipher else "".join(temp_sequence)
    
    def apply_all_errors(self):
        """Apply all errors (except transposition) to a sequence with a given error_rate."""
        global_error_rate=self.error_rate
        self.error_rate = self.error_rate / 4
        self.sequence = self.add_addition_error()
        self.sequence = self.add_deletion_error()
        self.sequence = self.add_substitution_error()
        self.sequence = self.add_duplication_error()
        self.error_rate = global_error_rate
        return self.sequence