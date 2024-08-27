"""error_sequence.py contains the ErrorSequence class. This class is used to add errors to a sequence of numbers and is used by the Cipher class.
The errors that can be added are:
• addition • deletion • substitution • transposition • duplication errors.
When creating an instance of the ErrorSequence class, the error rate can best set. 
"""
import random

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