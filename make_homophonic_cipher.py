"""homophonic_cipher.py contains the HonophonicCipher class, that is used to generate
synthetic cryptological data from provided text (as file, directory or string).
The data that is generated uses homophonic substitution."""

import os
import random
import argparse
import csv
from collections import defaultdict
from typing import Union
import string

en_unigrams = {"1500-1549" : [8.0, 1.42, 1.72, 4.57, 15.26, 2.44, 1.72, 8.44, 5.02, 0.02, 0.66, 3.76, 2.47, 6.68, 7.33, 1.4, 0.02, 5.14, 5.77, 9.64, 3.0, 0.65, 2.01, 0.09, 2.68, 0.06],
              "1550-1599" : [8.0, 1.42, 1.72, 4.57, 15.26, 2.44, 1.72, 8.44, 5.02, 0.02, 0.66, 3.76, 2.47, 6.68, 7.33, 1.4, 0.02, 5.14, 5.77, 9.64, 3.0, 0.65, 2.01, 0.09, 2.68, 0.06],
              "1500-1599" : [8.0, 1.42, 1.72, 4.57, 15.26, 2.44, 1.72, 8.44, 5.02, 0.02, 0.66, 3.76, 2.47, 6.68, 7.33, 1.4, 0.02, 5.14, 5.77, 9.64, 3.0, 0.65, 2.01, 0.09, 2.68, 0.06],
              "1600-1649" : [7.96, 1.55, 2.19, 4.34, 13.58, 2.48, 1.74, 7.49, 6.57, 0.08, 0.6, 3.77, 2.47, 6.99, 7.55, 1.6, 0.07, 5.61, 6.08, 9.71, 2.89, 0.79, 1.92, 0.13, 1.72, 0.07],
              "1650-1699" : [7.96, 1.55, 2.19, 4.34, 13.58, 2.48, 1.74, 7.49, 6.57, 0.08, 0.6, 3.77, 2.47, 6.99, 7.55, 1.6, 0.07, 5.61, 6.08, 9.71, 2.89, 0.79, 1.92, 0.13, 1.72, 0.07],
              "1600-1699" : [7.96, 1.55, 2.19, 4.34, 13.58, 2.48, 1.74, 7.49, 6.57, 0.08, 0.6, 3.77, 2.47, 6.99, 7.55, 1.6, 0.07, 5.61, 6.08, 9.71, 2.89, 0.79, 1.92, 0.13, 1.72, 0.07],
              "1700-1749" : [7.81, 1.5, 2.65, 4.14, 12.71, 2.48, 1.73, 6.31, 7.0, 0.15, 0.5, 3.71, 2.7, 6.84, 7.73, 1.97, 0.11, 6.05, 6.37, 9.21, 2.91, 1.11, 2.04, 0.19, 2.02, 0.04],
              "1750-1799" : [7.81, 1.5, 2.65, 4.14, 12.71, 2.48, 1.73, 6.31, 7.0, 0.15, 0.5, 3.71, 2.7, 6.84, 7.73, 1.97, 0.11, 6.05, 6.37, 9.21, 2.91, 1.11, 2.04, 0.19, 2.02, 0.04],
              "1750-1849" : [7.81, 1.5, 2.65, 4.14, 12.71, 2.48, 1.73, 6.31, 7.0, 0.15, 0.5, 3.71, 2.7, 6.84, 7.73, 1.97, 0.11, 6.05, 6.37, 9.21, 2.91, 1.11, 2.04, 0.19, 2.02, 0.04],
              "1700-1799" : [7.81, 1.5, 2.65, 4.14, 12.71, 2.48, 1.73, 6.31, 7.0, 0.15, 0.5, 3.71, 2.7, 6.84, 7.73, 1.97, 0.11, 6.05, 6.37, 9.21, 2.91, 1.11, 2.04, 0.19, 2.02, 0.04],
              "1800-1849" : [7.97, 1.54, 2.64, 4.17, 12.57, 2.42, 1.93, 6.22, 6.98, 0.14, 0.66, 4.01, 2.58, 6.99, 7.68, 1.77, 0.11, 5.93, 6.37, 9.17, 2.82, 1.02, 2.17, 0.17, 1.91, 0.05],
              "1850-1899" : [7.97, 1.54, 2.64, 4.17, 12.57, 2.42, 1.93, 6.22, 6.98, 0.14, 0.66, 4.01, 2.58, 6.99, 7.68, 1.77, 0.11, 5.93, 6.37, 9.17, 2.82, 1.02, 2.17, 0.17, 1.91, 0.05],
              "1850-1849" : [7.97, 1.54, 2.64, 4.17, 12.57, 2.42, 1.93, 6.22, 6.98, 0.14, 0.66, 4.01, 2.58, 6.99, 7.68, 1.77, 0.11, 5.93, 6.37, 9.17, 2.82, 1.02, 2.17, 0.17, 1.91, 0.05],
              "1800-1899" : [7.97, 1.54, 2.64, 4.17, 12.57, 2.42, 1.93, 6.22, 6.98, 0.14, 0.66, 4.01, 2.58, 6.99, 7.68, 1.77, 0.11, 5.93, 6.37, 9.17, 2.82, 1.02, 2.17, 0.17, 1.91, 0.05],
              "1900-1949" : [7.97, 1.56, 2.59, 4.09, 12.42, 2.25, 2.0, 6.19, 7.14, 0.13, 0.71, 4.2, 2.54, 6.92, 7.64, 1.74, 0.11, 5.73, 6.5, 9.26, 2.85, 0.97, 2.22, 0.16, 2.04, 0.04], #CHANGED 2024-04-07. Because corpus(Lampeter) is 1500-1800
              "1950-1999" : [7.97, 1.56, 2.59, 4.09, 12.42, 2.25, 2.0, 6.19, 7.14, 0.13, 0.71, 4.2, 2.54, 6.92, 7.64, 1.74, 0.11, 5.73, 6.5, 9.26, 2.85, 0.97, 2.22, 0.16, 2.04, 0.04],
              "1900-1999" : [7.97, 1.56, 2.59, 4.09, 12.42, 2.25, 2.0, 6.19, 7.14, 0.13, 0.71, 4.2, 2.54, 6.92, 7.64, 1.74, 0.11, 5.73, 6.5, 9.26, 2.85, 0.97, 2.22, 0.16, 2.04, 0.04]
              }

class HomophonicCipher: # encrypt/generate, text/file/folder, codes_per_letter, language, alphabet, frequency, seed, time_period
  """
  A class for encrypting plaintext(s) using homophonic substitution.
  """

  def __init__(self, set_frequencies:dict, set_alphabet:str=string.ascii_uppercase,
               set_punctuation:str=string.punctuation, set_homophonicity: Union[int, str]=2, set_seed=None):
    

    self.alphabet = set_alphabet
    self.punctuation = set_punctuation if '-' in set_punctuation else set_punctuation + '-' #CHANGED 2024-04-07
    self.frequency_dict = set_frequencies
    self.set_seed = set_seed
    self.homophonicity=set_homophonicity #CHANGED 2024-04-12 Added so that this can be set from prompt
    self.encryption_dict = self.initialize_encryption_dict()

  def initialize_encryption_dict(self):
    if isinstance(self.homophonicity, int):
        return self._even_splits(self.alphabet, self.homophonicity)
    elif self.homophonicity == "uneven":
        encryption_dict = {}
        for years, freqs in self.frequency_dict.items():
            partial_dict = self._uneven_splits(self.alphabet, freqs)
            encryption_dict.update(partial_dict)
        return encryption_dict
    else:
        raise ValueError(f"Invalid homophonicity setting: {self.homophonicity}")


  def encrypt_homophonic(self, plaintexts, plaintext_type, include_errors=False, error_type="all",
                       error_frequency=0.1, include_spacing=False, max_length=150, set_seed=None):
    """Encrypts given plaintext(s) using homophonic substitution based on the
    specified input type. Dynamically generates a new encryption dictionary for each plaintext
    with a new random seed unless specified."""

    # Set dynamic random seed if none provided
    if set_seed is None:
        set_seed = random.SystemRandom().randint(0, 999999)
    random.seed(set_seed)  # Seed the random number generator

    # Initialize the encryption dictionary with the new seed
    self.encryption_dict = self.initialize_encryption_dict()

    # Initialize the dictionary to store encrypted outputs
    self.plaintext_dict = defaultdict(dict)

    # Handle different types of plaintext input
    if plaintext_type == "string":
        self._process_text(plaintexts, include_spacing, include_errors, error_type, error_frequency, max_length)
    elif plaintext_type == "file":
        content = self._read_file_content(plaintexts)
        self._process_text(content, include_spacing, include_errors, error_type, error_frequency, max_length, set_seed)
    elif plaintext_type == "folder":
        for filename in os.listdir(plaintexts):
            file_path = os.path.join(plaintexts, filename)
            if file_path.endswith('.txt'):
                content = self._read_file_content(file_path)
                self._process_text(content, include_spacing, include_errors, error_type, error_frequency, max_length, filename)

    # Optionally, reseed the RNG for external uses beyond this function
    random.seed()  # Reseed RNG to system time or other source to avoid using the deterministic seed beyond this point

  def _process_text(self, text, include_spacing, include_errors, error_type, error_frequency, max_length, source=None):
    formatted_text = self._format_plaintext(text, include_spacing)
    chunks = self._chunk_plaintext(formatted_text, max_length)
    encrypted_chunks = []
    dictionaries_used = []  # Store dictionaries used for each chunk

    for index, chunk in enumerate(chunks):
        self.encryption_dict = self.initialize_encryption_dict()
        if include_errors:
            chunk = self._insert_errors(chunk, error_type, error_frequency)

        encrypted_chunk = self._encryption_homophonic(chunk, self.encryption_dict)
        encrypted_chunks.append(encrypted_chunk)
        dictionaries_used.append(self.encryption_dict.copy())  # Make a copy of the dictionary

    self.plaintext_dict[source] = {
        "to_encrypt": chunks,
        "encrypted": encrypted_chunks,
        "dictionaries": dictionaries_used  # Store dictionaries for each chunk
    }

  def _read_file_content(self, file_path):
    """
    Reads the file content, skipping metadata lines.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = []
        skip_metadata = True
        for line in file:
            if skip_metadata and line.startswith('#'):
                continue  # Skip metadata lines
            skip_metadata = False  # Once a non-metadata line is encountered, read normally
            content.append(line)
        return ''.join(content)

  def _format_plaintext(self, text_to_encrypt:str, include_spacing:bool):
    """
    Cleans and formats a given text by removing punctuation, spaces, newlines,
    and tabs, and converting all characters to uppercase.
    """

    clean_plaintext = ""

    if include_spacing == True:
      for char in text_to_encrypt:
        #if char not in self.punctuation and char != "\n" and char != "\t": ##CHANGED THIS 2024-04-07
        # Filter out non-ASCII characters by checking the ordinal number is within ASCII range
        if 0 <= ord(char) <= 127: ##CHANGED THIS 2024-04-07
          if char not in self.punctuation and char != "\n" and char != "\t" and not char.isdigit():
  
            clean_plaintext += char.upper()
    else:
      for char in text_to_encrypt:
        #if char not in self.punctuation and char != " " and char != "\n" and \ ##CHANGED THIS 2024-04-07
        # Filter out non-ASCII characters by checking the ordinal number is within ASCII range
        if 0 <= ord(char) <= 127: ##CHANGED THIS 2024-04-07
          if char not in self.punctuation and char != " " and char != "\n" and char != "\t" and not char.isdigit():

            clean_plaintext += char.upper()

    return clean_plaintext

  def _insert_errors(self, plaintext:str, set_seed, error_type:str,
                     error_frequency:float):
    """
    Inserts errors into a plaintext string based on specified parameters.
    """

    random.seed(set_seed)
    pt_len = len(plaintext)
    pt_with_errors = ""

    if error_frequency >= 1:
      raise ValueError(f"{error_frequency} not valid. Must be less than 1.")

    error_types = ["additions", "deletions", "doubles"]
    total_errors = max(1, int(pt_len*error_frequency))
    error_indexes = random.sample(range(pt_len), total_errors)
    sorted_indexes = sorted(error_indexes)

    if error_type != "all" and error_type not in error_types:
      error_types.append("all")
      raise ValueError(f"'{error_type}' not valid. Available options: {error_types}")

    elif error_type == "all":
      for i, char in enumerate(plaintext):
        if i in sorted_indexes:
          pt_with_errors += self._make_error_choice(char, random.choice(error_types))
        else:
          pt_with_errors += char

    else:
      for i, char in enumerate(plaintext):
        if i in sorted_indexes:
          pt_with_errors += self._make_error_choice(char, error_type)
        else:
          pt_with_errors += char

    return pt_with_errors


  def _make_error_choice(self, plaintext_char:str, error_type:str):
    """
    Modifies a plaintext character based on the specified type of error.
    """

    if error_type == "additions":
      rand_char = random.choice(self.alphabet)
      return plaintext_char + rand_char
    elif error_type == "deletions":
      return ""
    elif error_type == "doubles":
      return plaintext_char + plaintext_char
    else:
      raise ValueError(f"{error_type} not valid.")


  def _chunk_plaintext(self, text_string:str, max_length:int):
      """
      Divides a text string into chunks of a specified maximum length.
      Only returns chunks that are exactly max_length characters long.

      Parameters:
      - text_string (str): The string to be chunked.
      - max_length (int): The exact length of each chunk.

      Returns:
      - list of str: A list containing the chunked substrings of exact length max_length.
      """
      return [text_string[i:i+max_length] for i in range(0, len(text_string), max_length) if len(text_string[i:i+max_length]) == max_length]


  def _even_splits(self, alphabet, num_options):
    random.seed(self.set_seed)  # Use the dynamically updated instance seed
    total_numbers_needed = len(alphabet) * num_options
    random_unique_numbers = [random.randint(100, 999) for _ in range(total_numbers_needed)]

    encryption_dict = {}
    index = 0
    for char in alphabet:
        encryption_dict[char] = random_unique_numbers[index:index + num_options]
        index += num_options

    return encryption_dict

  def _uneven_splits(self, alphabet, frequencies):
    # Generate a unique seed for each call to ensure unique encryption mappings
    random_seed = random.SystemRandom().randint(0, 999999)
    random.seed(random_seed)

    avg = sum(frequencies) / len(frequencies)
    sort_freq = sorted(set(frequencies))
    less_idx, more_idx = self._set_index(sort_freq, avg)
    options_dict_freq = self._frequency_to_options_dict(less_idx, more_idx, sort_freq)

    total_numbers_needed = sum(options_dict_freq.values()) + 50
    random_unique_numbers = [f"{i:03d}" for i in random.sample(range(0, 1000), total_numbers_needed)]

    encryption_dict = {}
    starting_index = 0
    for i, let in enumerate(alphabet):
        let_freq = frequencies[i]
        num_options = options_dict_freq[let_freq]
        encrypts = random_unique_numbers[starting_index:starting_index + num_options]

        encryption_dict[let] = encrypts
        starting_index += num_options

    return encryption_dict






  def _set_index(self, sorted_frequencies, average_frequency):
    """
    Determines the indices for splitting the list of sorted frequencies into
    categories below and above the average; the "under" group (frequencies below
    or equal to the average) is split into two, and the "over" group
    (frequencies above the average) is split into three parts
    """

    total_under = 0
    total_over = 0

    for i, freq in enumerate(sorted_frequencies):
      if freq <= average_frequency:
        total_under += 1
      else:
        total_over += 1

    under_option_split = round(total_under/2)
    over_option_split = round(total_over/3)

    return under_option_split, over_option_split


  def _frequency_to_options_dict(self, low_freq_split_index,
                                 high_freq_split_index, sorted_frequencies):
    """
    Creates a dictionary mapping each unique frequency to a specified number of
    options based on its position. Frequencies are categorized into five groups.
    """

    options_dictionary = dict()

    for i, freq in enumerate(sorted_frequencies):
      if i < low_freq_split_index:
        options_dictionary[freq] = 1
      elif i < (low_freq_split_index*2):
        options_dictionary[freq] = 2
      elif i < ((low_freq_split_index*2)+high_freq_split_index):
        options_dictionary[freq] = 3
      elif i < ((low_freq_split_index*2)+(high_freq_split_index*2)):
        options_dictionary[freq] = 4
      else:
        options_dictionary[freq] = 5

    return options_dictionary


  def _encryption_homophonic(self, plaintext, encryption_dictionary):
    """
    Encrypts a given plaintext using a homophonic substitution cipher. This
    function iterates over each letter in the plaintext and substitutes it with
    a random choice from the corresponding list of codes in the encryption
    dictionary. If a letter in the plaintext does not have a corresponding entry
    in the encryption dictionary, it is left unchanged.
    """

    ciphertext = []

    for let in plaintext:
      try:
        cipher_let = random.choice(encryption_dictionary[let])
        ciphertext.append(str(cipher_let))
      except:
        ciphertext.append(let)

    return "_".join(ciphertext)


def set_alpha_to_crypts(encryption_dictionary):
  """
  Converts an encryption dictionary into a list of formatted strings, each
  representing a letter and its encrypted values.
  """

  set_list = []

  for let in encryption_dictionary.keys():
    vals = encryption_dictionary[let]
    ttt = ""

    ttt += let
    ttt += ":["
    for i, val in enumerate(vals):
      if i != 0:
        ttt += "|" + str(val)
      else:
        ttt += str(val)

    ttt += "]"

    set_list.append(ttt)

  return set_list




def writing_function(index, plain_text, cipher_text, dictionary, file_name):
    dict_as_string = ','.join([f'{key}:[{"|".join(map(str, vals))}]' for key, vals in dictionary.items()])
    return [[index, plain_text, dict_as_string, cipher_text, file_name]]


# # Main function for homophonic encryption
def homophonicity_type(value):
  """This is ugly, but it's the "only" way to check for an integer or the string 'uneven' in argparse."""
  try:
      # First, try to convert it to an integer
      return int(value)
  except ValueError:
      # If it fails, check if it is the string 'uneven'
      if value == "uneven":
          return value
      else:
          # If it's neither an integer nor 'uneven', raise an error
          raise argparse.ArgumentTypeError("Homophonicity must be an integer or 'uneven'.")

def main_homophonic():


# Make sure to pass a dynamic seed or None to use dynamic seeding
    parser = argparse.ArgumentParser(description="Encrypt text or file homophonic")
    parser.add_argument("plaintexts", help="string, file, or directory")
    parser.add_argument("plaintext_type", help="'string', 'file', or 'folder'")
    parser.add_argument("output_file", type=str, help="Name of output file")
    parser.add_argument("--include_errors", type=bool, default=False,
                        help="To include errors or not")
    parser.add_argument("--error_type", type=str, default="all",
                        help="'all', 'additions', 'deletions', or 'doubles'")
    parser.add_argument("--error_frequency", type=float, default=0.00,  #changed from 0.05 2024-08
                        help="How often errors will appear in ciphertext")
    parser.add_argument("--include_spacing", type=bool, default=False,
                        help="To include spacing or not")
    parser.add_argument("--set_length", type=int, default=150, help="Fixed length for each output line")
    parser.add_argument("--set_seed", default=None)
    parser.add_argument("--set_alphabet", type=str, default=string.ascii_uppercase)
    parser.add_argument("--set_punctuation", type=str, default=string.punctuation)
    parser.add_argument("--set_homophonicity", type=homophonicity_type, default=2)

    args = parser.parse_args()

    homoCi = HomophonicCipher(en_unigrams, set_alphabet=args.set_alphabet,
                              set_punctuation=args.set_punctuation, set_homophonicity=args.set_homophonicity)

    homoCi.encrypt_homophonic(plaintexts=args.plaintexts,
                              plaintext_type=args.plaintext_type,
                              include_errors=args.include_errors,
                              error_type=args.error_type,
                              error_frequency=args.error_frequency,
                              include_spacing=args.include_spacing,
                              max_length=args.set_length, set_seed=args.set_seed)

    file_names = homoCi.plaintext_dict.keys()
    rows_to_write = []
    line_count = 0

    for filename in file_names:
        encrypted_chunks = homoCi.plaintext_dict[filename]["encrypted"]
        plain_texts = homoCi.plaintext_dict[filename]["to_encrypt"]
        dictionaries_used = homoCi.plaintext_dict[filename]["dictionaries"]  # Retrieve the list of dictionaries

        for i, (plain_text, cipher_text, dict_used) in enumerate(zip(plain_texts, encrypted_chunks, dictionaries_used)):
            row = writing_function(i, plain_text, cipher_text, dict_used, filename)
            rows_to_write.extend(row)
            line_count += 1
            if line_count >= 1000:
                break
        if line_count >= 1000:
            break


    write_filename = args.output_file + ".csv"
    with open(write_filename, "w", newline='', encoding='us-ascii', errors='replace') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(rows_to_write)

    print(f"Processed {line_count} lines into {write_filename}.")

if __name__ == "__main__":
    main_homophonic()
