import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Union
import pandas as pd
from cipher import Cipher, CipherKey, ErrorSequence
from cipher_alignment_model import CipherAlignmentModel, NLTKIBMModelAdapter, GenericIBMModelAdapter
from alignment import Alignment

@dataclass
class KeyRecreationResult:
    """Dataclass for storing the results of the evaluation of a recreated cipherkey."""
    cipher: Cipher
    evaluations: list[tuple[CipherAlignmentModel, tuple[float, float, float, CipherKey]]] # List of tuples with the model and the evaluation results
    def __str__(self):
        return f"{self.cipher}\n{self.evaluations}"
    
@dataclass
class AlignmentResult:
    """Dataclass for storing the results of the evaluations of alignments."""
    cipher: Cipher 
    evaluations: list[dict]  # Use a list of dictionaries to hold detailed results per alignment

    def __str__(self):
        results_str = '\n'.join(str(evaluation) for evaluation in self.evaluations)
        return f"{self.cipher}\n{results_str}"
    

def sample_lines_from_large_file(file_path, sample_size):
    """Sample random lines from a large file using reservoir sampling."""
    sampled_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i < sample_size:
                sampled_lines.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sampled_lines[j] = line
    return sampled_lines

def create_row(cipher, result, model_name, n_iter)-> dict[str, Union[str, float, int, None]]:
    """Creates a 'row' to be used in a data-table with the information for one cipher."""

    precision = result[0]
    recall = result[1]
    f1_score = result[2]
    recreated_key = result[3]

    return {
        #'homophonicity': cipher.homophonicity,
        'csv_filename': cipher.csv_filename,
        'homophonicity': cipher.homophonicity,
        'error': str(cipher.error.error_type if cipher.error is not None else None),
        'error rate': str(cipher.error.error_rate if cipher.error is not None else None),
        'precision': round(precision, 4) if precision is not None else None,
        'recall': round(recall, 4) if recall is not None else None,
        'f1_score': round(f1_score, 4)if f1_score is not None else None,
        'filename': cipher.filename,
        'model': model_name,
        'segment_size': cipher.seq_size,
        'iterations': n_iter,
        'id_tag': cipher.id_tag,
        'missing_keys': recreated_key.check_missing_keys(cipher.original_reference_key) if recreated_key is not None else None,
        'recreated_key': recreated_key if recreated_key is not None else None
    }

def write_to_file(ciphers_and_results: list[KeyRecreationResult], filename: str) -> None:
    """Write cipher information and evaluation results to either an Excel file or a CSV file."""
    # Create a list of dictionaries, each containing the information for one cipher
    data = []
    for result in ciphers_and_results:
        for model, evaluation in result.evaluations:
            data.append(create_row(result.cipher, evaluation, model.NAME, model.n_iter))
    # Then, create the DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to an Excel file
    df.to_excel(filename, index=False)
    print(f"Results for Cipher key written to {filename}")

def write_alignment_to_file(ciphers_and_results: list[AlignmentResult], filename: str) -> None:
    """Write alignment evaluation results to an Excel file."""
    data = []
    for alignment_result in ciphers_and_results:
        for evaluation in alignment_result.evaluations:
            data.append({
                'cipher_id': alignment_result.cipher.id_tag,
                'csv_filename': alignment_result.cipher.csv_filename,
                'homophonicity': alignment_result.cipher.homophonicity,
                'model': evaluation['model'],
                'error': evaluation['error'],
                'error rate': evaluation['error rate'],
                'length': evaluation['length'],
                'checklength': evaluation['checklength'],
                'iterations': evaluation['iterations'],
                'segment_size': alignment_result.cipher.seq_size,
                'precision': evaluation['precision'],
                'recall': evaluation['recall'],
                'f1_score': evaluation['f1_score'],
                #'AER': evaluation['AER'],

                #'Levenshtein Distance': evaluation['Levenshtein Distance'],
                'alignment': evaluation['alignments']
            })

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Alignment results written to {filename}")

def train_and_evaluate_model(model: CipherAlignmentModel, cipher: Cipher) -> tuple[float, float, float, CipherKey]:
    """Trains the model and evaluates it on the given cipher. Returns the precision, recall and F1-score.
    Also returns the recreated key."""
    reference_key=cipher.original_reference_key
    recreated_key=CipherKey(model.translation_probs)
    precision, recall, f1_score =recreated_key.compare_keys(reference_key)
    return precision, recall, f1_score, recreated_key

def evaluate_alignments(model: CipherAlignmentModel, cipher: Cipher, continuous=False) -> dict:
    """Evaluates the alignments produced by the model for the given cipher."""
    #print("EVALUATING ALIGNMENTS!!!")
    alignments = model.align_sentences(continuous=continuous)
    
    concatenated_alignment_data = []
    for alignment_obj in alignments:
        
        concatenated_alignment_data.extend(alignment_obj.alignment)
    concatenated_alignment = Alignment(concatenated_alignment_data, model.NAME)


    print(f"LEN: {len(concatenated_alignment)}")
    # Calculate metrics on the concatenated alignment
    aer = concatenated_alignment.calculate_aer(cipher.reference_alignment_data)
    ld = concatenated_alignment.calculate_levenshtein(cipher.reference_alignment_data)
    precision, recall, fscore = concatenated_alignment.get_precision_recall_fscore(cipher.reference_alignment_data)

    evaluation_details = {
        "csv_filename": cipher.csv_filename,
        'homophonicity': cipher.homophonicity,
        'error': str(cipher.error.error_type if cipher.error is not None else None),
        'error rate': str(cipher.error.error_rate if cipher.error is not None else None),
        'length': len(cipher.plaintext),
        "checklength": len(concatenated_alignment), 
        "model": model.NAME,
        "iterations": model.n_iter,
        "segment_size": cipher.seq_size,
        "precision": precision,
        "recall": recall,
        "f1_score": fscore,
        #"AER": aer,
        #"Levenshtein Distance": ld,
        "alignments": concatenated_alignment if len(concatenated_alignment) < 1000 else 'Too many to display',
    }
    return evaluation_details

def generate_xlsx_file_name(name:str) -> str:
    """Generates a file name based on the current date and time."""
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M")
    return f'{name}_{now_str}.xlsx'


        

def main():
    # seq_sizes = [5]
    # n_iters = [5]
    # sample_size = 1

    seq_sizes = [5]
    n_iters = [10]
    sample_size = 100

    cipherkey_out_filename = generate_xlsx_file_name("cipherkey_evaluation_OPTIMIZED_FINAL")
    alignment_out_filename = generate_xlsx_file_name("alignment_evaluation_OPTIMIZED_FINAL")

    path_to_file = 'test_data.csv'
    # your code here
    #path_to_file = 'test_data\\en-HistCorp-1500-1800-100-THREE.csv'
    # path_to_file= "DATA_no_errors"
    #path_to_file= "tiny_test"
    #path_to_file = "Generated-Gutenberg-English-Homophonic-Ciphertexts"
    #path_to_file='Generated-Historic-English-1500-1800-Homophonic-Ciphertexts'
    key_results = []
    alignment_results = []

    start = time.time()
    for seq_size in seq_sizes:
        sampled_lines = sample_lines_from_large_file(path_to_file, sample_size)
        list_of_random_ciphers = []

        error_types = ["addition", "deletion", "substitution", "duplication", "all"]
        for random_line in sampled_lines:
            line = random_line.strip().split(';')

            # Create a Cipher object with the correct ciphertext
            cipher_correct = Cipher(id_tag=line[0],plaintext=line[1],str_key=line[2],ciphertext=line[3],filename=line[4], homophonicity=line[5], csv_filename=path_to_file, seq_size=seq_size)
            list_of_random_ciphers.append(cipher_correct)
            # Create Cipher objects with different types of errors
            for error_type in error_types:
                error = ErrorSequence(line[3], error_type)
                cipher_erroneous = Cipher(id_tag=line[0],plaintext=line[1],str_key=line[2],ciphertext=line[3],filename=line[4], homophonicity=line[5], csv_filename=path_to_file, error=error, seq_size=seq_size)
                list_of_random_ciphers.append(cipher_erroneous)
        
        for n_iter in n_iters:
            print(f"Processing with seq_size={seq_size} and n_iter={n_iter}")
            for cipher in list_of_random_ciphers:
                print(f"Processing cipher {cipher.id_tag}")

                # models = [GenericIBMModelAdapter(cipher.bitext, n_iter, "model1", use_null=False),
                #     GenericIBMModelAdapter(cipher.bitext, n_iter, "model1", use_null=True),
                #     NLTKIBMModelAdapter(cipher.bitext, n_iter, "model1"),
                #     NLTKIBMModelAdapter(cipher.bitext, n_iter, "model1", use_null=False),
                models = [NLTKIBMModelAdapter(cipher.bitext, n_iter, "model2", use_null=False)]
                        #NLTKIBMModelAdapter(cipher.bitext, n_iter, "model3", use_null=False)]
                #GenericIBMModelAdapter(cipher.bitext, n_iter, "model1", use_null=False)
                for model in models:
                    print(f"Training model {model.NAME} on cipher {cipher.id_tag}")
                    # Evaluate key creation
                    key_evaluation = train_and_evaluate_model(model, cipher)
                    key_results.append(KeyRecreationResult(cipher, [(model, key_evaluation)]))
              
                    # Evaluate alignments
                    alignment_evaluation = evaluate_alignments(model, cipher, continuous=True)
                    alignment_results.append(AlignmentResult(cipher, [alignment_evaluation]))
            print(f"Completed processing for seq_size={seq_size} and n_iter={n_iter}")

    write_to_file(key_results, cipherkey_out_filename)
    write_alignment_to_file(alignment_results, alignment_out_filename)
    end = time.time()
    print("Evaluation complete")
    print(f"Current time: {datetime.now()}")
    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    main()
