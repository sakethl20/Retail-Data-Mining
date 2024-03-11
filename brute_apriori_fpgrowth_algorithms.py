import csv
import sys
import numpy as np
from itertools import chain, combinations
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import time

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

list_input = input("Enter a dataset to analyze: Amazon, Best Buy, Kmart, Nike, or General: ")
if list_input.lower() == 'amazon':
    itemset_list = pd.read_csv('amazonlist.csv')
    the_transactions = pd.read_csv('amazontransactions.csv')
elif list_input.lower() == 'best buy':
    itemset_list = pd.read_csv('bestbuylist.csv')
    the_transactions = pd.read_csv('bestbuytransactions.csv')
elif list_input.lower() in ['kmart', 'k-mart']:
    itemset_list = pd.read_csv('kmartlist.csv')
    the_transactions = pd.read_csv('kmarttransactions.csv')
elif list_input.lower() == 'nike':
    itemset_list = pd.read_csv('nikelist.csv')
    the_transactions = pd.read_csv('niketransactions.csv')
elif list_input.lower() == 'general':
    itemset_list = pd.read_csv('generallist.csv')
    the_transactions = pd.read_csv('generaltransactions.csv')
else:
    print("Enter Valid Info (correct spelling)")
    sys.exit()

order = sorted(itemset_list['ItemName'])

dataset = []
for lines in the_transactions['Transaction']:
    trans = list(lines.strip().split(','))
    trans1 = list(np.unique(trans))
    trans1.sort(key=lambda x: order.index(x.strip()))
    dataset.append(sorted(trans1))

transaction_num = len(dataset)

min_support = int(input("Enter the minimum support count (integer, percent between 1 and 100): "))
min_confidence = float(input("Enter the minimum confidence (decimal, percent between 0 and 100): "))

candidate_itemsets = {}               
freq_itemsets = {}    
support_count_L = {}   
itemset_size = 1     # set the initial itemset size
non_frequent = {itemset_size: []}

# populate candidate itemsets with singleton itemsets
candidate_itemsets[itemset_size] = [[f] for f in order]

# print the initialized candidate dictionary for verification
print("Candidate Itemsets:", candidate_itemsets)

def frequent_itemsets(candidate_itemsets, transactions, minimum_support, non_frequent):
    # Initialize lists to store frequent itemsets, their support counts, and new non-frequent itemsets
    frequent_itemsets_list = []
    support_count_list = []
    new_non_frequent = []
    
    # Get the total number of transactions in the dataset
    transaction_num = len(transactions)
    # Get the current itemset size
    K = len(non_frequent.keys())
    
    # Iterate through each candidate itemset
    for i in range(0, len(candidate_itemsets)):
        # Flag to check if the candidate itemset is a subset of any non-frequent itemset
        is_candidate_set = 0
        
        # Check against non-frequent itemsets of the previous size (if any)
        if K > 0:
            for j in non_frequent[K]:
                if set(j).issubset(set(candidate_itemsets[i])):
                    is_candidate_set = 1
                    break
        
        # If the candidate itemset is not a subset of any non-frequent itemset, proceed
        if is_candidate_set == 0:
            # Count the occurrences of the candidate itemset in the transactions
            frequent_count = count_items(candidate_itemsets[i], transactions)
            
            # Check if the frequent count meets the minimum support threshold
            if frequent_count >= (minimum_support / 100) * transaction_num:
                # Append the frequent itemset and its support count to the lists
                frequent_itemsets_list.append(candidate_itemsets[i])
                support_count_list.append(frequent_count)
            else:
                # If not frequent, add the itemset to the list of new non-frequent itemsets
                new_non_frequent.append(candidate_itemsets[i])
    
    # Return the lists containing frequent itemsets, their support counts, and new non-frequent itemsets
    return frequent_itemsets_list, support_count_list, new_non_frequent

def count_items(candidate_itemset, transactions):
    count = 0
    for i in range (0, len(transactions)):
        if set(candidate_itemset).issubset(set(transactions[i])):
            count += 1
    return count

def print_table (table, support_count):
    print("Itemset | Count")
    for i in range (0, len(table)):
        print("{} : {}".format(table[i], support_count[i]))
    print("\n\n")

def join_itemsets(itemset1, itemset2, order):
    itemset1.sort(key=lambda x: order.index(x))
    itemset2.sort(key=lambda x: order.index(x))
    
    for i in range(len(itemset1) - 1):
        if itemset1[i] != itemset2[i]:
            return []
    
    if order.index(itemset1[-1]) < order.index(itemset2[-1]):
        return itemset1 + [itemset2[-1]]
    
    return []

def get_candidate_set(frequent_itemsets, order):
    candidate_set = []
    for i in range(len(frequent_itemsets)):
        for j in range(i + 1, len(frequent_itemsets)):
            joined_itemset = join_itemsets(frequent_itemsets[i], frequent_itemsets[j], order)
            
            if len(joined_itemset) > 0:
                candidate_set.append(joined_itemset)
    
    return candidate_set

from itertools import combinations, chain

def possible_subsets(s):
    a = list(s)
    subsets = list(chain.from_iterable(combinations(a, r) for r in range(1, len(a) + 1)))
    return subsets

def write_association_rules(lhs_itemset, rhs_itemset, confidence, support, transaction_num, rule_number):
    association_rules_str = (
        f"Rule {rule_number}: {list(lhs_itemset)} -> {list(rhs_itemset)}\n"
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Support: {(support / transaction_num) * 100:.2f}%\n\n"
    )
    return association_rules_str

frequent_set, support, new_non_frequent = frequent_itemsets(candidate_itemsets[itemset_size], dataset, min_support, non_frequent)

freq_itemsets[itemset_size] = frequent_set
non_frequent[itemset_size] = new_non_frequent
support_count_L[itemset_size] = support

def print_itemset_table(itemset_type, itemsets, support_counts):
    print(f"\nTable {itemset_type}:\n")
    for i, itemset in enumerate(itemsets):
        itemset_str = ','.join(itemset)
        count = support_counts[i]
        print(f"{itemset_str} : {count}")
print_itemset_table("Candidate", candidate_itemsets[1], [count_items(item, dataset) for item in candidate_itemsets[1]])
print_itemset_table("Frequent", freq_itemsets[1], support_count_L[1])

start_time = time.time() ## Starting the time calc here
# Initialize variables for the brute-force approach
K = itemset_size + 1  # Increment the itemset size
candidate_set = 0  # Flag to check if there are frequent itemsets
candidate_itemsets = []  # List to store candidate itemsets at each iteration

# Continue generating candidate itemsets until none are frequent for the current itemset size
while candidate_set == 0:
    # Generate candidate itemsets using the brute-force method
    candid_itemsets = get_candidate_set(freq_itemsets[K-1], order)
    # Append the generated candidate itemsets to the list
    candidate_itemsets.append(candid_itemsets)
    print("Table Candidate Itemsets {}: \n".format(K))
    # Print the table of candidate itemsets along with their support counts
    print_table(candid_itemsets, [count_items(item, dataset) for item in candid_itemsets])
    
    # Calculate frequent itemsets and update dictionaries
    frequent_set, support, new_non_frequent = frequent_itemsets(candidate_itemsets[-1], dataset, min_support, non_frequent)
    # Update dictionaries with the frequent itemsets, non-frequent itemsets, and support counts
    freq_itemsets.update({K: frequent_set})
    non_frequent.update({K: new_non_frequent})
    support_count_L.update({K: support})
    
    # Check if there are no frequent itemsets for the current itemset size
    if len(freq_itemsets[K]) == 0:
        candidate_set = 1  # Set the flag to terminate the loop
    else:
        print(f"Table Frequent Itemsets {K}:\n")
        # Print the table of frequent itemsets along with their support counts
        print_table(freq_itemsets[K], support_count_L[K])
    
    K += 1  # Increment the itemset size for the next iteration

# Initialize variables for final association rules
final_association_rules = ""
rule_number = 1

# Loop over each itemset size (starting from 1) in the frequent itemsets
for itemset_size in range(1, len(freq_itemsets)):
    # Loop over each itemset in the current itemset size
    for itemset in freq_itemsets[itemset_size]:
        # Generate all possible non-empty subsets of the current itemset
        subsets = list(possible_subsets(set(itemset)))
        subsets.pop()  # Remove the empty set

        # Loop over each subset of the current itemset
        for subset in subsets:
            # Create item1 as a set representing the current subset
            item1 = set(subset)
            # Convert the current itemset to a set
            itemset_set = set(itemset)
            # Generate item2 as the complement of item1 in the itemset
            item2 = itemset_set - item1
            
            # Calculate support counts for the itemsets and subsets
            support_frequent_set = count_items(itemset_set, dataset)
            support_item1 = count_items(item1, dataset)
            support_item2 = count_items(item2, dataset)

            # Calculate confidence of the association rule
            confidence = support_frequent_set / support_item1

            # Check if the confidence and support meet the user-defined thresholds
            if confidence >= (min_confidence / 100) and support_frequent_set >= (min_support / 100) * transaction_num:
                # Generate and append the association rule to the final result
                final_association_rules += write_association_rules(item2, item1, confidence, support_frequent_set, transaction_num, rule_number)
                # Increment the rule number for the next association rule
                rule_number += 1

end_time = time.time() ## Ending the time calc here
elapsed_time = end_time - start_time

# Print the final association rules
print("Final Association Rules: \n")
print(final_association_rules)
print(f"\nProgram execution time: {elapsed_time:.4f} seconds")

from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder

start_time = time.time() ## Starting the time calc here

# Use Apriori implementation from mlxtend
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
mlxtend_freq_itemsets = mlxtend_apriori(df, min_support=(min_support / 100), use_colnames=True)

# Extract association rules from frequent itemsets using mlxtend
mlxtend_rules = mlxtend_association_rules(mlxtend_freq_itemsets, metric="confidence", min_threshold=min_confidence / 100)

# Initialize variables for final association rules (mlxtend)
final_association_rules_mlxtend = ""
rule_number_mlxtend = 1

# Loop over each rule generated by mlxtend
for idx, row in mlxtend_rules.iterrows():
    # Extract items and details from mlxtend result
    item1_mlxtend = set(row['antecedents'])
    item2_mlxtend = set(row['consequents'])
    confidence_mlxtend = row['confidence']
    support_mlxtend = row['support'] * transaction_num  # Convert support to count

    # Check if the confidence and support meet the user-defined thresholds
    if confidence_mlxtend >= (min_confidence / 100) and support_mlxtend >= (min_support / 100) * transaction_num:
        # Generate and append the association rule to the final result (mlxtend)
        final_association_rules_mlxtend += write_association_rules(item2_mlxtend, item1_mlxtend, confidence_mlxtend, support_mlxtend, transaction_num, rule_number_mlxtend)
        # Increment the rule number for the next association rule
        rule_number_mlxtend += 1

end_time = time.time() ## Ending the time calc here
elapsed_time = end_time - start_time

# Print the final association rules from mlxtend
print("Final Association Rules (Apriori): \n")
print(final_association_rules_mlxtend)
print(f"\nProgram execution time: {elapsed_time:.4f} seconds")

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

start_time = time.time() ## Starting the time calc here

# Use TransactionEncoder for one-hot encoding
te = TransactionEncoder()
one_hot_df = te.fit_transform(dataset)
one_hot_df = pd.DataFrame(one_hot_df, columns=te.columns_)

# Find frequent itemsets using FPGrowth
frequent_itemsets_mlxtend = fpgrowth(one_hot_df, min_support=min_support / 100, use_colnames=True)
print(f"\nFrequent Itemsets using FPGrowth:")
print(frequent_itemsets_mlxtend)

# Generate association rules
rules_mlxtend = association_rules(frequent_itemsets_mlxtend, metric="confidence", min_threshold=min_confidence / 100)

end_time = time.time() ## Ending the time calc here
elapsed_time = end_time - start_time

print("\nAssociation Rules using FPGrowth:")
print(rules_mlxtend)
print(f"\nProgram execution time: {elapsed_time:.4f} seconds") 









