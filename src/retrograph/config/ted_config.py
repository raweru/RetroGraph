"""
Configuration constants for Tree Edit Distance (TED) scoring.

These values define the costs for deletion, insertion, and renaming operations
when comparing synthesis trees using the APTED algorithm. Renaming costs
depend on node type and similarity of reaction classifications (based on namerxn).
"""

# Cost for deleting a node
COST_DELETE = 1.0

# Cost for inserting a node
COST_INSERT = 1.0

# Cost for renaming nodes of different types (e.g., 'mol' vs 'reaction')
COST_RENAME_TYPE_MISMATCH = 1.0


# CLASSIFICATION_AWARE specific costs:

# Cost for comparing reaction nodes with different classification lengths.
# Primarily penalizes comparisons between recognized reaction classes that have 3 levels 
# of classification (e.g., '1.2.3') and 'Unrecognized' reactions (e.g., '0.0').
COST_RENAME_REACTION_LEN_MISMATCH = 1.0

# Cost for renaming reaction nodes where the first classification number differs
# (e.g., "1.2.3" vs "2.2.3")
COST_RENAME_REACTION_CLASS_1_DIFF = 1.0

# Cost for renaming reaction nodes where the second classification number differs
# (e.g., "1.2.3" vs "1.3.3")
COST_RENAME_REACTION_CLASS_2_DIFF = 0.5

# Cost for renaming reaction nodes where only the third classification number differs
# (e.g., "1.2.3" vs "1.2.4") - considered very similar
COST_RENAME_REACTION_CLASS_3_DIFF = 0.0