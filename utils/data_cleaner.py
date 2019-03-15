'''
data_cleaner.py

cleans the warfarin data and writes it to a new csv that can just 
be easily loaded later
'''

'''
Observations about the data:
- "Ethnicities" only takes values "not Hispanic or Latino" and "Hispanic or Latino" (and unknown)
        - can be converted to 0 or 1
- "Race" is one of 'Asian', 'Black or African American', 'White', or 'Unknown'
- Can drop comments on the data (under 'unknown')
- Comorbidities are as strings and listed separated by semicolons
- Indication for Warfarin Treatment is the same (numerical codes)
- Medications is the same (strings)
        - hypothetically, we could just create indicators for each of the medications
          present and it would capture all the info from the other medication columns
'''

import pandas as pd
import numpy as np

FILENAME = '../data/warfarin.csv'

# general columns that can be safely dropped
DROP = ['Unknown', 'PharmGKB', 'Project Site', \
        'statin', 'Amiodarone', "Carbamazepine",\
        'Phenytoin', 'Rifampin', 'Antibiotics',\
        'Azoles', 'Supplements']

SEMICOLONS = ['Comorbidities', 'Indication for Warfarin Treatment',\
              'Medications']

def gather_categories(series, default_val):
    '''
    Gets the category names for non-numerical, semicolon-separated entries
    '''
    categories = set()
    categories.add(default_val)
    for entry in series:
        if type(entry) is float: 
            continue 
        indiv_entries = entry.strip().split(';')
        for x in indiv_entries:
            categories.add(x.strip().lower())
    return list(categories)

def fill_indicators(df, categories):
    pass

    
def main():
    raw = pd.read_csv(FILENAME)
    for colname in raw.columns:
        for col in DROP:
            if col in colname:
                raw.drop(colname, axis=1, inplace=True)



if __name__ == '__main__':
    main()