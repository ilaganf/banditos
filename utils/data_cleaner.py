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
NEW_FILENAME = '../data/warfarin_clean.csv'

# general columns that can be safely dropped
DROP = ['Unknown', 'PharmGKB', 'Project Site', \
        'statin', 'Amiodarone', "Carbamazepine",\
        'Phenytoin', 'Rifampin', 'Antibiotics',\
        'Azoles', 'Supplements', 'Unnamed','`']

# categories that are semicolon-separated, plus indicator prefix
SEMICOLONS = [('Comorbidities', 'comorb: '), ('Indication for Warfarin Treatment', 'indication: '),\
              ('Medications', 'med: ')]

def gather_categories(series, prefix):
    '''
    Gets the category names for non-numerical, semicolon-separated entries
    then creates a dataframe with the appropriate indicators filled out
    '''
    categories = set()
    # gathers all the unique categories to be turned into column headers
    for entry in series:
        if type(entry) is float: 
            continue 
        indiv_entries = entry.strip().split(';')
        for x in indiv_entries:
            categories.add(x.strip().lower())

    df_dict = {prefix + cat:[] for cat in categories}
    for val in series:
        is_float = type(val) is float
        for cat in categories:
            entry = 0 if is_float else int(cat in val.strip().lower())
            df_dict[prefix + cat].append(entry)
    return pd.DataFrame(df_dict)


def main():
    raw = pd.read_csv(FILENAME)

    # drop useless columns
    for colname in raw.columns:
        for col in DROP:
            if col in colname:
                raw.drop(colname, axis=1, inplace=True)

    # create clean indicator columns for semicolon-separated columns
    for col, prefix in SEMICOLONS:
        new_df = gather_categories(raw[col], prefix)
        raw.drop(col, axis=1, inplace=True)
        raw = pd.concat([raw, new_df], axis=1)

    # condense age
    age_dict = {'10 - 19':1, '20 - 29':2, '30 - 39':3, '40 - 49':4, \
                '50 - 59':5, '60 - 69':6, '70 - 79':7, '80 - 89':8, \
                '90+':9}
    age_vals = []
    for val in raw['Age']:    
        new_val = 0 if val not in age_dict else age_dict[val]
        age_vals.append(new_val)
    raw['Age'] = age_vals

    # condense race and ethnicity
    raw['Ethnicity'] = (raw['Ethnicity'] == 'Hispanic or Latino').astype(int)
    races = ['Asian', 'Black or African American', 'White']
    race_dict = {'Asian':[], 'Black or African American':[], 'White':[], 'Unknown Race':[]}
    for val in raw['Race']:
        if type(val) is float: # nan, should default to unknown
            for race in races:
                race_dict[race].append(0)
            race_dict['Unknown Race'].append(1)
        else:
            for race in races:
                race_dict[race].append(int(race in val))
            race_dict['Unknown Race'].append(int('Unknown' in val))
    raw.drop('Race', axis=1, inplace=True)
    raw = pd.concat([raw, pd.DataFrame(race_dict)], axis=1)

    # drop patients that don't have ground truth
    raw = raw[raw.loc[:,'Therapeutic Dose of Warfarin'].notna()]

    clean = pd.get_dummies(raw, prefix='indic')
    clean.fillna(0, inplace=True) # in case nas escaped earlier somehow
    clean.to_csv(NEW_FILENAME)

if __name__ == '__main__':
    main()