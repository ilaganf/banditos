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
import csv

INPUT_FILENAME = '../data/warfarin.csv'
OUTPUT_FILENAME = "../data/warfarin_clean6.csv"

def main(ignore_missing_data=False):
    # Only works with float or integer values
    def get_val_or_mean(val, data, key):
        if val and val != "NA":
            return val
        else:
            #return "NA"
            return np.nanmean(data[key].values)

    data = pd.read_csv('../data/warfarin_clean4.csv')

    new_row_names = ["Age", 'Weight (kg)', 'Height (cm)', "Asian", "Black or African American", 'Unknown Race', "Mixed race", 'Therapeutic Dose of Warfarin',
                     "Unknown or mixed race", "Amiodarone (Cordarone)", "Enzyme inducer status"]
    old_row_names = ["Gender", "Race", "Ethnicity","Age","Height (cm)","Weight (kg)","Indication for Warfarin Treatment"]

    age_dict = {'10 - 19': 1, '20 - 29': 2, '30 - 39': 3, '40 - 49': 4, \
                            '50 - 59':5, '60 - 69':6, '70 - 79':7, '80 - 89':8, '90+':9}

    with open(INPUT_FILENAME, "r") as ifile:
        csv_reader = csv.DictReader(ifile, delimiter=",")
        with open(OUTPUT_FILENAME, "w") as ofile:
            ofile.write(",".join(new_row_names) + "\n")
            for row in csv_reader:
                if ignore_missing_data:
                    is_missing_val = False
                    for col_name in ["Age", "Race", "Ethnicity", "Height (cm)", 'Weight (kg)', "Therapeutic Dose of Warfarin",
                                         "Amiodarone (Cordarone)", "Medications", "Rifampin or Rifampicin"]:
                        val = row[col_name]
                        if val == "NA" or val == "":
                            is_missing_val = True
                    if is_missing_val:
                        continue

                new_values = []
                age = row["Age"]
                converted_age = get_val_or_mean(age, data, "Age") if age not in age_dict else age_dict[age]
                new_values.append(converted_age)

                new_values.append(get_val_or_mean(row["Weight (kg)"], data, "Weight (kg)"))
                new_values.append(get_val_or_mean(row['Height (cm)'], data, 'Height (cm)'))

                race = row["Race"]
                new_values.append(int(race == "Asian"))
                new_values.append(int(race == "Black or African American"))
                new_values.append(int(race == "Unknown"))


                ethnicity = row["Ethnicity"]
                new_values.append(int(race != "Unknown" and ethnicity == "Hispanic or Latino"))

                dose = row["Therapeutic Dose of Warfarin"]
                new_values.append(get_val_or_mean(dose, data, "Therapeutic Dose of Warfarin"))

                # Unknown or mixed race
                new_values.append(int(race == "Unknown"))   # No mention of mixed race

                # Amiodarone status
                status = row["Amiodarone (Cordarone)"]
                new_values.append(status if status and status != "NA" else 0)

                # Enzyme inducer status
                meds = row["Medications"]
                inducer_status = 0
                for med in ["carbamazepine", "phenytoin"]:
                    if med in meds and ("not " + med) not in meds:
                        inducer_status = 1
                rif = row["Rifampin or Rifampicin"]
                if rif != "NA" and rif != "0":
                    inducer_status = 1
                new_values.append(inducer_status)

                new_values = [str(val) for val in new_values]
                ofile.write(",".join(new_values) + "\n")


if __name__ == '__main__':
    main()