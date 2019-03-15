'''
Refined dataloader
'''

'''
Observations about the data:
- "Ethnicities" only takes values "not Hispanic or Latino" and "Hispanic or Latino" (and unknown)
        - can be converted to 0 or 1
- "Race" is one of 'Asian', 'Black or African American', 'White', or 'Unknown'
- Can drop comments on the data (under 'unknown')
- Comorbidities are as strings and listed separated by semicolons
'''

from DataLoader import DataLoader

class RefinedDL(DataLoader):
    '''
    Less lazy data loader that makes efforts to better clean and present the data
    '''
    def __init__(self, filename_csv, seed=420):
        raw = pd.read_csv(filename_csv)
        cleaned = self._clean(raw)

    def _clean(self, raw):
        for colname in raw.columns:
            pass
        pass