from zipfile import ZipFile
import os
from tqdm import tqdm 
for i in tqdm(os.listdir('/group/jug/Aman/Omniglot_data/')):
    with ZipFile(f'/group/jug/Aman/Omniglot_data/{i}', 'r') as zip_ref:
        zip_ref.extractall('/group/jug/Aman/Omniglot/')