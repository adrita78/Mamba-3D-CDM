import msgpack
import os
import numpy as np
from types import SimpleNamespace
from multiprocessing import Pool

from torch.utils import data
from rdkit import Chem
import numpy as np
import glob, pickle, random
import os.path as osp
import torch, tqdm
import copy
from torch_geometric.data import Dataset, DataLoader
from utils.featurization import featurize_mol
from torch_geometric.transforms import BaseTransform
from collections import defaultdict
import math
from functools import cache

class MambaDataset(Dataset):
    def __init__(self, root, split_path, mode, types, dataset, cache = None , transform=None, num_workers=1, limit_molecules=None, max_confs= 10):
        super(MambaDataset, self).__init__(root, transform)

    """
    Part of the code taken from Torsional Diffusion https://github.com/gcorso/torsional-diffusion
    
    """
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.max_confs = max_confs

        if cache: cache+= "." + mode
        self.cache = cache
        if cache and os.path.exists(cache):
          print('Reusing preprocessing from cache', cache)
          with open(cache, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
          print("Preprocessing the dataset...")
          self.datapoints = self.preprocess_datapoints(root, split_path, mode, num_workers, limit_molecules)
          if cache:
            print('Saving preprocessing to cache', cache)
            with open(cache, "wb") as f:
                pickle.dump(self.datapoints, f)

        file_path = os.path.join(self.root, 'datapoints.pickle')  # Changed for clarity
        #print(f"Saving preprocessed datapoints to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(self.datapoints, f)

        if limit_molecules:
            self.datapoints = self.datapoints[:limit_molecules]
        if not self.datapoints:
            raise ValueError("No valid datapoints found. Check the dataset structure and preprocessing logic.")

    def preprocess_datapoints(self, root, split_path, mode, num_workers, limit_molecules):
        #print("Loading the split file...")
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        #print(f"Loaded {len(split)} indices for mode: {mode}.")

        if limit_molecules:
            split = split[:limit_molecules]

        smiles_files = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles_files = smiles_files[split]
        #print(f"Preparing to process {len(smiles_files)} SMILES files...")

        datapoints = []
        if num_workers > 1:
            #print(f"Using {num_workers} workers for multiprocessing...")
            p = Pool(num_workers)
            p.__enter__()

        with tqdm.tqdm(total=len(smiles_files)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for t in map_fn(self.filter_smiles, smiles_files):
                if t:
                    datapoints.append(t)
                pbar.update()

        if num_workers > 1:
            p.__exit__(None, None, None)

        #print(f"Fetched {len(datapoints)} molecules successfully.")
        #print(f"Failures summary: {dict(self.failures)}")
        return datapoints

    def filter_smiles(self, smile_file):
        #print(f"Processing file: {smile_file}...")
        if not os.path.exists(smile_file):
            #print(f"File not found: {smile_file}")
            self.failures['raw_pickle_not_found'] += 1
            return False

        try:
            mol_dic = self.open_pickle(smile_file)
            #print(f"Loaded pickle file. Keys: {list(mol_dic.keys())}")
        except Exception as e:
            #print(f"Error loading pickle file: {e}")
            self.failures['pickle_load_failed'] += 1
            return False

        smile = mol_dic['smiles']
        print(f"SMILES: {smile}")

        if '.' in smile:
            #print(f"Skipping SMILES with dot: {smile}")
            self.failures['dot_in_smile'] += 1
            return False

        mol = Chem.MolFromSmiles(smile)
        if not mol:
            #print(f"Failed to parse SMILES: {smile}")
            self.failures['mol_from_smiles_failed'] += 1
            return False

        conformers = mol_dic.get('conformers', [])
        if len(conformers) == 0:
            #print(f"No conformers found for SMILES: {smile}")
            self.failures['no_conformers'] += 1
            return False

        data = self.featurize_mol(mol_dic)
        if not data:
            #print(f"Featurization failed for SMILES: {smile}")
            self.failures['featurize_mol_failed'] += 1
            return False

        return data

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        return data

    def open_pickle(self, mol_path):
        #print(f"Opening pickle file: {mol_path}")
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        #print(f"Featurizing molecule...")
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)
        N = confs[0]['rd_mol'].GetNumAtoms()
        if N < 4:
            return None
        #print(f"Number of atoms in the molecule: {N}")
        #print(f"Max Confs: {self.max_confs}")
        pos = torch.zeros([self.max_confs, N, 3])
        pos_mask = torch.zeros(self.max_confs, dtype=torch.int64)
        k = 0
        for conf in confs:
            mol = conf['rd_mol']

            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                #print(f"Error sanitizing conformer: {e}")
                continue

            if conf_canonical_smi != canonical_smi:
                continue
            #print(f"Number of conformers in mol: {mol.GetNumConformers()}")
            k = 0
            for conformer_index in range(mol.GetNumConformers()):
              pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
              pos_mask[k] = 1
              k += 1
              correct_mol = mol
              if k == self.max_confs:
                break

        if k == 0:
            print(f"No valid conformers found for {canonical_smi}.")
            return None
        #pos_tensor = torch.vstack(pos) if len(pos) > 1 else pos[0]
        data = featurize_mol(correct_mol, self.types)
        data.canonical_smi, data.mol, data.pos = canonical_smi, correct_mol, pos[k]
        #print(f"Featurization successful. Molecule: {canonical_smi}")
        return data



def construct_loader(args, modes=('train', 'val')):
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    types = qm9_types if args.dataset == 'qm9' else drugs_types

    for mode in modes:
        dataset = MambaDataset(args.data_dir, args.split_path, mode, dataset=args.dataset,
                                   types=types, transform=transform,
                                   num_workers=args.num_workers,
                                   limit_molecules=args.limit_train_mols,
                                   cache=args.cache,
                                   )
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False if mode == 'test' else True,drop_last = False)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
