from datamodule.CLModule import CLDataModule
import torch
import numpy as np
from typing import List
import random

class PairingBatchSamplerV2(torch.utils.data.Sampler[List[int]]):
    def __init__(self, dataset, batch_size, shuffle, classes, 
        main_class_split:float, 
    ):
        self.main_class_split = main_class_split
        self.batch_size = batch_size
        self.classes = classes

        targets = None
        if isinstance(dataset, torch.utils.data.Subset):
            targets = np.take(dataset.dataset.targets, dataset.indices).tolist()
        else:
            targets = dataset.targets

        self._check_cl_numb(targets=targets, classes=classes)
        indices = self._get_indices_by_class(targets)
        indices = self._split_batch_main_indices(indices)
        indices_main, indices_other = self._split_batch_main_frequency_indices(indices)
        indices_other = self._shuffle_other(indices_other)
        batch = self._concat_batch(indices_main, indices_other, shuffle)
        self.batch_indices = self._serve_indices(batch, shuffle)

        # number of indices
        #print(torch.amax(torch.tensor(self.batch_indices), dim=(0, 1))) 
    def _check_cl_numb(self, targets, classes):
        u = np.unique(targets)
        cl_size = len(u)
        if(cl_size < len(classes)):
            raise Exception(f'Dataset class count {cl_size} - {u} is smaller than classes count {len(classes)} - {classes}.')

    def _get_indices_by_class(self, targets):
        batched_class_indices = {}

        for cl in self.classes:
            cl_indices = np.isin(np.array(targets), [cl])
            cl_indices_list = np.where(cl_indices)[0].tolist()

            batched_class_indices[cl] = cl_indices_list
        return batched_class_indices

    def _split_batch_main_indices(self, batched_class_indices):
        splitted_batch_indices = {}
        for key, val in batched_class_indices.items():
            numb_of_batches = int(len(val) // self.batch_size)
            splitted_batch_indices[key] = np.array_split(val, numb_of_batches)
        return splitted_batch_indices

    def _split_batch_main_frequency_indices(self, splitted_batch_indices):
        splitted_main = {}
        splitted_other = {}
        sizeof_main_cl_in_batch = int(np.floor(self.batch_size * self.main_class_split))
        for key, val in splitted_batch_indices.items():
            splitted_main[key] = []
            splitted_other[key] = []
            for batch in val:
                splitted_main[key].append(batch[:sizeof_main_cl_in_batch])
                # some indices may be lost here. 
                # moreover size may be zero
                splitted_other[key].append(batch[sizeof_main_cl_in_batch:self.batch_size])

            splitted_main[key] = np.stack(splitted_main[key], axis=0)
            splitted_other[key] = np.stack(splitted_other[key], axis=0)

        return splitted_main, splitted_other

    def _shuffle_other(self, splitted_other):
        other = []
        other_shuffled = {}
        sizeof_other_cl_in_batch = self.batch_size - int(np.floor(self.batch_size * self.main_class_split))
        for key, val in splitted_other.items():
            if(len(val[0]) == 0):
                return splitted_other
            other.append(np.reshape(val, -1))
        other = np.concatenate(other)
        np.random.shuffle(other)

        other = np.reshape(other, (len(splitted_other), -1, sizeof_other_cl_in_batch))

        for idx, (key, val) in enumerate(splitted_other.items()):
            other_shuffled[key] = other[idx]

        return other_shuffled

    def _concat_batch(self, main_indices, other_indices, shuffle_batch:bool):
        result = {}
        for key, val in main_indices.items():
            result[key] = []
            for batch_main, batch_other in zip(val, other_indices[key]):
                conc = np.concatenate((batch_main, batch_other))
                if(shuffle_batch):
                    np.random.shuffle(conc)
                result[key].append(conc)
                
            result[key] = np.stack(result[key])

        return result

    def _serve_indices(self, indices_dict: dict, shuffle:bool):
        indices_batch_array = []
        for val in indices_dict.values():
            indices_batch_array.append(val)
        indices_batch_array = np.concatenate(indices_batch_array)
        if(shuffle):
            np.random.shuffle(indices_batch_array)
        return indices_batch_array

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)
        

class PairingBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
        For the dataset with a subset of classes.
        DO NOT use this with BatchSampler to return a batch of indices.
        Use this as a argument for batch_sampler.
        It splits the batch into 2 groups: main and rest. The main part has an indices of majority / main class 
        and rest part has indices from any other classes except the main class.
        You can choose the size of the main part and the frequency of the appearance of
        the batches of specified main class. For example you can choose that class 0 will be a majority class
        in batches in 20% of cases.

        If the interface is incompatible, then use lambda as adapter.
        This class operates only on its own dataset indices, where it points to the corresponding data.
    """
    def __init__(self, dataset, batch_size, shuffle, classes, 
        main_class_split:float, 
        classes_frequency:list[float],
        try_at_least_x_main_batches=2,
        try_to_full_fill_up=False,
        min_dataset_size=4,
    ):
        """
            main_class_split: 0.55 for batch size = 32 will mean that 18 samples will be from majority class 
                and 14 will be from the rest.
                Note: main_class_split may be less than 0.5, then the main class will not be for the most part
                and if there are only 2 classes, then vocabulary will remain the same in this class but 
                another part of the framework may interpret this differently.
            classes_frequency: length of the list must be the same size as the number of classes.
            try_at_least_x_main_batches: at least this many batches will represent every class. 
                Class is represented by batch by their main part. 
            try_to_full_fill_up: try to use all avaliable data to create all possible batches. It means 
                that some borders like classes_frequency may be slightly affected. For example if in summary exist
                only 40 unique items, then create two batches of size 32 (default size) and 8.
            min_dataset_size: minimum dataset size that is a multiple of the batch size for each class
        """
        #super().__init__() # optional
        self.main_class_split = main_class_split
        self.batch_size = batch_size
        self.classes = classes
        self.classes_frequency = classes_frequency
        self.try_to_full_fill_up = try_to_full_fill_up

        if len(self.classes) != len(classes_frequency):
            raise Exception(f"Wrong number of classes and classes_frequency:"
            f"{len(self.classes)} -- {len(classes_frequency)}. They must be equal")

        targets = None
        if isinstance(dataset, torch.utils.data.Subset):
            targets = np.take(dataset.dataset.targets, dataset.indices).tolist()
        else:
            targets = dataset.targets
        
        for cl in self.classes:
            if cl not in targets:
                raise Exception(f"In target of this dataset found no class of label: {cl}. " +
                f"Target has classes of labels {set(targets)}")

        boundary = min_dataset_size * self.batch_size * len(self.classes_frequency)
        if(len(dataset) <= boundary):
            raise Exception(f"Size of the dataset too low. Not enough items to create datasampler." +
                f"Length of dataset: {len(dataset)} must be greater than: {boundary}")

        number_of_idxs_in_main_class = int(np.ceil(batch_size * main_class_split)) # or np.floor

        batched_class_indices = self.__split_to_main_classes(
            targets=targets, 
            number_of_idxs_in_main_class=number_of_idxs_in_main_class
        )

        real_number_of_batches = len(targets) // self.batch_size

        mixed_targets_indices, \
        batched_classes_main_indices, \
        batched_classes_other_indices = self.__change_main_classes_quantity(
            classes_frequency=classes_frequency, 
            batched_class_indices=batched_class_indices, 
            real_number_of_batches=real_number_of_batches,
            try_at_least_x_main_batches=try_at_least_x_main_batches
        )
  
        batched_classes_main_indices, data_buffer = self.__fill_up_batches(
            batched_classes_main_indices=batched_classes_main_indices, 
            batched_classes_other_indices=batched_classes_other_indices,
            mixed_targets_indices=mixed_targets_indices,
            targets=targets, 
        )

        # remove dimention of the class and shuffle
        self.batches_sequences = []
        for batch in batched_classes_main_indices.values():
            if shuffle:
                random.shuffle(batch)
            self.batches_sequences.extend(batch)

        for idx, b in enumerate(self.batches_sequences):
            if(len(b) != self.batch_size):
                raise Exception(f"Batch {idx} does not have required size of {self.batch_size}. It has length of {len(b)}")

        # here batch can be the size different than self.batch_size
        #data_buffer.print()
        #print(len(self.batches_sequences))
        if(self.try_to_full_fill_up):
            reused = self.__reuse_unused(data_buffer)
            self.batches_sequences.extend(reused)
        #data_buffer.print()
        #print(len(self.batches_sequences))

        if shuffle:
            random.shuffle(self.batches_sequences)

    def __split_to_main_classes(self, targets, number_of_idxs_in_main_class):
        """
            Convert the class label into sequence of the data indices belonging to this class.
            Split class indices into batches of size number_of_idxs_in_main_class.
            Later these batches should be expanded to the size of batch_size.
            Return dictionary like -- classes: [batches[indices]]
        """
        # and split them into batches of size number_of_idxs_in_main_class
        batched_class_indices = {}

        for cl in self.classes:
            cl_indices = np.isin(np.array(targets), [cl])
            cl_indices_list = np.where(cl_indices)[0].tolist()

            # add everything from this class to one batch (only one batch of this class can be created)
            if(number_of_idxs_in_main_class > len(cl_indices_list)):
                batched_class_indices[cl] = [cl_indices_list]

            else: # split by the size of the main part. The other part will be added later.
                numb_of_batches = int(len(cl_indices_list) // number_of_idxs_in_main_class)
                split_cl_indices_list = []
                for split_array in np.array_split(cl_indices_list, numb_of_batches): 
                    split_cl_indices_list.append(split_array.tolist()) # must be like that to cast numpy array to list
                batched_class_indices[cl] = split_cl_indices_list
        return batched_class_indices

    def get_nth_key(dictionary, n=0):
        if n < 0:
            n += len(dictionary)
        for i, key in enumerate(dictionary.keys()):
            if i == n:
                return key
        raise IndexError("dictionary index out of range") 

    def __change_main_classes_quantity(self, 
            classes_frequency, 
            batched_class_indices, 
            real_number_of_batches,
            try_at_least_x_main_batches
        ):
        """
            From the class frequency decide how many batches will be used from this class.
            Rest of the batches are set to be used as fillers to other classes.
            Return
                mixed_targets_indices - list of unused (not main) indices
                batched_classes_main_indices - dictionary of main part like -- classes: [batches[indices]]
                batched_classes_other_indices - dictionary of the rest part of the class like -- classes: [batches[indices]]
        """
        mixed_targets_indices = []
        batched_classes_main_indices = {}
        batched_classes_other_indices = {}
        remaining_number_of_batches = real_number_of_batches
        for freq, (classs, batch) in zip(classes_frequency, batched_class_indices.items()):
            numb_of_main_batches = int(np.floor(freq * real_number_of_batches))
            remaining_number_of_batches -= numb_of_main_batches

            if(numb_of_main_batches < try_at_least_x_main_batches):
                numb_of_main_batches = try_at_least_x_main_batches

            batched_classes_main_indices[classs] = batch[:numb_of_main_batches]
            
            other = batch[numb_of_main_batches:]
            batched_classes_other_indices[classs] = other
            mixed_targets_indices.extend(np.reshape(other, -1)) # add ony those that wont be used as main

        return mixed_targets_indices, batched_classes_main_indices, batched_classes_other_indices

    def __flatten_list(self, l) -> list:
        if(isinstance(l, list)):
            if(len(l) != 0 and isinstance(l[-1], list)):
                newl = [item for sublist in l for item in sublist]
                return self.__flatten_list(newl)
        return l

    def concatenate_dict(self, buffers: dict, exclude):
        ret = []
        for classs, buffer in buffers.items():
            if(classs != exclude):
                b = self.__flatten_list(buffer)
                ret.extend(b)
        return ret

    def __fill_up_batches(self,
            batched_classes_main_indices, 
            batched_classes_other_indices,
            mixed_targets_indices,
            targets, 
        ):
        """
            Fill the main batches to the size of batch_size, where the filler of the main batch
            does not have any indices of current class.
            It is the same as removing partial batches from 'mixed_targets_indices' that are already used in
            'batched_classes_main_indices' or 'batched_classes_other_indices'.
            Return dictionary like -- classes: [batches[indices]]
        """
        data_buffer = SharedData()
        
        random.shuffle(mixed_targets_indices) # needs to be to prevent from forming a cycle of the size less than len(classes)
        for (classs, batch_main), batch_other in zip(
                batched_classes_main_indices.items(), batched_classes_other_indices.values()
            ):

            #complement_current_main = self.concatenate_dict(batched_classes_other_indices, classs)
            complement_current_main = self.__flatten_list(batch_main) 
            complement_current_main += self.__flatten_list(batch_other)

            # get all possible items for this class
            complement_current_main = np.setdiff1d(mixed_targets_indices, complement_current_main) 

            data_buffer.add(classs, complement_current_main)
            #self.__show_to_what_class_belongs_to(targets, complement_current_main)
            #print(classs, len(complement_current_main), len(self.__flatten_list(batch_main)), len(self.__flatten_list(batch_other)), len(self.__flatten_list(batch_main)) + len(self.__flatten_list(batch_other)))

        for classs, batch_main in batched_classes_main_indices.items():

            buffer_used_idxs = []
            for idx, indices in enumerate(batch_main):
                diff = self.batch_size - len(indices)

                complement = []
                for _ in range(diff):
                    complement.append(data_buffer.popReversed(classs))
                indices.extend(complement) # extend to full batch

        return batched_classes_main_indices, data_buffer

    def __show_to_what_class_belongs_to(self, targets, buffer):
        for cl in self.classes:
            cl_indices = np.isin(np.array(targets), [cl])
            cl_indices_list = np.where(cl_indices)[0].tolist()
            b = np.reshape(buffer, -1)
            r = np.setdiff1d(b, np.reshape(cl_indices_list, -1))

            string = ''
            if(len(buffer) == len(r)):
                string += ' fully not belongs to;'
            if(len(r) == 0):
                string += ' fully belongs to;'

            print(len(r), cl, string)

    def __reuse_unused(self, data_buffer):
        rest = data_buffer.dumpAllUnique()
        batch = []
        while len(rest) != 0:
            batch.append(rest[:self.batch_size])
            rest = rest[self.batch_size:]
        return batch

    def __iter__(self):
        return iter(self.batches_sequences)

    def __len__(self):
        return len(self.batches_sequences)
        
    def __decrease_array(array, by_amount):
        if len(array) == 0:
            return array
        array =  array[:by_amount]

        return array

class SharedData():
    """.git/
        Class that treats items in buffers as shared. Removing from one buffer can remove from others
        if selected item exist in them.
    """
    def __init__(self):
        self.possible_buffer = {}
        self.internal_counter = 0

    def size(self, key):
        return len(self.possible_buffer[key])

    def fullSize(self):
        s = 0
        for v in self.possible_buffer.values():
            s += len(v)
        return s

    def add(self, key, possible_buffer: list):
        if(isinstance(possible_buffer, list)):
            self.possible_buffer[key] = possible_buffer
        elif(isinstance(possible_buffer, np.ndarray)):
            self.possible_buffer[key] = possible_buffer.tolist()

    def pop(self, key, rand=True):
        '''
            Return item assigned to the key.
        '''
        if(len(self.possible_buffer[key]) == 0):
            raise Exception(f"Cannot pop value. Buffer for key {key} is empty (len = 0).")
        if(rand):
            pos = random.randint(0, len(self.possible_buffer[key]) - 1)
        else:
            pos = 0
        index = self.possible_buffer[key][pos]
        self.possible_buffer[key].remove(index)
        for v in self.possible_buffer.values():
            if(index in v):
                v.remove(index)
        return index

    def popReversed(self, key, rand=True):
        '''
            Return from any buffer except from buffer with 'key'.
        '''
        if(len(self.possible_buffer) == 1):
            raise Exception("Only one buffer exist. Cannot get reversed item")

        keys = list(self.possible_buffer.keys())
        not_loops = True
        if(rand):
            while sum([len(x) for x in list(self.possible_buffer.values())]) != 0:
                not_loops = False
                mykey = keys[random.randint(0, len(keys) - 1)]
                if(mykey != key):
                    b = self.possible_buffer[mykey]
                    if(len(b) == 0):
                        continue
                    pos = random.randint(0, len(b) - 1)
                    break
        else:
            while sum([x for x in list(self.possible_buffer.values())]) != 0:
                not_loops = False
                mykey = keys[self.internal_counter]
                pos = 0
                self.internal_counter += 1
                if self.internal_counter >= len(keys):
                    self.internal_counter = 0
                if(len(mykey) == 0):
                    continue
                if(mykey != key):
                    break
            
        if(not_loops):
            raise Exception("Error: no more items left in any buffer.")
        index = self.possible_buffer[mykey][pos]

        # remove item from all buffers
        for v in self.possible_buffer.values():
            if(index in v):
                v.remove(index)
        return index

    def print(self):
        s = 0
        unique = set()
        for k, v in self.possible_buffer.items():
            print(f"Key: {k} -- Size: {len(v)}")
            s += len(v)
            unique = unique.union(v)
        print(f"In total: {s}")
        print(f"In total unique: {len(unique)}")

    def dumpAllUnique(self):
        unique = set()
        for k, v in self.possible_buffer.items():
            unique = unique.union(v)
            self.possible_buffer[k] = []
        return list(unique)


class PairingSampler(torch.utils.data.Sampler):
    """
        TODO not finished
        For the dataset that has all classes.
    """
    
    def __init__(self, dataset, batch_size, main_class_split:float, classes:list, task_split:list):
        super().__init__()
        self.main_class_split = main_class_split
        self.data_size = len(dataset)
        self.batch_size = batch_size
        self.classes = classes

        class_indices = []
        for cl in classes:
            cl_indices = np.isin(np.array(dataset.targets), [cl])
            cl_indices_list = np.where(cl_indices)[0]
            class_indices.append(cl_indices_list)

        for t_split in task_split:
            classes_per_task = [class_indices[cl] for cl in t_split]

        

        self.sequence = list(range(dataSize))[startIndex * batchSize:]
        random.Random(seed).shuffle(self.sequence)

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

class PairingDataModule(CLDataModule):
    """
        TODO not finished
        Operates on datasets and splits them into datasets tasks.
        The dataset task is a set, where first part of size main_class_split of the data 
        refers to a task with specified index and the second part consist of
        mixed up tasks.
    """
    def __init__(self, 
        main_class_split=0.5,
        classes: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.main_class_split = main_class_split
        self.classes = classes

    def setup_tasks(self):
        train_datasets = split_by_class_dataset(self.train_dataset, classes)
        test_datasets = split_by_class_dataset(self.test_dataset, classes)

        self.train_datasets = pair_sets(train_datasets, self.main_class_split)
        self.test_datasets = pair_sets(test_dataset, self.main_class_split)


