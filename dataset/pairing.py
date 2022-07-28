from dataset.CLModule import CLDataModule
import torch
import numpy as np
from typing import List
import random

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
        try_to_full_fill_up=True,
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
        if(isinstance(l, list) and isinstance(l[-1], list)):
            newl = [item for sublist in l for item in sublist]
            return self.__flatten_list(newl)
        else:
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
            complement_current_main = self.__flatten_list(batch_main) + (self.__flatten_list(batch_other))

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
            raise Exception("Error: no more items in any buffer left.")
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


