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
        It divide the batch into 2 groups: main and rest. The main part has an indices of majority / main class 
        and rest part has indices from any other classes except the main class.
        You can choose the size of the main part and the frequency of the appearance of
        the batches of specified main class. For example you can choose that class 0 will be a majority class
        in batches in 20% of cases.

        If the interface is incompatible, then use lambda as adapter.
    """
    def __init__(self, dataset, batch_size, shuffle, classes, 
        main_class_split:float, 
        classes_frequency:list[float],
        try_at_least_x_main_batches=2
    ):
        """
            main_class_split: 0.55 for batch size = 32 will mean that 18 samples will be from majority class 
                and 14 will be from rest.
                Note: main_class_split may be less than 0.5, then the main class will not be for the most part
                and if there are only 2 classes, then vocabulary will remain the same in this class but 
                another part of the framework may interpret this differently.
            classes_frequency: length of the list must be the same size as the number of classes.
            try_at_least_x_main_batches: at least this many batches will represent every class. 
                Class is represented by batch by their main part. 
        """
        #super().__init__() # optional
        self.main_class_split = main_class_split
        self.batch_size = batch_size
        self.classes = classes
        self.classes_frequency = classes_frequency

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

        number_of_main_class_idx_in_batch = int(np.ceil(batch_size * main_class_split)) # or np.floor

        #print("-------------------------------")
        #print(targets)
        #print(len(targets))
        #print(number_of_main_class_idx_in_batch)

        batched_class_indices = self.__split_to_main_classes(
            targets=targets, 
            number_of_main_class_idx_in_batch=number_of_main_class_idx_in_batch
        )

        #print("-------------------------------")
        #print(sum_number_of_batches)
        #print(len(batched_class_indices))
        #print(batched_class_indices)

        real_number_of_batches = len(targets) // self.batch_size

        all_targets_indices, \
        correct_batched_classes_main_indices, \
        correct_batched_classes_rest_main_indices = self.__change_main_classes_quantity(
            classes_frequency=classes_frequency, 
            batched_class_indices=batched_class_indices, 
            real_number_of_batches=real_number_of_batches,
            try_at_least_x_main_batches=try_at_least_x_main_batches
        )
  
        correct_batched_classes_main_indices = self.__fill_up_batches(
            correct_batched_classes_main_indices=correct_batched_classes_main_indices, 
            correct_batched_classes_rest_main_indices=correct_batched_classes_rest_main_indices,
            all_targets_indices=all_targets_indices
        )

        # remove dimention of the class and shuffle
        self.batches_sequences = []
        for classes in correct_batched_classes_main_indices:
            for batch in classes:
                if shuffle:
                    random.shuffle(batch)
                self.batches_sequences.append(batch)

        if shuffle:
            random.shuffle(self.batches_sequences)

        for idx, b in enumerate(self.batches_sequences):
            if(len(b) != self.batch_size):
                raise Exception(f"Batch {idx} does not have required size of {self.batch_size}. It has length of {len(b)}")
        
        #print("---------------------------------")
        #print(correct_batched_classes_main_indices)
        #print(self.batches_sequences)
        #exit()

    def __split_to_main_classes(self, targets, number_of_main_class_idx_in_batch):
        """
            Convert the class label into sequence of the data indices belonging to this class.
            Split class indices into batches of size number_of_main_class_idx_in_batch.
            Later thise batches should be expanded to the size of batch_size.
            Return list like <classes<batches<indices>>>
        """
        # 
        # and split them into batches of size number_of_main_class_idx_in_batch
        batched_class_indices = []
        #sum_number_of_batches = 0

        for cl in self.classes:
            cl_indices = np.isin(np.array(targets), [cl])
            cl_indices_list = np.where(cl_indices)[0].tolist()
            # add everything from this class to one batch
            if(number_of_main_class_idx_in_batch > len(cl_indices_list)):
                batched_class_indices.append([cl_indices_list])
                #sum_number_of_batches += len(cl_indices_list)
            else:# split by the size of the main part. The rest part will be added later.
                numb_of_batches = int(len(cl_indices_list) // number_of_main_class_idx_in_batch)
                split_cl_indices_list = []
                for split_array in np.array_split(cl_indices_list, numb_of_batches): 
                    # must be like that to cast numpy array to list
                    split_cl_indices_list.append(split_array.tolist())
                batched_class_indices.append(split_cl_indices_list)
                #sum_number_of_batches += numb_of_batches
        return batched_class_indices

    def __change_main_classes_quantity(self, 
            classes_frequency, 
            batched_class_indices, 
            real_number_of_batches,
            try_at_least_x_main_batches
        ):
        """
            From the class frequency decide how many batches will be used from this class.
            Rest of the batches are set to be used as fillers to other classes.
            Returns list like <classes<batches<indices>>>
        """
        all_targets_indices = []
        correct_batched_classes_main_indices = []
        correct_batched_classes_rest_main_indices = []
        for freq, class_batch in zip(classes_frequency, batched_class_indices):
            numb_of_batches = int(np.floor(freq * real_number_of_batches))
            if(numb_of_batches < try_at_least_x_main_batches):
                numb_of_batches = try_at_least_x_main_batches
            correct_batched_classes_main_indices.append(class_batch[:numb_of_batches])
            
            tmp = class_batch[numb_of_batches:]
            #print("----------------")
            #print(numb_of_batches)
            #print(len(class_batch))
            #print(real_number_of_batches)
            #print(freq)
            correct_batched_classes_rest_main_indices.append(tmp)
            all_targets_indices.extend(tmp) # add ony those that wont be used as main
        return all_targets_indices, correct_batched_classes_main_indices, correct_batched_classes_rest_main_indices

    def __fill_up_batches(self,
            correct_batched_classes_main_indices, 
            correct_batched_classes_rest_main_indices,
            all_targets_indices
        ):
        """
            Fill the main batches to the size of batch_size, where the filler of the main batch
            do not has any indices of current class.
            Returns list like <classes<batches<indices>>>
        """
        for batched_main_indices, batched_rest_main_indices in zip(
                correct_batched_classes_main_indices, correct_batched_classes_rest_main_indices
            ):
            if len(batched_main_indices) == 0:
                #print("No batch")
                continue

            #print("----------------")
            #print(all_targets_indices)
            #print(np.reshape(batched_main_indices, [-1]))
            #print("aa", batched_main_indices)
            #print(np.reshape(batched_rest_main_indices, [-1]))
            complement_main = np.setdiff1d(
                np.setdiff1d(all_targets_indices, np.reshape(batched_main_indices, [-1])),
                    np.reshape(batched_rest_main_indices, [-1]))

            
            #list(all_targets_indices - 
            #    set(np.reshape(batched_main_indices, [-1])) - 
            #    set(np.reshape(batched_rest_main_indices, [-1]))) # remove all indices of main class
            #print("complement", complement_main)
            #print("main", batched_main_indices)
            #print("rest", batched_rest_main_indices)

            #print(batched_main_indices)
            #print(complement_main)
            #print(all_targets_indices)
            #print(batched_main_indices)
            #print(batched_rest_main_indices)
            buffer_used = []
            for batch_indices in batched_main_indices:
                diff = self.batch_size - len(batch_indices)
                batch_indices.extend(complement_main[:diff]) # extend to full batch
                buffer_used.extend(batch_indices) # add full batch to used indices
                complement_main = complement_main[diff:] # remove used indices

            #print(buffer_used)
            #exit()
            all_targets_indices = np.setdiff1d(all_targets_indices, buffer_used) # remove indices used in batch
        return correct_batched_classes_main_indices

    def __iter__(self):
        return iter(self.batches_sequences)

    def __len__(self):
        return len(self.batches_sequences)
        
    def __decrease_array(array, by_amount):
        if len(array) == 0:
            return array
        array =  array[:by_amount]

        return array

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


