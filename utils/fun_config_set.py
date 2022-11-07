from select import select
from utils.functional import dream_objective
from utils.functional import select_task
from utils.functional import target_processing
from utils.functional import task_split

from typing import Union

from model.SAE import SAE_CIFAR
from model.vgg import vgg11_bn
from model.ResNet import ResNet18, Resnet20C100
from model.overlay import CLModelWithIslands, CLModel, CLModelIslandsTest

class FunConfigSetBase():
    # must be {main phrase: rule}
    SPECIFIC_COMBINATIONS = {
        'OBJECTIVE-SAE-STANDALONE-DIVERSITY': 'SAE',
        'OBJECTIVE-RESNET20-C100-DIVERSITY': 'RESNET20C100',
        'OBJECTIVE-RESNET20-C100-CHANNEL': 'RESNET20',
        'TARGET-LATENT-DECODE': 'CL-MODEL-ISLAND-TEST',
        'CL-MODEL': 'TARGET-DEFAULT',
        'TARGET-DEFAULT': 'CL-MODEL',
    }
    
    GET_MODEL = {
        'SAE': SAE_CIFAR,
        'VGG': vgg11_bn,
        'RESNET18': ResNet18,
        'RESNET20C100': Resnet20C100,
    }

    GET_OVERLAY = {
        'CL-MODEL': CLModel,
        'CL-MODEL-ISLAND': CLModelWithIslands,
        'CL-MODEL-ISLAND-TEST': CLModelIslandsTest,
    }
    
    def __init__(
        self, 
        dream_obj_type: Union[list[str], str], 
        select_task_type: Union[list[str], str],
        target_processing_type: Union[list[str], str],
        task_split_type: Union[list[str], str],
        mtype: str, 
        otype: str, 
        logger=None
    ) -> None:
        pass

class FunConfigSet(FunConfigSetBase):
    def __init__(self, 
        dream_obj_type: Union[list[str], str], 
        select_task_type: str,
        target_processing_type: str,
        task_split_type: str,
        mtype: str, 
        otype: str, 
        logger=None
    ) -> None:
        super().__init__(
            dream_obj_type=dream_obj_type, 
            select_task_type=select_task_type,
            target_processing_type=target_processing_type,
            task_split_type=task_split_type,
            mtype=mtype, 
            otype=otype, 
            logger=logger
        )

        self.model_constructor = None
        self.model_ov = None

        self.rule_buffer = []

        mtype = mtype.upper()
        if(mtype in FunConfigSet.GET_MODEL.keys()):
            self.model_constructor = FunConfigSet.GET_MODEL[mtype]
            self.rule_buffer.append(mtype)
        else:
            raise Exception(f"Unknown model type: {mtype}")

        otype = otype.upper()
        if(otype in FunConfigSet.GET_OVERLAY.keys()):
            self.model_ov = FunConfigSet.GET_OVERLAY[otype]
            self.rule_buffer.append(otype)
        else:
            raise Exception(f"Unknown model overlay type: {mtype}")

        self._init_validate_types(select_task_type)
        self._init_validate_types(target_processing_type)
        self._init_validate_types(task_split_type)

        #dream_obj_type = dream_obj_type if isinstance(dream_obj_type, list) else [dream_obj_type]
        #select_task_type = select_task_type if isinstance(select_task_type, list) else [select_task_type]
        #target_processing_type = target_processing_type if isinstance(target_processing_type, list) else [target_processing_type]
        #task_split_type = task_split_type if isinstance(task_split_type, list) else [task_split_type]

        self._init_register_template(
            dream_obj_type, 
            dream_objective.DreamObjectiveManager.GET_OBJECTIVE, 
            "Unknown dream objective type",
        )
        self._init_register_template(
            target_processing_type, 
            target_processing.TargetProcessingManager.GET_TARGET_PROCESSING, 
            "Unknown target processing type",
        )
        self._init_register_template(
            task_split_type, 
            task_split.TaskSplitManager.GET_TASK_SPLIT_PROCESSING, 
            "Unknown task split type",
        )
        self._init_register_template(
            select_task_type, 
            select_task.SelectTaskManager.GET_SELECT_TASK_PROCESSING,
            "Unknown select task type",
        )
        
        self._check_specific_combination()

        self.dream_obj_manager = dream_objective.DreamObjectiveManager(dream_obj_type)
        self.select_task_manager = select_task.SelectTaskManager(select_task_type)
        self.target_processing_manager = target_processing.TargetProcessingManager(target_processing_type)
        self.task_split_manager = task_split.TaskSplitManager(task_split_type)

    def is_target_processing_latent(self) -> bool:
        return self.target_processing_manager.is_latent()
    
    def is_dream_objective_latent(self) -> bool:
        return self.dream_obj_manager.is_latent()

    def init_dream_objectives(self, **kwargs):
        self.dream_obj_manager.init_objectives_creator(**kwargs)

    def _init_register_template(self, ftype, fdict, error_mss: str):
        if(not isinstance(ftype, list)):
            ftype = [ftype]
        for fun in ftype:
            fun = fun.upper()
            if(fun not in fdict):
                raise Exception(f"{error_mss}: {fun}")
            self._buffer_specific_combination(fun)
        
    def _buffer_specific_combination(self, fun_name):
        if(fun_name in FunConfigSet.SPECIFIC_COMBINATIONS):
            self.rule_buffer.append(fun_name)

    def _init_validate_types(self, obj):
        if(isinstance(obj, tuple)):
            raise Exception(f'Wrong type tuple')
        elif(isinstance(obj, list)):
            raise Exception(f'Wrong type list')
        elif(isinstance(obj, dict)):
            raise Exception(f'Wrong type dictionary')

    def _check_specific_combination(self):
        for fun_name in self.rule_buffer:
            if(fun_name in FunConfigSet.SPECIFIC_COMBINATIONS):
                rule_name = FunConfigSet.SPECIFIC_COMBINATIONS[fun_name]
                if(rule_name not in self.rule_buffer):
                    raise Exception(f"Broken rule: {fun_name}::{rule_name}\nBuffer: {self.rule_buffer}")

    def dream_objective(self, *args, **kwargs):
        return self.dream_obj_manager(*args, **kwargs)

    def select_task(self, *args, **kwargs):
        return self.select_task_manager(*args, **kwargs)

    def target_processing(self, *args, **kwargs):
        return self.target_processing_manager(*args, **kwargs)

    def task_split(self, *args, **kwargs):
        return self.task_split_manager(*args, **kwargs)

    def model(self, **mkwargs):
        return self.model_constructor(**mkwargs)

    def model_overlay(self, *margs, **mkwargs):
        return self.model_ov(*margs, **mkwargs)

class FunConfigSetPredefined(FunConfigSet):
    PREDEFINED_TYPES = {
        "decode": [
            "select-decremental", 
            "target-latent-decode", 
            "split-decremental", 
            "objective-latent-lossf-creator", 
            "sae", 
            "cl-model-island-test"
        ],
        "island-mean-std": [
            "select-decremental", 
            "target-latent-sample-normal-buffer", 
            "split-decremental", 
            "objective-latent-lossf-creator", 
            "sae", 
            "cl-model-island"
        ],
        "last-point": [
            "select-decremental", 
            "target-latent-buffer-last-point", 
            "split-decremental", 
            "objective-latent-lossf-creator", 
            "sae", 
            "cl-model-island"
        ],
        "custom_lossf_test": [
            "select-decremental", 
            "target-default", 
            "split-decremental", 
            "objective-channel", 
            "sae", 
            "cl-model"
        ],
        "resnet-pretrain-c100": [
            "select-decremental", 
            "target-default", 
            "split-decremental", 
            "objective-channel", 
            "resnet20c100", 
            "cl-model"
        ]

    }

    def __init__(self, name_type: str, logger=None):
        if(name_type not in FunConfigSetPredefined.PREDEFINED_TYPES):
            raise Exception(f"Unknown predefined type: {name_type}")

        type_list = FunConfigSetPredefined.PREDEFINED_TYPES[name_type]
        select_task_type = type_list[0]
        target_processing_type = type_list[1]
        task_split_type = type_list[2]
        dream_obj_type = type_list[3]
        mtype = type_list[4]
        otype = type_list[5]

        super().__init__(
            dream_obj_type=dream_obj_type, 
            select_task_type=select_task_type,
            target_processing_type=target_processing_type,
            task_split_type=task_split_type,
            mtype=mtype, 
            otype=otype, 
            logger=logger
        )