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
        'CLMODEL': 'TARGET-DEFAULT',
        'TARGET-DEFAULT': 'CLMODEL',
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
        select_task_type: Union[list[str], str],
        target_processing_type: Union[list[str], str],
        task_split_type: Union[list[str], str],
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
        self.dream_obj = None
        self.select_t = None
        self.target_proc = None
        self.task_spl = None
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

        dream_obj_type = dream_obj_type if isinstance(dream_obj_type, list) else [dream_obj_type]
        select_task_type = select_task_type if isinstance(select_task_type, list) else [select_task_type]
        target_processing_type = target_processing_type if isinstance(target_processing_type, list) else [target_processing_type]
        task_split_type = task_split_type if isinstance(task_split_type, list) else [task_split_type]

        def set_obj(self, fun):
            self.dream_obj = fun
        self._init_fun_template(
            dream_obj_type, 
            dream_objective.DreamObjectiveManager.GET_OBJECTIVE, 
            "Unknown dream objective type",
            set_obj
        )

        def set_target_process(self, fun):
            self.target_proc = fun
        self._init_fun_template(
            target_processing_type, 
            target_processing.TargetProcessingManager.GET_TARGET_PROCESSING, 
            "Unknown target processing type",
            set_target_process
        )

        def set_split_task(self, fun):
            self.task_spl = fun
        self._init_fun_template(
            task_split_type, 
            task_split.TaskSplitManager.GET_SELECT_TASK_PROCESSING, 
            "Unknown task split type",
            set_split_task
        )

        def set_select_task(self, fun):
            self.select_t = fun
        self._init_fun_template(
            select_task_type, 
            select_task.SelectTaskManager.GET_SELECT_TASK_PROCESSING,
            "Unknown select task type",
            set_select_task
        )

        self._check_specific_combination()
        
    def _buffer_specific_combination(self, fun_name):
        if(fun_name in FunConfigSet.SPECIFIC_COMBINATIONS):
            self.rule_buffer.append(fun_name)

    def _check_specific_combination(self):
        for fun_name in self.rule_buffer:
            if(fun_name in FunConfigSet.SPECIFIC_COMBINATIONS):
                rule_name = FunConfigSet.SPECIFIC_COMBINATIONS[fun_name]
                if(rule_name not in self.rule_buffer):
                    raise Exception(f"Broken rule: {fun_name}::{rule_name}\n{self.rule_buffer}")

    def _init_fun_template(self, ftype, fdict, error_mss: str, set_val_f):
        for fun in ftype:
            fun = fun.upper()
            if(fun not in fdict):
                raise Exception(f"{error_mss}: {fun}")
            self._buffer_specific_combination(fun)
            set_val_f(self, fdict[fun])
          
    def dream_objective(self, *args, **kwargs):
        return self.dream_obj(*args, **kwargs)

    def select_task(self, *args, **kwargs):
        return self.select_t(*args, **kwargs)

    def target_processing(self, *args, **kwargs):
        return self.target_proc(*args, **kwargs)

    def task_split(self, *args, **kwargs):
        return self.task_spl(*args, **kwargs)

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
        "normal-buffer": [
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
        "resnet-default": [
            "select-decremental", 
            "target-default", 
            "split-decremental", 
            "objective-resnet20-c100-channel", 
            "resnet", 
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