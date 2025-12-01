"""Generation scheduler module - manages token generation order and scheduling strategies"""

import math
import random
import torch
import numpy as np
from typing import List, Union, Optional
from abc import ABC, abstractmethod


class BaseGenerationScheduler(ABC):
    """Base class for generation schedulers"""
    
    def __init__(self, num_steps: int, total_tokens: int):
        """
        Args:
            num_steps: Number of generation steps
            total_tokens: Total number of tokens
        """
        self.num_steps = num_steps
        self.total_tokens = total_tokens
        self.token_order = None
        self.num_step_list = None
        
    @abstractmethod
    def generate_token_order(self) -> torch.Tensor:
        """Generate token order, returns a tensor with shape (total_tokens,)"""
        pass
    
    @abstractmethod
    def generate_step_schedule(self) -> List[int]:
        """Generate token count schedule for each step, returns a list with length num_steps"""
        pass
    
    def schedule(self) -> List[List[int]]:
        """Execute scheduling, returns a list of token positions needed for each step"""
        if self.token_order is None:
            self.token_order = self.generate_token_order()
        
        if self.num_step_list is None:
            self.num_step_list = self.generate_step_schedule()
        
        # Verify scheduling correctness
        assert sum(self.num_step_list) == self.total_tokens, \
            f"Scheduled total doesn't match: {sum(self.num_step_list)} != {self.total_tokens}"
        
        # Generate scheduling list
        schedule_list = []
        start_idx = 0
        for step_tokens in self.num_step_list:
            end_idx = start_idx + step_tokens
            step_tokens_indices = self.token_order[start_idx:end_idx].tolist()
            schedule_list.append(step_tokens_indices)
            start_idx = end_idx
        
        return schedule_list


class SequentialScheduler(BaseGenerationScheduler):
    """Sequential generation scheduler"""
    
    def generate_token_order(self) -> torch.Tensor:
        """Generate sequential token order: [0, 1, 2, ..., total_tokens-1]"""
        return torch.arange(self.total_tokens)
    
    def generate_step_schedule(self) -> List[int]:
        """Uniform grouping"""
        base_size = self.total_tokens // self.num_steps
        remainder = self.total_tokens % self.num_steps
        
        # Allocate one extra token for the first remainder steps
        schedule = [base_size + 1] * remainder + [base_size] * (self.num_steps - remainder)
        return schedule


class RandomScheduler(BaseGenerationScheduler):
    """Random generation scheduler"""
    
    def __init__(self, num_steps: int, total_tokens: int, seed: Optional[int] = None):
        super().__init__(num_steps, total_tokens)
        self.seed = seed
        
    def generate_token_order(self) -> torch.Tensor:
        """Generate random token order"""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return torch.randperm(self.total_tokens)
    
    def generate_step_schedule(self) -> List[int]:
        """Uniform grouping"""
        base_size = self.total_tokens // self.num_steps
        remainder = self.total_tokens % self.num_steps
        
        # Allocate one extra token for the first remainder steps
        schedule = [base_size + 1] * remainder + [base_size] * (self.num_steps - remainder)
        return schedule


class SchedulerFactory:
    """Scheduler factory class"""
    
    @staticmethod
    def create_scheduler(scheduler_type: str, 
                       num_steps: int, 
                       total_tokens: int,
                       **kwargs) -> BaseGenerationScheduler:
        """
        Create scheduler
        
        Args:
            scheduler_type: Scheduler type ('sequential', 'random')
            num_steps: Number of generation steps
            total_tokens: Total number of tokens
            **kwargs: Additional parameters
                - seed: Random seed (random scheduler)
        """
        scheduler_classes = {
            'sequential': SequentialScheduler,
            'random': RandomScheduler,
        }
        
        if scheduler_type not in scheduler_classes:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        scheduler_class = scheduler_classes[scheduler_type]
        
        return scheduler_class(num_steps=num_steps, total_tokens=total_tokens, **kwargs)


def get_generation_scheduler(scheduler_type: str,
                           num_steps: int,
                           total_tokens: int,
                           seed: Optional[int] = None) -> BaseGenerationScheduler:
    """
    Utility function: get generation scheduler
    
    Args:
        scheduler_type: Scheduler type
        num_steps: Number of generation steps
        total_tokens: Total number of tokens
        seed: Random seed (random scheduler needs)
    
    Returns:
        BaseGenerationScheduler instance
    """
    kwargs = {}
    
    if scheduler_type == 'random' and seed is not None:
        kwargs['seed'] = seed
    
    return SchedulerFactory.create_scheduler(
        scheduler_type=scheduler_type,
        num_steps=num_steps,
        total_tokens=total_tokens,
        **kwargs
    )