from typing import List, Union
import numpy as np
from collections import deque

"""
    Different reward curricula that can be used by the Nermo envs for gradually increasing the weight of the secondary reward components.
    The reward curricula are integrated in the training environments and called within the `step` function to adapt the curriculum factors and for the reward computation where the curriculum factors are used.
"""

class BaseCurriculum(object):
    def __init__(self, initial_curriculum_factors: List[float], max_curriculum_factors: Union[float, List[float]]):
        self.initial_curriculum_factors = np.array(initial_curriculum_factors)
        self.n_steps = 0
        self.max_curriculum_factors = [max_curriculum_factors] if isinstance(max_curriculum_factors, float) else max_curriculum_factors
        self._curriculum_factors = initial_curriculum_factors
    
    @property
    def curriculum_factors(self):
        if self.max_curriculum_factors is not None:
            return np.minimum(self._curriculum_factors, self.max_curriculum_factors)
        return self._curriculum_factors

    def step_curriculum(self, primary_reward: float, secondary_reward: float):
        self.n_steps += 1


class LinearTimeCurriculum(BaseCurriculum):
    def __init__(self, increase_per_step: Union[float, List[float]], **kwargs):
        self.increase_per_step = increase_per_step

        super().__init__(**kwargs)

    def step_curriculum(self, primary_reward: float, secondary_rewards: List[float]):
        self.n_steps += 1

        self._curriculum_factors = np.array(self.initial_curriculum_factors) + (self.n_steps * np.array(self.increase_per_step))


class ProportionalSecondaryRewardCurriculum(BaseCurriculum):
    def __init__(self, secondary_reward_proportions: Union[float, List[float]], adaption_steps: List[int], reward_buffer_size=2000, **kwargs):
        self.secondary_reward_proportions = [secondary_reward_proportions] if isinstance(secondary_reward_proportions, float) else secondary_reward_proportions
        self.secondary_reward_proportions = np.array(self.secondary_reward_proportions)
        self.adaption_steps = adaption_steps
        self.reward_buffer_size = reward_buffer_size

        self.primary_reward_buffer = deque([], maxlen=self.reward_buffer_size)
        self.secondary_rewards_buffer = deque([], maxlen=self.reward_buffer_size)

        super().__init__(**kwargs)

    def step_curriculum(self, primary_reward: float, secondary_rewards: List[float]):
        self.n_steps += 1

        self.primary_reward_buffer.append(primary_reward)
        self.secondary_rewards_buffer.append(secondary_rewards)

        if self.n_steps in self.adaption_steps:
            abs_primary_reward = np.abs(np.sum(self.primary_reward_buffer))
            abs_secondary_rewards = np.abs(np.sum(self.secondary_rewards_buffer, axis=0))
            self._curriculum_factors = np.divide(self.secondary_reward_proportions * abs_primary_reward, abs_secondary_rewards, out=np.zeros_like(abs_secondary_rewards), where=abs_secondary_rewards!=0)



# Experimental/not recommended
class DiscreteAdaptiveCurriculum(BaseCurriculum):
    def __init__(self, increase_on_success: float, success_treshold: float, n_adaption_steps: int, **kwargs):
        self.increase_on_success = increase_on_success
        self.success_treshold = success_treshold
        self.n_adaption_steps = n_adaption_steps

        self.n_success_events = 0
        self.primary_reward_buffer = deque([], maxlen=self.n_adaption_steps)

        super().__init__(**kwargs)

    def step_curriculum(self, primary_reward: float, secondary_rewards: List[float]):
        self.n_steps += 1

        self.primary_reward_buffer.append(primary_reward)

        if len(self.primary_reward_buffer) >= self.n_adaption_steps and np.mean(self.primary_reward_buffer) >= self.success_treshold:
            self.n_success_events += 1
            self.primary_reward_buffer = deque([], maxlen=self.n_adaption_steps)

        self._curriculum_factors = self.initial_curriculum_factors + (self.n_success_events * self.increase_on_success) 

class RewardPreservingAdaptiveCurriculum(BaseCurriculum):
    def __init__(self, preserved_reward_proportion: float, adaption_steps: List[int], reward_buffer_size=2000, **kwargs):
        self.preserved_reward_proportion = preserved_reward_proportion
        self.adaption_steps = adaption_steps
        self.reward_buffer_size = reward_buffer_size

        self.primary_reward_buffer = deque([], maxlen=self.reward_buffer_size)
        self.secondary_rewards_buffer = deque([], maxlen=self.reward_buffer_size)

        super().__init__(**kwargs)

    def step_curriculum(self, primary_reward: float, secondary_rewards: List[float]):
        self.n_steps += 1

        self.primary_reward_buffer.append(primary_reward)
        self.secondary_rewards_buffer.append(secondary_rewards)

        if self.n_steps in self.adaption_steps:
            primary_reward = np.sum(self.primary_reward_buffer)
            secondary_rewards = np.sum(self.secondary_rewards_buffer, axis=0)
            
            previous_total_reward = primary_reward + np.dot(self.curriculum_factors, secondary_rewards)
            if np.sum(secondary_rewards) != 0:
                self._curriculum_factors = [(self.preserved_reward_proportion * previous_total_reward - primary_reward) / np.sum(secondary_rewards)] * len(secondary_rewards)
            else:
                self._curriculum_factors = [0] * len(secondary_rewards)
