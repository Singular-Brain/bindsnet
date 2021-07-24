from abc import ABC, abstractmethod

import torch


class AbstractReward(ABC):
    # language=rst
    """
    Abstract base class for reward computation.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        # language=rst
        """
        Computes/modifies reward.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        pass

    @abstractmethod
    def online_compute(self, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        pass

class MovingAvgRPE(AbstractReward):
    # language=rst
    """
    Computes reward prediction error (RPE) based on an exponential moving average (EMA)
    of past rewards.
    """

    def __init__(self, **kwargs) -> None:
        # language=rst
        """
        Constructor for EMA reward prediction error.
        """
        self.reward_predict = torch.tensor(0.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(0.0)  # Predicted reward per episode.
        self.rewards_predict_episode = (
            []
        )  # List of predicted rewards per episode (used for plotting).

    def compute(self, **kwargs) -> torch.Tensor:
        # language=rst
        """
        Computes the reward prediction error using EMA.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Current reward.
        :return: Reward prediction error.
        """
        # Get keyword arguments.
        reward = kwargs["reward"]

        return reward - self.reward_predict

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Updates the EMAs. Called once per episode.

        Keyword arguments:

        :param Union[float, torch.Tensor] accumulated_reward: Reward accumulated over
            one episode.
        :param int steps: Steps in that episode.
        :param float ema_window: Width of the averaging window.
        """
        # Get keyword arguments.
        accumulated_reward = kwargs["accumulated_reward"]
        steps = torch.tensor(kwargs["steps"]).float()
        ema_window = torch.tensor(kwargs.get("ema_window", 10.0))

        # Compute average reward per step.
        reward = accumulated_reward / steps

        # Update EMAs.
        self.reward_predict = (
            1 - 1 / ema_window
        ) * self.reward_predict + 1 / ema_window * reward
        self.reward_predict_episode = (
            1 - 1 / ema_window
        ) * self.reward_predict_episode + 1 / ema_window * accumulated_reward
        self.rewards_predict_episode.append(self.reward_predict_episode.item())


class DynamicDopamineInjection(AbstractReward):
    # language=rst
    """

    """
    def compute(self, **kwargs) -> None:
        # language=rst
        """
        Computes/modifies reward.
        """
        self.l = kwargs.get('dopaminergic_layer')
        self.n_labels = kwargs.get('n_labels')
        self.n_per_class = kwargs.get('neuron_per_class')
        self.dopamine_per_spike = kwargs.get('dopamine_per_spike')
        self.tc_reward = kwargs.get('tc_reward')
        self.dopamine_base = kwargs.get('dopamine_base')
        dt = torch.as_tensor(self.dt)
        self.decay = torch.exp(-dt / self.tc_reward) 

        self.label = kwargs['labels']

        self.dopamine = self.dopamine_base
        return self.dopamine
        
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        pass

    def online_compute(self, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        s = self.network.layers[self.l].s
        assert s.shape[0] == 1, "This method has not yet been implemented for batch_size>1 !" 
        self.dopamine = (
                        self.decay
                        * (self.dopamine - self.dopamine_base)
                        + self.dopamine_base
        )
        target_spikes = (s[self.label*self.n_per_class: (self.label+1)*self.n_per_class]).sum()
        self.dopamine += target_spikes * self.dopamine_per_spike

        return self.dopamine