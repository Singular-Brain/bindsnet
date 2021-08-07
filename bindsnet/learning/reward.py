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
        self.layers = kwargs.get('dopaminergic_layers')
        self.n_labels = kwargs.get('n_labels')
        self.n_per_class = kwargs.get('neuron_per_class')
        self.dopamine_per_spike = kwargs.get('dopamine_per_spike', 0.01)
        self.dopamine_for_correct_pred = kwargs.get('dopamine_for_correct_pred', 1.0)
        self.tc_reward = kwargs.get('tc_reward')
        self.dopamine_base = kwargs.get('dopamine_base', 0.002)
        self.give_reward = kwargs.get('give_reward', False)
        dt = torch.as_tensor(self.dt)
        self.decay = torch.exp(-dt / self.tc_reward)

        self.variant = kwargs['variant']

        self.label = kwargs['labels']

        self.dopamine = self.dopamine_base
        if self.give_reward: 
            self.dopamine += self.dopamine_for_correct_pred
        
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
        # assert s.shape[0] == 1, "This method has not yet been implemented for batch_size>1 !" 
        target_spikes = self.layers[f"output_{self.label}"].s
        self.dopamine = (
                        self.decay
                        * (self.dopamine - self.dopamine_base)
                        + self.dopamine_base
        ).to(target_spikes.device)

        self.dopamine += target_spikes.sum() * self.dopamine_per_spike
        # target_spikes = (s[:,self.label*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)
        # if self.dopamine_for_correct_pred != 0 or self.variant == 'true_pred':
        #     label_spikes = [0.0]*self.n_labels
        #     for i in range(self.n_labels):
        #         label_spikes[i] = (s[:,i*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)
        #     if target_spikes == max(label_spikes):
        #         self.dopamine += self.dopamine_for_correct_pred

        # if self.variant == 'true_pred':
        #     if target_spikes == max(label_spikes):
        #         self.dopamine += target_spikes * self.dopamine_per_spike
        # else:
        #     self.dopamine += target_spikes * self.dopamine_per_spike

        return self.dopamine