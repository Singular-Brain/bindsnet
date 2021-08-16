from abc import ABC, abstractmethod
import torch
from bindsnet import manual_seed


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
        self.single_output_layer = kwargs['single_output_layer']
        self.tc_dps = kwargs.get('tc_dps')

        self.dopamine_per_spike = kwargs.get('dopamine_per_spike', 0.01)
        self.negative_dopamine_per_spike = kwargs.get('negative_dopamine_per_spike', 0.0)
        self.dopamine_for_correct_pred = kwargs.get('dopamine_for_correct_pred', 1.0)
        self.tc_reward = kwargs.get('tc_reward')
        self.dopamine_base = kwargs.get('dopamine_base', 0.002)
        self.give_reward = kwargs['give_reward']
        dt = torch.as_tensor(self.dt)
        self.decay = torch.exp(-dt / self.tc_reward)

        self.variant = kwargs['variant']

        self.label = kwargs.get('labels', None)

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
        if self.label is None:
            return 0
        
        if self.single_output_layer:
            s = self.layers.s
            assert s.shape[0] == 1, "This method has not yet been implemented for batch_size>1 !" 
            self.dopamine = (
                            self.decay
                            * (self.dopamine - self.dopamine_base)
                            + self.dopamine_base
            ).to(s.device)

            target_spikes = (s[:,self.label*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)                

            if self.variant == 'true_pred':
                label_spikes = [0.0]*self.n_labels
                for i in range(self.n_labels):
                    label_spikes[i] = (s[:,i*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)
                if target_spikes == max(label_spikes):
                    self.dopamine += target_spikes * self.dopamine_per_spike
                else:
                    self.dopamine -= max(label_spikes) * self.negative_dopamine_per_spike

                
            elif self.variant == "pure_per_spike":
                self.dopamine += target_spikes * self.dopamine_per_spike - (s.sum()-target_spikes) * self.negative_dopamine_per_spike
            
            else:
                raise ValueError("variant not specified")

        else:
            target_layer = self.layers[f"output_{self.label}"]
            target_spikes = target_layer.s
            
            self.dopamine = (
                    self.decay
                    * (self.dopamine - self.dopamine_base)
                    + self.dopamine_base
                    ).to(target_spikes.device)

            if self.variant == 'true_pred': 
                output_layers_spikes = [self.layers[l].s.sum() for l in self.layers if l.startswith("output")]   
                if target_spikes.sum() == max(output_layers_spikes): 
                    self.dopamine += target_spikes.sum() * self.dopamine_per_spike
                else :
                    self.dopamine -= max(output_layers_spikes).sum() * self.negative_dopamine_per_spike
                    
            elif self.variant == "pure_per_spike":
                for name, layer in self.layers.items():
                    if layer == target_layer:
                        self.dopamine += layer.s.sum() * self.dopamine_per_spike
                    else:
                        self.dopamine -= layer.s.sum() * self.negative_dopamine_per_spike
            
            else:
                raise ValueError("variant not specified")
        
        return self.dopamine


class DopaminergicRPE(AbstractReward):
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
        self.reward_predict = torch.tensor(1.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(1.0)  # Predicted reward per episode.
        self.rewards_predict_episode = ([])  # List of predicted rewards per episode (used for plotting).\

        self.reward_predict_pos = torch.tensor(1.0)  # Predicted reward (per step).
        self.reward_predict_episode_pos = torch.tensor(1.0)  # Predicted reward per episode.
        self.rewards_predict_episode_pos = ([])  # List of predicted rewards per episode (used for plotting).
        
        self.reward_predict_neg = torch.tensor(1.0)  # Predicted reward (per step).
        self.reward_predict_episode_neg = torch.tensor(1.0)  # Predicted reward per episode.
        self.rewards_predict_episode_neg = ([])  # List of predicted rewards per episode (used for plotting).
        
        self.accumulated_reward = torch.tensor(1.0)
        self.variant = None

    def compute(self, **kwargs) -> torch.Tensor:
        # language=rst
        """
        Computes the reward prediction error using EMA.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Current reward.
        :return: Reward prediction error.
        """
        self.td_nu = kwargs.get('td_nu',0.0001)
        self.dps_base = kwargs.get('dopamine_per_spike_base', 0.01)
        self.negative_dps_base = kwargs.get('negative_dopamine_per_spike_base', 0.0)
        self.layers = kwargs.get('dopaminergic_layers')
        self.n_labels = kwargs.get('n_labels')
        self.n_per_class = kwargs.get('neuron_per_class')
        self.single_output_layer = kwargs['single_output_layer']
        self.tc_reward = kwargs.get('tc_reward')
        self.dopamine_for_correct_pred = kwargs.get('dopamine_for_correct_pred', 1.0)
        self.dopamine_base = kwargs.get('dopamine_base', 0.002)
        dt = torch.as_tensor(self.dt)
        self.decay = torch.exp(-dt / self.tc_reward)
        self.label = kwargs.get('labels', None)
        self.dopamine = self.dopamine_base
        self.variant = kwargs['variant']
        self.sub_variant = kwargs['sub_variant']

        if self.sub_variant == 'just_decay':
            self.dps = self.dps_base
            self.negative_dps = self.negative_dps_base

        elif self.sub_variant == 'normal':
            self.dps = self.dps_base / self.reward_predict_episode_pos
            self.negative_dps = self.negative_dps_base / self.reward_predict_episode_neg
        
        elif self.sub_variant == 'td_error':
            if self.accumulated_reward > 0 :
                self.dps = self.dps_base - self.td_nu*(self.accumulated_reward-self.reward_predict_episode_pos)
            else :
                self.negative_dps = self.negative_dps_base + self.td_nu*(self.accumulated_reward-self.reward_predict_episode_neg)
        
        else:
            raise ValueError('sub_variant not specified!')


        return self.dopamine


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
        self.accumulated_reward = kwargs["accumulated_reward"]
        steps = torch.tensor(kwargs["steps"]).float()
        ema_window = torch.tensor(kwargs.get("ema_window", 10.0))

        # Compute average reward per step.
        self.reward = self.accumulated_reward / steps

        # Update EMAs.
        self.reward_predict = (
            1 - 1 / ema_window
        ) * self.reward_predict + 1 / ema_window * self.reward
        self.reward_predict_episode = (
            1 - 1 / ema_window
        ) * self.reward_predict_episode + 1 / ema_window * self.accumulated_reward
        self.rewards_predict_episode.append(self.reward_predict_episode.item())

        if self.accumulated_reward > 0 :
            self.reward_pos = self.accumulated_reward / steps
            self.reward_predict_pos = (1 - 1 / ema_window) * self.reward_predict_pos + 1 / ema_window * self.reward_pos
            self.reward_predict_episode_pos = (1 - 1 / ema_window) * self.reward_predict_episode_pos + 1 / ema_window * self.accumulated_reward
            self.rewards_predict_episode_pos.append(self.reward_predict_episode_pos.item()) 
        
        else:
            self.reward_neg = self.accumulated_reward / steps
            self.reward_predict_neg = (1 - 1 / ema_window) * self.reward_predict_neg + 1 / ema_window * self.reward_neg
            self.reward_predict_episode_neg = (1 - 1 / ema_window) * self.reward_predict_episode_neg + 1 / ema_window * self.accumulated_reward
            self.rewards_predict_episode_neg.append(self.reward_predict_episode_neg.item())             


    def online_compute(self, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per
        episode.
        """
        if self.label is None:
            return 0
        
        if self.single_output_layer:
            s = self.layers.s
            assert s.shape[0] == 1, "This method has not yet been implemented for batch_size>1 !" 
            self.dopamine = (
                            self.decay
                            * (self.dopamine - self.dopamine_base)
                            + self.dopamine_base
            ).to(s.device)
            target_spikes = (s[:,self.label*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)                
            #self.dopamine += target_spikes * self.dps #- (s.sum()-target_spikes) * self.negative_dopamine_per_spike
        
            if self.variant == 'true_pred':
                label_spikes = [0.0]*self.n_labels
                for i in range(self.n_labels):
                    label_spikes[i] = (s[:,i*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)
                if target_spikes == max(label_spikes):
                    self.dopamine += target_spikes * self.dps
                else:
                    self.dopamine -= max(label_spikes) * self.negative_dps
            elif self.variant == "pure_per_spike":
                self.dopamine += target_spikes * self.dps - (s.sum()-target_spikes) * self.negative_dps
            else:
                raise ValueError("variant not specified")
        
        
        else:
            target_layer = self.layers[f"output_{self.label}"]
            target_spikes = target_layer.s
            
            self.dopamine = (
                    self.decay
                    * (self.dopamine - self.dopamine_base)
                    + self.dopamine_base
                    ).to(target_spikes.device)

            if self.variant == 'true_pred':
                output_layers_spikes = [self.layers[l].s.sum() for l in self.layers if l.startswith("output")]   
                if target_spikes.sum() == max(output_layers_spikes): 
                    self.dopamine += target_spikes.sum() * self.dps
                else :
                    self.dopamine -= max(output_layers_spikes).sum() * self.negative_dps
                    
            elif self.variant == "pure_per_spike":
                for name, layer in self.layers.items():
                    if layer == target_layer:
                        self.dopamine += layer.s.sum() * self.dps
                    else:
                        self.dopamine -= layer.s.sum() * self.negative_dps
            
            else:
                raise ValueError("variant not specified")
        
        return self.dopamine