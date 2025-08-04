import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy #fl_server.py
import logging
from datetime import datetime
import json
import os


class FederatedServer:
    """Federated Learning Server for coordinating training and attacks"""

    def __init__(self, global_model: nn.Module, device: str = 'cuda',
                 aggregation_method: str = 'fedavg', malicious: bool = False,
                 attack_config: Optional[Dict[str, Any]] = None):
        """
        Initialize FL Server

        Args:
            global_model: Global model to be distributed to clients
            device: Computing device
            aggregation_method: Method for aggregating client updates
            malicious: Whether server performs gradient inversion attacks
            attack_config: Configuration for attacks if malicious=True
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        self.malicious = malicious
        self.attack_config = attack_config or {}

        # Server state
        self.current_round = 0
        self.client_updates = {}
        self.round_history = []
        self.attack_results = {}

        # Model parameters tracking
        self.initial_model_state = copy.deepcopy(self.global_model.state_dict())

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Performance metrics
        self.metrics = {
            'communication_costs': [],
            'convergence_metrics': [],
            'attack_success_rates': [],
            'privacy_costs': []
        }

    def start_round(self) -> Dict[str, torch.Tensor]:
        """
        Start a new federated learning round

        Returns:
            Current global model parameters
        """
        self.current_round += 1
        self.client_updates = {}

        self.logger.info(f"Starting FL Round {self.current_round}")

        # Return current global model parameters
        return self._get_model_parameters()

    def _get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return {name: param.clone().detach()
                for name, param in self.global_model.named_parameters()}

    def receive_client_update(self, client_id: str,
                              gradients: Dict[str, torch.Tensor],
                              num_samples: int,
                              client_metrics: Optional[Dict[str, float]] = None):
        """
        Receive update from a client

        Args:
            client_id: Unique client identifier
            gradients: Client's parameter gradients
            num_samples: Number of samples used by client
            client_metrics: Optional client performance metrics
        """
        self.client_updates[client_id] = {
            'gradients': gradients,
            'num_samples': num_samples,
            'metrics': client_metrics or {},
            'timestamp': datetime.now()
        }

        self.logger.info(f"Received update from client {client_id} "
                         f"(samples: {num_samples})")

    def aggregate_and_update(self, defense_mechanism: Optional[Any] = None) -> Dict[str, Any]:
        """
        Aggregate client updates and update global model

        Args:
            defense_mechanism: Optional defense to apply during aggregation

        Returns:
            Round results including attack outcomes if malicious
        """
        if not self.client_updates:
            self.logger.warning("No client updates received")
            return {}

        # Perform gradient inversion attack if malicious server
        attack_results = {}
        if self.malicious:
            attack_results = self._perform_gradient_inversion_attack()

        # Apply defense mechanism if provided
        defended_updates = self.client_updates
        if defense_mechanism:
            defended_updates = self._apply_server_defense(defense_mechanism)

        # Aggregate client updates
        aggregated_gradients = self._aggregate_gradients(defended_updates)

        # Update global model
        self._update_global_model(aggregated_gradients)

        # Record round results
        round_results = {
            'round': self.current_round,
            'num_clients': len(self.client_updates),
            'total_samples': sum(update['num_samples'] for update in self.client_updates.values()),
            'attack_results': attack_results,
            'convergence_metrics': self._compute_convergence_metrics()
        }

        self.round_history.append(round_results)

        return round_results

    def _perform_gradient_inversion_attack(self) -> Dict[str, Any]:
        """Perform gradient inversion attack on client gradients"""
        from attacks.adapter_attack import AdapterGradientInversionAttack
        from attacks.prefix_attack import PrefixGradientInversionAttack
        from attacks.bias_attack import BiasGradientInversionAttack
        from attacks.lora_attack import LoRAGradientInversionAttack

        attack_results = {}

        # Determine PEFT method from attack config
        peft_method = self.attack_config.get('peft_method', 'adapter')

        # Select appropriate attack class
        attack_classes = {
            'adapter': AdapterGradientInversionAttack,
            'prefix': PrefixGradientInversionAttack,
            'bias': BiasGradientInversionAttack,
            'lora': LoRAGradientInversionAttack
        }

        if peft_method not in attack_classes:
            self.logger.error(f"Unknown PEFT method: {peft_method}")
            return attack_results

        # Initialize attack
        attack_class = attack_classes[peft_method]
        attacker = attack_class(self.global_model, self.device, **self.attack_config)

        # Perform attack on each client's gradients
        for client_id, update in self.client_updates.items():
            try:
                # Setup malicious parameters
                attacker.setup_malicious_parameters()

                # Extract gradients
                client_gradients = update['gradients']

                # Reconstruct patches from gradients
                recovered_patches = attacker.reconstruct_patches(client_gradients)

                # Evaluate attack success
                attack_metrics = {
                    'recovered_patches': len(recovered_patches),
                    'attack_type': peft_method,
                    'client_id': client_id
                }

                attack_results[client_id] = {
                    'patches': recovered_patches,
                    'metrics': attack_metrics
                }

                self.logger.info(f"Attack on client {client_id}: "
                                 f"recovered {len(recovered_patches)} patches")

            except Exception as e:
                self.logger.error(f"Attack failed on client {client_id}: {e}")
                attack_results[client_id] = {'error': str(e)}

        return attack_results

    def _apply_server_defense(self, defense_mechanism) -> Dict[str, Any]:
        """Apply server-side defense to client updates"""
        defended_updates = {}

        for client_id, update in self.client_updates.items():
            try:
                # Apply defense to client gradients
                defended_gradients = defense_mechanism.apply_defense_with_validation(
                    update['gradients']
                )

                # Create defended update
                defended_updates[client_id] = {
                    **update,
                    'gradients': defended_gradients,
                    'defense_applied': defense_mechanism.__class__.__name__
                }

            except Exception as e:
                self.logger.error(f"Defense failed for client {client_id}: {e}")
                defended_updates[client_id] = update  # Use original if defense fails

        return defended_updates

    def _aggregate_gradients(self, client_updates: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Aggregate client gradients using specified method"""
        from .aggregation import FedAvgAggregator, WeightedAggregator, MedianAggregator

        # Select aggregation method
        if self.aggregation_method == 'fedavg':
            aggregator = FedAvgAggregator()
        elif self.aggregation_method == 'weighted':
            aggregator = WeightedAggregator()
        elif self.aggregation_method == 'median':
            aggregator = MedianAggregator()
        else:
            self.logger.warning(f"Unknown aggregation method: {self.aggregation_method}, using FedAvg")
            aggregator = FedAvgAggregator()

        # Prepare data for aggregation
        gradients_list = []
        weights_list = []

        for client_id, update in client_updates.items():
            gradients_list.append(update['gradients'])
            weights_list.append(update['num_samples'])

        # Aggregate
        aggregated_gradients = aggregator.aggregate(gradients_list, weights_list)

        return aggregated_gradients

    def _update_global_model(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """Update global model with aggregated gradients"""
        learning_rate = self.attack_config.get('server_lr', 0.01)

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= learning_rate * aggregated_gradients[name]

    def _compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics for current round"""
        metrics = {}

        # Compute parameter change from initial state
        current_params = self._get_model_parameters()
        initial_params = self.initial_model_state

        total_change = 0.0
        total_norm = 0.0

        for name in current_params:
            if name in initial_params:
                param_change = torch.norm(current_params[name] - initial_params[name]).item()
                param_norm = torch.norm(current_params[name]).item()

                total_change += param_change ** 2
                total_norm += param_norm ** 2

        metrics['total_parameter_change'] = np.sqrt(total_change)
        metrics['total_parameter_norm'] = np.sqrt(total_norm)

        if total_norm > 0:
            metrics['relative_change'] = metrics['total_parameter_change'] / metrics['total_parameter_norm']
        else:
            metrics['relative_change'] = 0.0

        return metrics

    def get_global_model(self) -> nn.Module:
        """Get current global model"""
        return self.global_model

    def get_round_results(self, round_num: Optional[int] = None) -> Dict[str, Any]:
        """Get results for specific round or current round"""
        if round_num is None:
            round_num = self.current_round

        for round_result in self.round_history:
            if round_result['round'] == round_num:
                return round_result

        return {}

    def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of all attack results"""
        if not self.malicious:
            return {'message': 'Server is not malicious'}

        total_attacks = 0
        successful_attacks = 0
        total_patches_recovered = 0

        for round_result in self.round_history:
            attack_results = round_result.get('attack_results', {})

            for client_id, client_attack in attack_results.items():
                if 'error' not in client_attack:
                    total_attacks += 1
                    patches_recovered = len(client_attack.get('patches', []))
                    total_patches_recovered += patches_recovered

                    if patches_recovered > 0:
                        successful_attacks += 1

        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        avg_patches = total_patches_recovered / total_attacks if total_attacks > 0 else 0.0

        return {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'success_rate': success_rate,
            'avg_patches_per_attack': avg_patches,
            'total_patches_recovered': total_patches_recovered
        }

    def save_results(self, filepath: str):
        """Save server results to file"""
        results = {
            'server_config': {
                'aggregation_method': self.aggregation_method,
                'malicious': self.malicious,
                'attack_config': self.attack_config
            },
            'round_history': self.round_history,
            'attack_summary': self.get_attack_summary(),
            'final_convergence': self._compute_convergence_metrics()
        }

        # Convert tensors to lists for JSON serialization
        def tensor_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_list(item) for item in obj]
            else:
                return obj

        serializable_results = tensor_to_list(results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        self.logger.info(f"Server results saved to {filepath}")


class MaliciousServer(FederatedServer):
    """Specialized malicious server for gradient inversion attacks"""

    def __init__(self, global_model: nn.Module, peft_method: str,
                 device: str = 'cuda', **kwargs):
        """
        Initialize malicious server

        Args:
            global_model: Global model with PEFT method
            peft_method: Type of PEFT method for targeted attacks
            device: Computing device
            **kwargs: Additional configuration
        """
        attack_config = {
            'peft_method': peft_method,
            'malicious_params': True,
            **kwargs
        }

        super().__init__(global_model, device, 'fedavg', malicious=True, attack_config=attack_config)

        # Setup malicious parameters during initialization
        self._setup_malicious_model()

    def _setup_malicious_model(self):
        """Setup malicious parameters in the global model"""
        peft_method = self.attack_config['peft_method']

        self.logger.info(f"Setting up malicious {peft_method} parameters")

        if peft_method == 'adapter':
            self._setup_malicious_adapters()
        elif peft_method == 'prefix':
            self._setup_malicious_prefix()
        elif peft_method == 'bias':
            self._setup_malicious_bias()
        elif peft_method == 'lora':
            self._setup_malicious_lora()

    def _setup_malicious_adapters(self):
        """Setup malicious adapter parameters"""
        with torch.no_grad():
            for i, block in enumerate(self.global_model.blocks):
                if hasattr(block, 'attn_peft') and block.attn_peft is not None:
                    adapter = block.attn_peft

                    # Set down projection to extract patterns
                    nn.init.zeros_(adapter.down_proj.weight)
                    nn.init.zeros_(adapter.down_proj.bias)

                    # Set up projection for identity mapping
                    nn.init.zeros_(adapter.up_proj.weight)
                    min_dim = min(adapter.up_proj.weight.shape[0], adapter.down_proj.weight.shape[0])
                    adapter.up_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim) * 0.1

    def _setup_malicious_prefix(self):
        """Setup malicious prefix parameters"""
        with torch.no_grad():
            for block in self.global_model.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'prefix_encoder'):
                    prefix_encoder = block.attn.prefix_encoder

                    # Set prefix embeddings to position patterns
                    for j in range(prefix_encoder.prefix_length):
                        encoding = torch.zeros(self.global_model.embed_dim, device=self.device)
                        for d in range(0, self.global_model.embed_dim, 2):
                            encoding[d] = np.sin(j / (10000 ** (2 * d / self.global_model.embed_dim)))
                            if d + 1 < self.global_model.embed_dim:
                                encoding[d + 1] = np.cos(j / (10000 ** (2 * (d + 1) / self.global_model.embed_dim)))
                        prefix_encoder.prefix_embeddings.data[j] = encoding

    def _setup_malicious_bias(self):
        """Setup malicious bias parameters"""
        with torch.no_grad():
            for i, block in enumerate(self.global_model.blocks):
                pos_encoding = self.global_model.pos_embed[0, i % self.global_model.pos_embed.size(1)] if hasattr(
                    self.global_model, 'pos_embed') else None

                if pos_encoding is not None:
                    # Set various bias modules
                    bias_modules = ['attn_qkv_bias', 'attn_proj_bias', 'mlp_fc1_bias', 'mlp_fc2_bias']

                    for bias_name in bias_modules:
                        if hasattr(block, bias_name):
                            bias_module = getattr(block, bias_name)
                            if hasattr(bias_module, 'bias'):
                                bias_dim = bias_module.bias.shape[0]
                                pos_dim = min(bias_dim, len(pos_encoding))
                                bias_module.bias[:pos_dim] = pos_encoding[:pos_dim] * 0.1

    def _setup_malicious_lora(self):
        """Setup malicious LoRA parameters"""
        with torch.no_grad():
            for name, module in self.global_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A
                    lora_B = module.lora_B

                    # Set A matrix to extract input patterns
                    nn.init.normal_(lora_A, mean=0, std=0.02)

                    # Set B matrix for identity-like transformation
                    nn.init.zeros_(lora_B)
                    min_dim = min(lora_B.shape[0], lora_B.shape[1], lora_A.shape[0])
                    for i in range(min_dim):
                        lora_B[i % lora_B.shape[0], i] = 0.1
