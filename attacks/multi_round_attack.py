import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any #multi_round_attack.py
from attacks.attack_base import BaseGradientInversionAttack
from attacks.adapter_attack import AdapterGradientInversionAttack
from attacks.prefix_attack import PrefixGradientInversionAttack
from attacks.bias_attack import BiasGradientInversionAttack
from attacks.lora_attack import LoRAGradientInversionAttack


class MultiRoundFederatedAttack:
    """Multi-round gradient inversion attack for federated learning scenarios"""

    def __init__(self, peft_method: str, device: str = 'cuda', **attack_kwargs):
        """
        Initialize multi-round federated attack

        Args:
            peft_method: PEFT method ('adapter', 'prefix', 'bias', 'lora')
            device: Computing device
            **attack_kwargs: Arguments for specific attack class
        """
        self.peft_method = peft_method
        self.device = device
        self.attack_kwargs = attack_kwargs

        # Initialize attack class based on PEFT method
        self.attack_class = self._get_attack_class(peft_method)

        # Multi-round configuration
        self.fl_config = {
            'num_rounds': 10,
            'num_clients': 5,
            'local_epochs': 1,
            'aggregation_method': 'fedavg',
            'attack_frequency': 1,  # Attack every N rounds
            'patience': 3,  # Stop if no improvement for N rounds
            'min_improvement': 0.01  # Minimum PSNR improvement to continue
        }

        # Results tracking
        self.round_results = {}
        self.cumulative_patches = []
        self.best_round = 0
        self.best_psnr = 0.0

    def _get_attack_class(self, peft_method: str) -> type:
        """Get appropriate attack class for PEFT method"""
        attack_classes = {
            'adapter': AdapterGradientInversionAttack,
            'prefix': PrefixGradientInversionAttack,
            'bias': BiasGradientInversionAttack,
            'lora': LoRAGradientInversionAttack
        }

        if peft_method not in attack_classes:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")

        return attack_classes[peft_method]

    def run_multi_round_attack(self, client_models: List[nn.Module],
                               server_model: nn.Module,
                               client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                               target_patches: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Run multi-round federated learning attack

        Args:
            client_models: List of client models
            server_model: Server model (potentially malicious)
            client_data: List of (data, labels) tuples for each client
            target_patches: Ground truth patches for evaluation

        Returns:
            Multi-round attack results
        """
        print(f"Starting multi-round {self.peft_method.upper()} attack...")
        print(f"Configuration: {self.fl_config['num_rounds']} rounds, {len(client_models)} clients")

        all_recovered_patches = []
        round_metrics = []
        patience_counter = 0

        for round_num in range(self.fl_config['num_rounds']):
            print(f"\n=== Round {round_num + 1}/{self.fl_config['num_rounds']} ===")

            # Simulate federated learning round
            round_gradients = self._simulate_fl_round(client_models, server_model, client_data)

            # Perform attack if it's an attack round
            if (round_num + 1) % self.fl_config['attack_frequency'] == 0:
                round_attack_results = self._attack_round(server_model, round_gradients,
                                                          target_patches, round_num)

                # Update cumulative results
                all_recovered_patches.extend(round_attack_results['recovered_patches'])
                round_metrics.append(round_attack_results['metrics'])

                # Check for improvement
                current_psnr = round_attack_results['metrics'].get('psnr', 0.0)
                if current_psnr > self.best_psnr + self.fl_config['min_improvement']:
                    self.best_psnr = current_psnr
                    self.best_round = round_num
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.fl_config['patience']:
                    print(f"Early stopping at round {round_num + 1} (no improvement for {patience_counter} rounds)")
                    break

            # Update server model (simulate aggregation)
            self._update_server_model(server_model, round_gradients)

        # Compile final results
        final_results = self._compile_final_results(all_recovered_patches, round_metrics,
                                                    target_patches)

        return final_results

    def _simulate_fl_round(self, client_models: List[nn.Module],
                           server_model: nn.Module,
                           client_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Simulate one round of federated learning"""

        # Distribute server model to clients
        for client_model in client_models:
            self._copy_peft_parameters(server_model, client_model)

        # Collect gradients from clients
        client_gradients = {}

        for client_id, (client_model, (data, labels)) in enumerate(zip(client_models, client_data)):
            # Client local training
            client_model.train()
            client_model.zero_grad()

            # Forward pass
            outputs, _ = client_model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass
            loss.backward()

            # Extract gradients
            client_grads = {}
            for name, param in client_model.named_parameters():
                if self._is_peft_parameter(name) and param.grad is not None:
                    client_grads[name] = param.grad.clone().detach()

            client_gradients[f'client_{client_id}'] = client_grads

        return client_gradients

    def _attack_round(self, server_model: nn.Module,
                      round_gradients: Dict[str, Dict[str, torch.Tensor]],
                      target_patches: Optional[List[torch.Tensor]],
                      round_num: int) -> Dict[str, Any]:
        """Perform attack for current round"""

        # Aggregate gradients (FedAvg)
        aggregated_gradients = self._aggregate_gradients(round_gradients)

        # Initialize attack for this round
        round_attacker = self.attack_class(server_model, self.device, **self.attack_kwargs)

        # Setup malicious parameters
        round_attacker.setup_malicious_parameters()

        # Simulate attack using aggregated gradients
        mock_data = torch.randn(1, 3, 32, 32, device=self.device)  # Dummy data for gradient extraction
        mock_labels = torch.randint(0, 100, (1,), device=self.device)  # Dummy labels

        # Inject aggregated gradients into attack
        attack_results = self._attack_with_aggregated_gradients(round_attacker, aggregated_gradients,
                                                                target_patches)

        # Add round information
        attack_results['round'] = round_num
        attack_results['num_clients'] = len(round_gradients)

        print(f"Round {round_num + 1} attack results:")
        if 'metrics' in attack_results:
            metrics = attack_results['metrics']
            print(f"  PSNR: {metrics.get('psnr', 0.0):.2f} dB")
            print(f"  Recovered patches: {len(attack_results.get('recovered_patches', []))}")

        return attack_results

    def _aggregate_gradients(self, client_gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate gradients using FedAvg"""
        aggregated = {}

        # Get parameter names from first client
        if client_gradients:
            first_client = list(client_gradients.values())[0]
            param_names = list(first_client.keys())

            # Average gradients across clients
            for param_name in param_names:
                param_grads = []

                for client_id, client_grads in client_gradients.items():
                    if param_name in client_grads:
                        param_grads.append(client_grads[param_name])

                if param_grads:
                    # Simple average (FedAvg)
                    aggregated[param_name] = torch.mean(torch.stack(param_grads), dim=0)

        return aggregated

    def _attack_with_aggregated_gradients(self, attacker: BaseGradientInversionAttack,
                                          aggregated_gradients: Dict[str, torch.Tensor],
                                          target_patches: Optional[List[torch.Tensor]]) -> Dict[str, Any]:
        """Perform attack using aggregated gradients"""

        # Inject aggregated gradients into attacker
        attacker.attack_results['gradients'] = aggregated_gradients

        # Reconstruct patches from aggregated gradients
        recovered_patches = attacker.reconstruct_patches(aggregated_gradients)
        attacker.attack_results['recovered_patches'] = recovered_patches

        # Evaluate if target patches provided
        if target_patches:
            attacker.attack_results['original_patches'] = target_patches
            metrics = attacker.evaluate_attack_success(target_patches, recovered_patches)
            attacker.attack_results['metrics'] = metrics
        else:
            attacker.attack_results['metrics'] = {'psnr': 0.0, 'ssim': 0.0, 'mse': 0.0}

        return attacker.attack_results

    def _copy_peft_parameters(self, source_model: nn.Module, target_model: nn.Module):
        """Copy PEFT parameters from source to target model"""
        with torch.no_grad():
            for (src_name, src_param), (tgt_name, tgt_param) in zip(
                    source_model.named_parameters(), target_model.named_parameters()):
                if self._is_peft_parameter(src_name) and src_name == tgt_name:
                    tgt_param.data.copy_(src_param.data)

    def _is_peft_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to PEFT method"""
        peft_indicators = {
            'adapter': ['peft', 'adapter'],
            'prefix': ['prefix'],
            'bias': ['bias'],
            'lora': ['lora_']
        }

        indicators = peft_indicators.get(self.peft_method, [])
        return any(indicator in param_name.lower() for indicator in indicators)

    def _update_server_model(self, server_model: nn.Module,
                             round_gradients: Dict[str, Dict[str, torch.Tensor]]):
        """Update server model with aggregated gradients"""
        aggregated_gradients = self._aggregate_gradients(round_gradients)

        with torch.no_grad():
            for name, param in server_model.named_parameters():
                if name in aggregated_gradients and self._is_peft_parameter(name):
                    # Simple gradient descent update
                    param.data -= 0.01 * aggregated_gradients[name]  # Learning rate = 0.01

    def _compile_final_results(self, all_recovered_patches: List[torch.Tensor],
                               round_metrics: List[Dict[str, float]],
                               target_patches: Optional[List[torch.Tensor]]) -> Dict[str, Any]:
        """Compile final multi-round attack results"""

        final_results = {
            'peft_method': self.peft_method,
            'total_rounds': len(round_metrics),
            'total_recovered_patches': len(all_recovered_patches),
            'best_round': self.best_round,
            'best_psnr': self.best_psnr,
            'round_metrics': round_metrics,
            'recovered_patches': all_recovered_patches,
            'config': self.fl_config
        }

        # Overall metrics
        if round_metrics:
            avg_psnr = np.mean([m.get('psnr', 0.0) for m in round_metrics])
            avg_ssim = np.mean([m.get('ssim', 0.0) for m in round_metrics])
            max_psnr = np.max([m.get('psnr', 0.0) for m in round_metrics])

            final_results['overall_metrics'] = {
                'average_psnr': avg_psnr,
                'average_ssim': avg_ssim,
                'maximum_psnr': max_psnr,
                'improvement_over_rounds': max_psnr - round_metrics[0].get('psnr', 0.0) if round_metrics else 0.0
            }

        # Final evaluation against target patches
        if target_patches and all_recovered_patches:
            final_metrics = self._evaluate_cumulative_attack(all_recovered_patches, target_patches)
            final_results['cumulative_metrics'] = final_metrics

        print(f"\n=== Multi-Round Attack Summary ===")
        print(f"PEFT Method: {self.peft_method.upper()}")
        print(f"Total Rounds: {final_results['total_rounds']}")
        print(f"Total Patches Recovered: {final_results['total_recovered_patches']}")
        print(f"Best Round: {final_results['best_round'] + 1}")
        print(f"Best PSNR: {final_results['best_psnr']:.2f} dB")

        if 'overall_metrics' in final_results:
            overall = final_results['overall_metrics']
            print(f"Average PSNR: {overall['average_psnr']:.2f} dB")
            print(f"Maximum PSNR: {overall['maximum_psnr']:.2f} dB")
            print(f"Improvement: {overall['improvement_over_rounds']:.2f} dB")

        return final_results

    def _evaluate_cumulative_attack(self, recovered_patches: List[torch.Tensor],
                                    target_patches: List[torch.Tensor]) -> Dict[str, float]:
        """Evaluate cumulative attack performance across all rounds"""
        # Remove duplicates and select best patches
        unique_patches = self._remove_duplicate_patches(recovered_patches)

        # Evaluate against targets
        if not unique_patches or not target_patches:
            return {'psnr': 0.0, 'ssim': 0.0, 'mse': float('inf')}

        psnr_values = []
        ssim_values = []
        mse_values = []

        min_len = min(len(unique_patches), len(target_patches))

        for i in range(min_len):
            # Compute metrics
            psnr = self._compute_psnr(target_patches[i], unique_patches[i])
            ssim = self._compute_ssim(target_patches[i], unique_patches[i])
            mse = torch.mean((target_patches[i] - unique_patches[i]) ** 2).item()

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)

        return {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'mse': np.mean(mse_values),
            'coverage': min_len / len(target_patches) if target_patches else 0.0
        }

    def _remove_duplicate_patches(self, patches: List[torch.Tensor]) -> List[torch.Tensor]:
        """Remove duplicate patches based on similarity"""
        if not patches:
            return []

        unique_patches = [patches[0]]
        similarity_threshold = 0.95

        for patch in patches[1:]:
            is_duplicate = False

            for unique_patch in unique_patches:
                if patch.shape == unique_patch.shape:
                    similarity = self._compute_ssim(patch, unique_patch)
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_patches.append(patch)

        return unique_patches

    def _compute_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute PSNR between two patches"""
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def _compute_ssim(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute simplified SSIM between two patches"""
        mu1 = torch.mean(original)
        mu2 = torch.mean(reconstructed)

        sigma1_sq = torch.var(original)
        sigma2_sq = torch.var(reconstructed)
        sigma12 = torch.mean((original - mu1) * (reconstructed - mu2))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / denominator
        return ssim.item()
