import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod #aggregation.py
import logging


class BaseAggregator(ABC):
    """Base class for gradient aggregation methods"""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize base aggregator

        Args:
            device: Computing device
        """
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # Aggregation statistics
        self.aggregation_stats = {
            'total_aggregations': 0,
            'client_participation': [],
            'gradient_norms': [],
            'aggregation_time': []
        }

    @abstractmethod
    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients from multiple clients

        Args:
            gradients_list: List of gradient dictionaries from clients
            weights: List of weights for each client (e.g., number of samples)

        Returns:
            Aggregated gradients
        """
        pass

    def _validate_inputs(self, gradients_list: List[Dict[str, torch.Tensor]],
                         weights: List[float]) -> bool:
        """Validate aggregation inputs"""
        if not gradients_list:
            self.logger.error("Empty gradients list provided")
            return False

        if len(gradients_list) != len(weights):
            self.logger.error("Mismatch between gradients and weights length")
            return False

        if any(w <= 0 for w in weights):
            self.logger.error("All weights must be positive")
            return False

        # Check gradient dictionary consistency
        if len(gradients_list) > 1:
            first_keys = set(gradients_list[0].keys())
            for i, grad_dict in enumerate(gradients_list[1:], 1):
                if set(grad_dict.keys()) != first_keys:
                    self.logger.error(f"Gradient keys mismatch at client {i}")
                    return False

        return True

    def _compute_gradient_norms(self, gradients_list: List[Dict[str, torch.Tensor]]) -> List[float]:
        """Compute L2 norms of client gradients"""
        norms = []

        for grad_dict in gradients_list:
            total_norm = 0.0
            for grad in grad_dict.values():
                if isinstance(grad, torch.Tensor):
                    total_norm += torch.norm(grad).item() ** 2
            norms.append(np.sqrt(total_norm))

        return norms

    def _update_stats(self, gradients_list: List[Dict[str, torch.Tensor]],
                      weights: List[float]):
        """Update aggregation statistics"""
        self.aggregation_stats['total_aggregations'] += 1
        self.aggregation_stats['client_participation'].append(len(gradients_list))

        gradient_norms = self._compute_gradient_norms(gradients_list)
        self.aggregation_stats['gradient_norms'].append(gradient_norms)


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator"""

    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.logger.info("FedAvg aggregator initialized")

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients using weighted averaging (FedAvg)

        Weight each client's gradient by their number of samples
        """
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for FedAvg aggregation")

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Initialize aggregated gradients
        aggregated_gradients = {}

        # Get parameter names from first client
        param_names = list(gradients_list[0].keys())

        for param_name in param_names:
            # Weighted sum of gradients for this parameter
            weighted_sum = None

            for grad_dict, weight in zip(gradients_list, normalized_weights):
                if param_name in grad_dict:
                    grad = grad_dict[param_name].to(self.device)

                    if weighted_sum is None:
                        weighted_sum = weight * grad
                    else:
                        weighted_sum += weight * grad

            if weighted_sum is not None:
                aggregated_gradients[param_name] = weighted_sum

        self._update_stats(gradients_list, weights)

        self.logger.info(f"FedAvg aggregation completed for {len(gradients_list)} clients")

        return aggregated_gradients


class WeightedAggregator(BaseAggregator):
    """Weighted aggregator with custom weighting schemes"""

    def __init__(self, weighting_scheme: str = 'sample_size', device: str = 'cuda'):
        """
        Initialize weighted aggregator

        Args:
            weighting_scheme: How to weight clients ('sample_size', 'uniform', 'inverse_loss')
            device: Computing device
        """
        super().__init__(device)
        self.weighting_scheme = weighting_scheme
        self.logger.info(f"Weighted aggregator initialized with {weighting_scheme} weighting")

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate with custom weighting scheme"""
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for weighted aggregation")

        # Compute weights based on scheme
        if self.weighting_scheme == 'sample_size':
            client_weights = weights  # Use provided weights (sample sizes)
        elif self.weighting_scheme == 'uniform':
            client_weights = [1.0] * len(gradients_list)  # Equal weights
        elif self.weighting_scheme == 'inverse_loss':
            # Weight inversely to loss (higher loss = lower weight)
            # This requires loss values in weights
            max_loss = max(weights)
            client_weights = [max_loss - w + 0.1 for w in weights]  # +0.1 to avoid zero
        else:
            self.logger.warning(f"Unknown weighting scheme: {self.weighting_scheme}, using sample_size")
            client_weights = weights

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Aggregate gradients
        aggregated_gradients = {}
        param_names = list(gradients_list[0].keys())

        for param_name in param_names:
            weighted_sum = None

            for grad_dict, weight in zip(gradients_list, normalized_weights):
                if param_name in grad_dict:
                    grad = grad_dict[param_name].to(self.device)

                    if weighted_sum is None:
                        weighted_sum = weight * grad
                    else:
                        weighted_sum += weight * grad

            if weighted_sum is not None:
                aggregated_gradients[param_name] = weighted_sum

        self._update_stats(gradients_list, weights)

        return aggregated_gradients


class MedianAggregator(BaseAggregator):
    """Coordinate-wise median aggregator for Byzantine robustness"""

    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.logger.info("Median aggregator initialized")

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using coordinate-wise median

        Robust against Byzantine clients but ignores sample size weights
        """
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for median aggregation")

        aggregated_gradients = {}
        param_names = list(gradients_list[0].keys())

        for param_name in param_names:
            # Collect all gradients for this parameter
            param_gradients = []

            for grad_dict in gradients_list:
                if param_name in grad_dict:
                    grad = grad_dict[param_name].to(self.device)
                    param_gradients.append(grad)

            if param_gradients:
                # Stack gradients along new dimension
                stacked_gradients = torch.stack(param_gradients, dim=0)

                # Compute coordinate-wise median
                median_grad = torch.median(stacked_gradients, dim=0)[0]
                aggregated_gradients[param_name] = median_grad

        self._update_stats(gradients_list, weights)

        self.logger.info(f"Median aggregation completed for {len(gradients_list)} clients")

        return aggregated_gradients


class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed mean aggregator for Byzantine robustness"""

    def __init__(self, trim_ratio: float = 0.2, device: str = 'cuda'):
        """
        Initialize trimmed mean aggregator

        Args:
            trim_ratio: Fraction of extreme values to trim (0-0.5)
            device: Computing device
        """
        super().__init__(device)
        self.trim_ratio = min(0.5, max(0.0, trim_ratio))  # Clamp to valid range
        self.logger.info(f"Trimmed mean aggregator initialized with trim_ratio={self.trim_ratio}")

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using trimmed mean

        Remove extreme values and average the remaining
        """
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for trimmed mean aggregation")

        num_clients = len(gradients_list)
        num_trim = int(num_clients * self.trim_ratio)

        if num_clients - 2 * num_trim <= 0:
            self.logger.warning("Too many clients to trim, using regular mean")
            return FedAvgAggregator(self.device).aggregate(gradients_list, weights)

        aggregated_gradients = {}
        param_names = list(gradients_list[0].keys())

        for param_name in param_names:
            # Collect gradients for this parameter
            param_gradients = []

            for grad_dict in gradients_list:
                if param_name in grad_dict:
                    grad = grad_dict[param_name].to(self.device)
                    param_gradients.append(grad)

            if param_gradients:
                # Stack gradients
                stacked_gradients = torch.stack(param_gradients, dim=0)

                # Sort along client dimension and trim extremes
                sorted_gradients, _ = torch.sort(stacked_gradients, dim=0)

                # Keep middle values
                start_idx = num_trim
                end_idx = num_clients - num_trim
                trimmed_gradients = sorted_gradients[start_idx:end_idx]

                # Compute mean of trimmed values
                mean_grad = torch.mean(trimmed_gradients, dim=0)
                aggregated_gradients[param_name] = mean_grad

        self._update_stats(gradients_list, weights)

        self.logger.info(f"Trimmed mean aggregation completed: "
                         f"trimmed {2 * num_trim}/{num_clients} extreme clients")

        return aggregated_gradients


class KrumAggregator(BaseAggregator):
    """Krum aggregator for Byzantine robustness"""

    def __init__(self, num_byzantine: int = 1, device: str = 'cuda'):
        """
        Initialize Krum aggregator

        Args:
            num_byzantine: Expected number of Byzantine clients
            device: Computing device
        """
        super().__init__(device)
        self.num_byzantine = num_byzantine
        self.logger.info(f"Krum aggregator initialized with {num_byzantine} expected Byzantine clients")

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using Krum algorithm

        Select gradient that minimizes sum of squared distances to closest gradients
        """
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for Krum aggregation")

        num_clients = len(gradients_list)

        if num_clients <= 2 * self.num_byzantine:
            self.logger.warning("Not enough clients for Krum, using FedAvg")
            return FedAvgAggregator(self.device).aggregate(gradients_list, weights)

        # Flatten all gradients for distance computation
        flattened_gradients = []

        for grad_dict in gradients_list:
            flattened = []
            for param_name in sorted(grad_dict.keys()):  # Ensure consistent ordering
                grad = grad_dict[param_name].to(self.device)
                flattened.append(grad.flatten())

            client_flattened = torch.cat(flattened)
            flattened_gradients.append(client_flattened)

        # Stack all flattened gradients
        all_gradients = torch.stack(flattened_gradients)  # [num_clients, total_params]

        # Compute pairwise distances
        distances = torch.cdist(all_gradients, all_gradients, p=2)  # L2 distances

        # For each client, find sum of distances to closest clients
        num_closest = num_clients - self.num_byzantine - 1
        krum_scores = []

        for i in range(num_clients):
            # Get distances from client i to all others
            client_distances = distances[i]

            # Remove distance to self (which is 0)
            other_distances = torch.cat([client_distances[:i], client_distances[i + 1:]])

            # Sum of distances to closest clients
            closest_distances, _ = torch.topk(other_distances, num_closest, largest=False)
            krum_score = torch.sum(closest_distances)
            krum_scores.append(krum_score)

        # Select client with minimum Krum score
        selected_client_idx = torch.argmin(torch.stack(krum_scores))
        selected_gradients = gradients_list[selected_client_idx]

        self._update_stats(gradients_list, weights)

        self.logger.info(f"Krum aggregation: selected client {selected_client_idx} out of {num_clients}")

        return selected_gradients


class FedProxAggregator(BaseAggregator):
    """FedProx aggregator with proximal term"""

    def __init__(self, mu: float = 0.01, device: str = 'cuda'):
        """
        Initialize FedProx aggregator

        Args:
            mu: Proximal term coefficient
            device: Computing device
        """
        super().__init__(device)
        self.mu = mu
        self.global_model_params = None
        self.logger.info(f"FedProx aggregator initialized with mu={mu}")

    def set_global_model_params(self, global_params: Dict[str, torch.Tensor]):
        """Set global model parameters for proximal term"""
        self.global_model_params = {k: v.clone() for k, v in global_params.items()}

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate with FedProx proximal term

        Adds proximal term to prevent client drift
        """
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for FedProx aggregation")

        # Start with FedAvg aggregation
        fedavg_aggregator = FedAvgAggregator(self.device)
        aggregated_gradients = fedavg_aggregator.aggregate(gradients_list, weights)

        # Add proximal term if global model parameters are available
        if self.global_model_params is not None:
            for param_name in aggregated_gradients:
                if param_name in self.global_model_params:
                    # Add proximal regularization: μ * (w - w_global)
                    proximal_term = self.mu * self.global_model_params[param_name]
                    aggregated_gradients[param_name] += proximal_term

        self._update_stats(gradients_list, weights)

        self.logger.info(f"FedProx aggregation completed with proximal term (mu={self.mu})")

        return aggregated_gradients


class AdaptiveAggregator(BaseAggregator):
    """Adaptive aggregator that selects aggregation method based on conditions"""

    def __init__(self, device: str = 'cuda',
                 byzantine_threshold: float = 0.5,
                 variance_threshold: float = 1.0):
        """
        Initialize adaptive aggregator

        Args:
            device: Computing device
            byzantine_threshold: Threshold for detecting Byzantine behavior
            variance_threshold: Threshold for gradient variance
        """
        super().__init__(device)
        self.byzantine_threshold = byzantine_threshold
        self.variance_threshold = variance_threshold

        # Initialize sub-aggregators
        self.fedavg = FedAvgAggregator(device)
        self.median = MedianAggregator(device)
        self.trimmed_mean = TrimmedMeanAggregator(device=device)

        self.logger.info("Adaptive aggregator initialized")

    def aggregate(self, gradients_list: List[Dict[str, torch.Tensor]],
                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Adaptively select aggregation method based on gradient analysis
        """
        if not self._validate_inputs(gradients_list, weights):
            raise ValueError("Invalid inputs for adaptive aggregation")

        # Analyze gradient patterns
        analysis = self._analyze_gradients(gradients_list)

        # Select appropriate aggregation method
        if analysis['suspected_byzantine_ratio'] > self.byzantine_threshold:
            self.logger.info("High Byzantine activity detected, using Median aggregation")
            selected_aggregator = self.median
        elif analysis['gradient_variance'] > self.variance_threshold:
            self.logger.info("High gradient variance detected, using Trimmed Mean aggregation")
            selected_aggregator = self.trimmed_mean
        else:
            self.logger.info("Normal conditions detected, using FedAvg aggregation")
            selected_aggregator = self.fedavg

        # Perform aggregation
        aggregated_gradients = selected_aggregator.aggregate(gradients_list, weights)

        self._update_stats(gradients_list, weights)

        return aggregated_gradients

    def _analyze_gradients(self, gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Analyze gradient patterns to detect anomalies"""
        analysis = {}

        # Compute gradient norms
        gradient_norms = self._compute_gradient_norms(gradients_list)

        # Calculate variance in gradient norms
        norm_variance = np.var(gradient_norms) if len(gradient_norms) > 1 else 0.0
        analysis['gradient_variance'] = norm_variance

        # Detect potential Byzantine clients
        if len(gradient_norms) > 1:
            mean_norm = np.mean(gradient_norms)
            std_norm = np.std(gradient_norms)

            # Count clients with extreme gradient norms
            byzantine_count = 0
            for norm in gradient_norms:
                if abs(norm - mean_norm) > 2 * std_norm:  # 2-sigma rule
                    byzantine_count += 1

            analysis['suspected_byzantine_ratio'] = byzantine_count / len(gradient_norms)
        else:
            analysis['suspected_byzantine_ratio'] = 0.0

        return analysis


# Factory function for creating aggregators
def create_aggregator(aggregator_type: str, device: str = 'cuda', **kwargs) -> BaseAggregator:
    """
    Create aggregator instance

    Args:
        aggregator_type: Type of aggregator
        device: Computing device
        **kwargs: Additional arguments for specific aggregators

    Returns:
        Aggregator instance
    """

    aggregator_types = {
        'fedavg': FedAvgAggregator,
        'weighted': WeightedAggregator,
        'median': MedianAggregator,
        'trimmed_mean': TrimmedMeanAggregator,
        'krum': KrumAggregator,
        'fedprox': FedProxAggregator,
        'adaptive': AdaptiveAggregator
    }

    if aggregator_type not in aggregator_types:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}. "
                         f"Available: {list(aggregator_types.keys())}")

    aggregator_class = aggregator_types[aggregator_type]
    return aggregator_class(device=device, **kwargs)


# Example usage and testing
if __name__ == "__main__":

    # Test different aggregators
    aggregator_types = ['fedavg', 'median', 'trimmed_mean', 'adaptive']

    # Create dummy gradients from 5 clients
    dummy_gradients = []
    weights = [100, 200, 150, 80, 120]  # Sample sizes

    for i in range(5):
        client_gradients = {
            'layer1.weight': torch.randn(10, 5) * (0.5 + i * 0.1),  # Different scales
            'layer1.bias': torch.randn(10) * (0.3 + i * 0.05),
            'layer2.weight': torch.randn(1, 10) * (0.4 + i * 0.08)
        }
        dummy_gradients.append(client_gradients)

    print("Testing aggregation methods:")

    for agg_type in aggregator_types:
        print(f"\nTesting {agg_type.upper()} aggregator:")

        try:
            aggregator = create_aggregator(agg_type, device='cpu')
            aggregated = aggregator.aggregate(dummy_gradients, weights)

            print(f"  ✓ {aggregator.__class__.__name__} completed successfully")
            print(f"  Aggregated {len(aggregated)} parameters")

            # Check aggregated gradient norms
            total_norm = 0.0
            for grad in aggregated.values():
                total_norm += torch.norm(grad).item() ** 2
            total_norm = np.sqrt(total_norm)

            print(f"  Aggregated gradient norm: {total_norm:.4f}")

        except Exception as e:
            print(f"  ✗ Error with {agg_type}: {e}")

    print("\nAggregation testing completed!")
