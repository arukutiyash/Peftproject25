import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset #fl_client.py
import copy
import logging
from typing import Dict, List, Tuple, Optional, Any, Union


class FederatedClient:
    """Federated Learning Client"""

    def __init__(self, client_id: str, model: nn.Module, train_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None, device: str = 'cuda',
                 peft_method: str = 'adapter'):
        """
        Initialize FL Client

        Args:
            client_id: Unique client identifier
            model: Local model (copy of global model)
            train_loader: Training data loader
            test_loader: Optional test data loader
            device: Computing device
            peft_method: PEFT method being used
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.peft_method = peft_method

        # Training configuration
        self.learning_rate = 0.01
        self.local_epochs = 1

        # Client metrics
        self.training_history = []
        self.current_round = 0

        # Setup logging
        self.logger = logging.getLogger(f"Client_{client_id}")

        # Defense mechanism
        self.defense = None

    def set_defense(self, defense_mechanism):
        """Set defense mechanism for privacy protection"""
        self.defense = defense_mechanism
        self.logger.info(f"Defense mechanism set: {defense_mechanism.__class__.__name__}")

    def update_model(self, global_parameters: Dict[str, torch.Tensor]):
        """Update local model with global parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_parameters:
                    param.data.copy_(global_parameters[name])

        self.logger.info("Model updated with global parameters")

    def local_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform local training

        Args:
            epochs: Number of local epochs (overrides default)

        Returns:
            Training results and gradients
        """
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()

        # Setup optimizer (only for PEFT parameters)
        peft_params = self._get_peft_parameters()
        optimizer = optim.SGD(peft_params, lr=self.learning_rate)

        # Training metrics
        total_loss = 0.0
        total_samples = 0
        batch_count = 0

        self.logger.info(f"Starting local training for {epochs} epochs")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

                # Track metrics
                batch_loss = loss.item()
                batch_size = data.size(0)

                epoch_loss += batch_loss * batch_size
                epoch_samples += batch_size
                total_loss += batch_loss * batch_size
                total_samples += batch_size
                batch_count += 1

        # Compute average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Extract gradients after training
        gradients = self._extract_gradients()

        # Apply defense if available
        if self.defense:
            defended_gradients = self.defense.apply_defense_with_validation(
                gradients, batch_data=data, model=self.model
            )
            gradients = defended_gradients

        # Training results
        training_results = {
            'client_id': self.client_id,
            'round': self.current_round,
            'epochs': epochs,
            'total_samples': total_samples,
            'avg_loss': avg_loss,
            'gradients': gradients,
            'defense_applied': self.defense.__class__.__name__ if self.defense else None
        }

        self.training_history.append(training_results)

        self.logger.info(f"Local training completed. Avg Loss: {avg_loss:.4f}, Samples: {total_samples}")

        return training_results

    def _get_peft_parameters(self) -> List[torch.nn.Parameter]:
        """Get PEFT-specific parameters for training"""
        peft_params = []

        if self.peft_method == 'adapter':
            for name, param in self.model.named_parameters():
                if 'peft' in name.lower() and param.requires_grad:
                    peft_params.append(param)

        elif self.peft_method == 'prefix':
            for name, param in self.model.named_parameters():
                if 'prefix' in name.lower() and param.requires_grad:
                    peft_params.append(param)

        elif self.peft_method == 'bias':
            bias_indicators = ['attn_qkv_bias', 'attn_proj_bias', 'mlp_fc1_bias', 'mlp_fc2_bias', 'norm1_bias',
                               'norm2_bias']
            for name, param in self.model.named_parameters():
                if any(indicator in name for indicator in bias_indicators) and param.requires_grad:
                    peft_params.append(param)

        elif self.peft_method == 'lora':
            for name, param in self.model.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    peft_params.append(param)

        return peft_params

    def _extract_gradients(self) -> Dict[str, torch.Tensor]:
        """Extract gradients from PEFT parameters"""
        gradients = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and self._is_peft_parameter(name):
                gradients[name] = param.grad.clone().detach()

        return gradients

    def _is_peft_parameter(self, param_name: str) -> bool:
        """Check if parameter belongs to PEFT method"""
        if self.peft_method == 'adapter':
            return 'peft' in param_name.lower()
        elif self.peft_method == 'prefix':
            return 'prefix' in param_name.lower()
        elif self.peft_method == 'bias':
            bias_indicators = ['attn_qkv_bias', 'attn_proj_bias', 'mlp_fc1_bias', 'mlp_fc2_bias', 'norm1_bias',
                               'norm2_bias']
            return any(indicator in param_name for indicator in bias_indicators)
        elif self.peft_method == 'lora':
            return 'lora_' in param_name

        return False

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate local model performance"""
        if self.test_loader is None:
            return {'message': 'No test data available'}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs, _ = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state"""
        return {name: param.clone().detach()
                for name, param in self.model.named_parameters()}

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get client training history"""
        return self.training_history

    def reset_model(self, global_parameters: Dict[str, torch.Tensor]):
        """Reset model to global parameters and clear gradients"""
        self.update_model(global_parameters)
        self.model.zero_grad()


class DataHeterogeneousClient(FederatedClient):
    """Client with heterogeneous data distribution"""

    def __init__(self, client_id: str, model: nn.Module,
                 full_dataset, class_distribution: Dict[int, float],
                 batch_size: int = 32, **kwargs):
        """
        Initialize client with specific class distribution

        Args:
            client_id: Client identifier
            model: Local model
            full_dataset: Complete dataset
            class_distribution: Dictionary mapping class_id to probability
            batch_size: Batch size for data loading
            **kwargs: Additional arguments
        """

        # Create heterogeneous data subset
        train_indices = self._create_heterogeneous_subset(
            full_dataset, class_distribution
        )

        train_subset = Subset(full_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        super().__init__(client_id, model, train_loader, **kwargs)

        self.class_distribution = class_distribution
        self.data_size = len(train_indices)

        self.logger.info(f"Client {client_id} initialized with {self.data_size} samples")
        self.logger.info(f"Class distribution: {class_distribution}")

    def _create_heterogeneous_subset(self, dataset, class_distribution: Dict[int, float]) -> List[int]:
        """Create subset based on class distribution"""
        from collections import defaultdict

        # Group indices by class
        class_indices = defaultdict(list)

        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        # Sample according to distribution
        selected_indices = []

        for class_id, probability in class_distribution.items():
            if class_id in class_indices and probability > 0:
                class_size = len(class_indices[class_id])
                num_samples = int(class_size * probability)

                # Randomly sample from this class
                if num_samples > 0:
                    import random
                    sampled_indices = random.sample(class_indices[class_id],
                                                    min(num_samples, class_size))
                    selected_indices.extend(sampled_indices)

        return selected_indices

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of client's data distribution"""
        class_counts = {}
        total_samples = 0

        for _, labels in self.train_loader:
            for label in labels:
                class_id = label.item()
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_samples += 1

        # Calculate actual distribution
        actual_distribution = {k: v / total_samples for k, v in class_counts.items()}

        return {
            'client_id': self.client_id,
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'target_distribution': self.class_distribution,
            'actual_distribution': actual_distribution,
            'class_counts': class_counts
        }


class AdversarialClient(FederatedClient):
    """Client that may perform adversarial actions"""

    def __init__(self, client_id: str, model: nn.Module, train_loader: DataLoader,
                 adversarial_type: str = 'byzantine', **kwargs):
        """
        Initialize adversarial client

        Args:
            client_id: Client identifier
            model: Local model
            train_loader: Training data
            adversarial_type: Type of adversarial behavior
            **kwargs: Additional arguments
        """
        super().__init__(client_id, model, train_loader, **kwargs)

        self.adversarial_type = adversarial_type
        self.adversarial_probability = 0.3  # Probability of adversarial action

        self.logger.info(f"Adversarial client initialized: {adversarial_type}")

    def local_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Perform local training with potential adversarial behavior"""

        # Decide whether to act adversarially this round
        import random
        act_adversarially = random.random() < self.adversarial_probability

        if act_adversarially:
            return self._adversarial_training(epochs)
        else:
            return super().local_training(epochs)

    def _adversarial_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Perform adversarial training"""

        if self.adversarial_type == 'byzantine':
            return self._byzantine_training(epochs)
        elif self.adversarial_type == 'model_poisoning':
            return self._model_poisoning_training(epochs)
        elif self.adversarial_type == 'gradient_noise':
            return self._gradient_noise_training(epochs)
        else:
            self.logger.warning(f"Unknown adversarial type: {self.adversarial_type}")
            return super().local_training(epochs)

    def _byzantine_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Byzantine behavior: send random gradients"""
        # Perform normal training first
        results = super().local_training(epochs)

        # Replace gradients with random values
        random_gradients = {}
        for name, grad in results['gradients'].items():
            random_gradients[name] = torch.randn_like(grad) * 0.1

        results['gradients'] = random_gradients
        results['adversarial_action'] = 'byzantine_gradients'

        self.logger.info("Applied Byzantine attack: random gradients")

        return results

    def _model_poisoning_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Model poisoning: amplify gradients"""
        results = super().local_training(epochs)

        # Amplify gradients
        amplification_factor = 5.0
        amplified_gradients = {}

        for name, grad in results['gradients'].items():
            amplified_gradients[name] = grad * amplification_factor

        results['gradients'] = amplified_gradients
        results['adversarial_action'] = 'gradient_amplification'

        self.logger.info("Applied model poisoning: gradient amplification")

        return results

    def _gradient_noise_training(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Add noise to gradients"""
        results = super().local_training(epochs)

        # Add noise to gradients
        noise_scale = 0.5
        noisy_gradients = {}

        for name, grad in results['gradients'].items():
            noise = torch.randn_like(grad) * noise_scale
            noisy_gradients[name] = grad + noise

        results['gradients'] = noisy_gradients
        results['adversarial_action'] = 'gradient_noise'

        self.logger.info("Applied gradient noise attack")

        return results
