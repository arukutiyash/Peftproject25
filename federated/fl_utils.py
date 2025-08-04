import torch
import torch.nn as nn
import numpy as np #fl_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
import os
from datetime import datetime
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class FLDataManager:
    """Utility class for managing federated learning data distribution"""

    @staticmethod
    def create_iid_distribution(dataset, num_clients: int,
                                split_ratio: float = 0.8) -> Dict[int, List[int]]:
        """
        Create IID (Independent and Identically Distributed) data split

        Args:
            dataset: Full dataset
            num_clients: Number of clients
            split_ratio: Train/test split ratio

        Returns:
            Dictionary mapping client_id to list of data indices
        """
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients

        # Shuffle indices
        all_indices = torch.randperm(total_samples).tolist()

        client_data = {}
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client

            if client_id == num_clients - 1:  # Last client gets remaining samples
                end_idx = total_samples

            client_indices = all_indices[start_idx:end_idx]
            client_data[client_id] = client_indices

        return client_data

    @staticmethod
    def create_non_iid_distribution(dataset, num_clients: int,
                                    alpha: float = 0.5,
                                    num_classes: int = 100) -> Dict[int, List[int]]:
        """
        Create Non-IID data distribution using Dirichlet distribution

        Args:
            dataset: Full dataset
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            num_classes: Number of classes in dataset

        Returns:
            Dictionary mapping client_id to list of data indices
        """
        from collections import defaultdict

        # Group indices by class
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        # Sample class proportions for each client using Dirichlet distribution
        client_data = {i: [] for i in range(num_clients)}

        for class_id in range(num_classes):
            if class_id not in class_indices:
                continue

            class_samples = class_indices[class_id]
            np.random.shuffle(class_samples)

            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)

            # Assign samples to clients based on proportions
            start_idx = 0
            for client_id, proportion in enumerate(proportions):
                num_samples = int(len(class_samples) * proportion)
                end_idx = start_idx + num_samples

                if client_id == num_clients - 1:  # Last client gets remaining
                    end_idx = len(class_samples)

                client_data[client_id].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx

        # Shuffle each client's data
        for client_id in client_data:
            np.random.shuffle(client_data[client_id])

        return client_data

    @staticmethod
    def analyze_data_distribution(client_data: Dict[int, List[int]],
                                  dataset, num_classes: int = 100) -> Dict[str, Any]:
        """Analyze the data distribution across clients"""
        analysis = {
            'total_clients': len(client_data),
            'total_samples': sum(len(indices) for indices in client_data.values()),
            'client_sizes': {},
            'class_distributions': {},
            'diversity_metrics': {}
        }

        # Analyze each client
        for client_id, indices in client_data.items():
            analysis['client_sizes'][client_id] = len(indices)

            # Count classes for this client
            class_counts = {i: 0 for i in range(num_classes)}
            for idx in indices:
                _, label = dataset[idx]
                class_counts[label] += 1

            analysis['class_distributions'][client_id] = class_counts

        # Compute diversity metrics
        all_class_counts = np.array([
            list(analysis['class_distributions'][cid].values())
            for cid in range(len(client_data))
        ])

        # Jensen-Shannon divergence between clients
        js_divergences = []
        for i in range(len(client_data)):
            for j in range(i + 1, len(client_data)):
                dist1 = all_class_counts[i] / (np.sum(all_class_counts[i]) + 1e-10)
                dist2 = all_class_counts[j] / (np.sum(all_class_counts[j]) + 1e-10)
                js_div = FLDataManager._jensen_shannon_divergence(dist1, dist2)
                js_divergences.append(js_div)

        analysis['diversity_metrics'] = {
            'avg_js_divergence': np.mean(js_divergences) if js_divergences else 0.0,
            'std_js_divergence': np.std(js_divergences) if js_divergences else 0.0,
            'min_client_size': min(analysis['client_sizes'].values()),
            'max_client_size': max(analysis['client_sizes'].values()),
            'avg_client_size': np.mean(list(analysis['client_sizes'].values()))
        }

        return analysis

    @staticmethod
    def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two probability distributions"""
        # Avoid log(0) by adding small epsilon
        p = p + 1e-10
        q = q + 1e-10

        # Normalize to ensure they sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Compute Jensen-Shannon divergence
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


class FLVisualizer:
    """Utility class for visualizing federated learning results"""

    @staticmethod
    def plot_training_curves(round_metrics: List[Dict[str, Any]],
                             save_path: Optional[str] = None):
        """Plot training curves across FL rounds"""
        rounds = [m['round'] for m in round_metrics]

        # Extract different metrics
        convergence_metrics = [m.get('convergence_metrics', {}) for m in round_metrics]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Parameter change over rounds
        if convergence_metrics and 'total_parameter_change' in convergence_metrics[0]:
            param_changes = [m.get('total_parameter_change', 0) for m in convergence_metrics]
            axes[0, 0].plot(rounds, param_changes, 'b-o')
            axes[0, 0].set_title('Parameter Change Over Rounds')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Total Parameter Change')
            axes[0, 0].grid(True)

        # Client participation
        num_clients = [m.get('num_clients', 0) for m in round_metrics]
        axes[0, 1].bar(rounds, num_clients, alpha=0.7)
        axes[0, 1].set_title('Client Participation Per Round')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Number of Clients')
        axes[0, 1].grid(True, alpha=0.3)

        # Total samples per round
        total_samples = [m.get('total_samples', 0) for m in round_metrics]
        axes[1, 0].plot(rounds, total_samples, 'g-s')
        axes[1, 0].set_title('Total Samples Per Round')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Total Samples')
        axes[1, 0].grid(True)

        # Attack success (if available)
        attack_metrics = []
        for m in round_metrics:
            attack_results = m.get('attack_results', {})
            if attack_results:
                total_attacks = len(attack_results)
                successful = sum(1 for result in attack_results.values()
                                 if 'error' not in result and len(result.get('patches', [])) > 0)
                success_rate = successful / total_attacks if total_attacks > 0 else 0
                attack_metrics.append(success_rate)
            else:
                attack_metrics.append(0)

        axes[1, 1].plot(rounds, attack_metrics, 'r-^')
        axes[1, 1].set_title('Attack Success Rate Per Round')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_data_distribution(client_data: Dict[int, List[int]],
                               dataset, num_classes: int = 100,
                               save_path: Optional[str] = None):
        """Plot data distribution across clients"""

        # Compute class distribution matrix
        distribution_matrix = np.zeros((len(client_data), num_classes))

        for client_id, indices in client_data.items():
            class_counts = {i: 0 for i in range(num_classes)}
            for idx in indices:
                _, label = dataset[idx]
                class_counts[label] += 1

            # Normalize to get proportions
            total_samples = len(indices)
            for class_id, count in class_counts.items():
                distribution_matrix[client_id, class_id] = count / total_samples

        # Create heatmap
        plt.figure(figsize=(20, max(8, len(client_data) * 0.3)))

        sns.heatmap(distribution_matrix,
                    xticklabels=range(num_classes),
                    yticklabels=[f'Client {i}' for i in range(len(client_data))],
                    cmap='Blues',
                    annot=False,
                    cbar_kws={'label': 'Class Proportion'})

        plt.title('Data Distribution Across Clients')
        plt.xlabel('Class ID')
        plt.ylabel('Client ID')

        # Reduce x-axis tick density for readability
        if num_classes > 20:
            tick_step = max(1, num_classes // 20)
            plt.xticks(range(0, num_classes, tick_step))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_attack_effectiveness_heatmap(attack_results: Dict[str, Dict[str, Any]],
                                          peft_methods: List[str],
                                          defense_methods: List[str],
                                          save_path: Optional[str] = None):
        """Plot heatmap of attack effectiveness across PEFT methods and defenses"""

        # Create matrix of attack success rates
        effectiveness_matrix = np.zeros((len(peft_methods), len(defense_methods)))

        for i, peft_method in enumerate(peft_methods):
            for j, defense_method in enumerate(defense_methods):
                key = f"{peft_method}_{defense_method}"
                if key in attack_results:
                    # Extract success rate or PSNR
                    result = attack_results[key]
                    if 'success_rate' in result:
                        effectiveness_matrix[i, j] = result['success_rate']
                    elif 'avg_psnr' in result:
                        # Convert PSNR to success rate (higher PSNR = more successful attack)
                        psnr = result['avg_psnr']
                        effectiveness_matrix[i, j] = min(1.0, psnr / 30.0)  # Normalize to [0,1]

        # Create heatmap
        plt.figure(figsize=(12, 8))

        sns.heatmap(effectiveness_matrix,
                    xticklabels=defense_methods,
                    yticklabels=peft_methods,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlBu_r',  # Red = high effectiveness
                    center=0.5,
                    cbar_kws={'label': 'Attack Success Rate'})

        plt.title('Attack Effectiveness: PEFT Methods vs Defense Mechanisms')
        plt.xlabel('Defense Method')
        plt.ylabel('PEFT Method')
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FLMetrics:
    """Utility class for computing FL-specific metrics"""

    @staticmethod
    def compute_convergence_metrics(round_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute convergence metrics across rounds"""
        if not round_history:
            return {}

        # Extract convergence data
        param_changes = []
        relative_changes = []

        for round_result in round_history:
            conv_metrics = round_result.get('convergence_metrics', {})
            if 'total_parameter_change' in conv_metrics:
                param_changes.append(conv_metrics['total_parameter_change'])
            if 'relative_change' in conv_metrics:
                relative_changes.append(conv_metrics['relative_change'])

        metrics = {
            'total_rounds': len(round_history),
            'final_parameter_change': param_changes[-1] if param_changes else 0,
            'convergence_rate': 0,
            'stability_score': 0
        }

        # Compute convergence rate (decrease in parameter change)
        if len(param_changes) > 1:
            initial_change = param_changes[0]
            final_change = param_changes[-1]
            if initial_change > 0:
                metrics['convergence_rate'] = (initial_change - final_change) / initial_change

        # Compute stability (variance in recent changes)
        if len(param_changes) > 5:
            recent_changes = param_changes[-5:]
            metrics['stability_score'] = 1.0 / (1.0 + np.var(recent_changes))

        return metrics

    @staticmethod
    def compute_privacy_metrics(defense_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute privacy-related metrics"""
        privacy_metrics = {
            'total_privacy_cost': 0.0,
            'average_privacy_cost': 0.0,
            'privacy_efficiency': 0.0  # Privacy per unit utility loss
        }

        if 'privacy_costs' in defense_results:
            costs = defense_results['privacy_costs']
            privacy_metrics['total_privacy_cost'] = sum(costs)
            privacy_metrics['average_privacy_cost'] = np.mean(costs) if costs else 0.0

        return privacy_metrics

    @staticmethod
    def compute_communication_efficiency(round_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute communication efficiency metrics"""
        if not round_history:
            return {}

        total_samples = sum(r.get('total_samples', 0) for r in round_history)
        total_rounds = len(round_history)
        avg_clients_per_round = np.mean([r.get('num_clients', 0) for r in round_history])

        return {
            'total_samples_processed': total_samples,
            'average_samples_per_round': total_samples / total_rounds if total_rounds > 0 else 0,
            'average_clients_per_round': avg_clients_per_round,
            'communication_rounds': total_rounds
        }


class FLExperimentManager:
    """Manager for running and tracking FL experiments"""

    def __init__(self, experiment_name: str, save_dir: str = "./fl_experiments"):
        """
        Initialize experiment manager

        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save experiment results
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.experiment_dir = os.path.join(save_dir, experiment_name)

        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Setup logging
        log_file = os.path.join(self.experiment_dir, 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)

        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'status': 'initialized'
        }

        self.results = {}

    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        config_file = os.path.join(self.experiment_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        self.metadata['config'] = config
        self.logger.info("Experiment configuration saved")

    def save_results(self, results: Dict[str, Any]):
        """Save experiment results"""
        self.results.update(results)

        # Save as JSON
        results_file = os.path.join(self.experiment_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save as pickle for complex objects
        pickle_file = os.path.join(self.experiment_dir, 'results.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)

        self.logger.info("Experiment results saved")

    def save_plots(self, round_history: List[Dict[str, Any]]):
        """Save visualization plots"""
        plots_dir = os.path.join(self.experiment_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Training curves
        training_curves_path = os.path.join(plots_dir, 'training_curves.png')
        FLVisualizer.plot_training_curves(round_history, training_curves_path)

        self.logger.info("Plots saved")

    def finalize_experiment(self):
        """Finalize experiment and save metadata"""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['status'] = 'completed'

        metadata_file = os.path.join(self.experiment_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        self.logger.info(f"Experiment {self.experiment_name} completed")


# Example usage and testing
if __name__ == "__main__":
    print("Testing FL utilities...")

    # Test data distribution creation
    print("\n1. Testing data distribution:")


    # Mock dataset for testing
    class MockDataset:
        def __init__(self, size=1000, num_classes=10):
            self.size = size
            self.num_classes = num_classes

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Random data and label
            return torch.randn(3, 32, 32), idx % self.num_classes


    mock_dataset = MockDataset(1000, 10)

    # Test IID distribution
    iid_data = FLDataManager.create_iid_distribution(mock_dataset, num_clients=5)
    print(f"  ✓ IID distribution created: {len(iid_data)} clients")

    # Test Non-IID distribution
    non_iid_data = FLDataManager.create_non_iid_distribution(mock_dataset, num_clients=5, alpha=0.1)
    print(f"  ✓ Non-IID distribution created: {len(non_iid_data)} clients")

    # Test distribution analysis
    analysis = FLDataManager.analyze_data_distribution(non_iid_data, mock_dataset, num_classes=10)
    print(f"  ✓ Distribution analysis completed")
    print(f"    Average JS divergence: {analysis['diversity_metrics']['avg_js_divergence']:.4f}")

    # Test metrics computation
    print("\n2. Testing metrics computation:")

    # Mock round history
    mock_round_history = [
        {
            'round': i,
            'num_clients': 5,
            'total_samples': 500,
            'convergence_metrics': {
                'total_parameter_change': 10.0 * np.exp(-i * 0.1),
                'relative_change': 0.1 * np.exp(-i * 0.1)
            }
        }
        for i in range(1, 11)
    ]

    conv_metrics = FLMetrics.compute_convergence_metrics(mock_round_history)
    print(f"  ✓ Convergence metrics: convergence_rate={conv_metrics.get('convergence_rate', 0):.4f}")

    comm_metrics = FLMetrics.compute_communication_efficiency(mock_round_history)
    print(f"  ✓ Communication metrics: total_samples={comm_metrics.get('total_samples_processed', 0)}")

    # Test experiment manager
    print("\n3. Testing experiment manager:")

    exp_manager = FLExperimentManager("test_experiment", "./test_experiments")

    test_config = {
        'num_clients': 5,
        'num_rounds': 10,
        'peft_method': 'adapter'
    }

    exp_manager.save_config(test_config)

    test_results = {
        'final_accuracy': 0.85,
        'convergence_round': 8
    }

    exp_manager.save_results(test_results)
    exp_manager.finalize_experiment()

    print(f"  ✓ Experiment saved to: {exp_manager.experiment_dir}")

    print("\nFL utilities testing completed!")
