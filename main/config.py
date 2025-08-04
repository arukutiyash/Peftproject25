import torch
from typing import Dict, Any, List #config.py


class ExperimentConfig:
    """Configuration class for all experiments"""

    def __init__(self):
        # Device configuration
        self.device = 'cpu'  # Set to 'cuda' if GPU available

        # Dataset configuration
        self.dataset_config = {
            'name': 'CIFAR100',
            'subset_size': 1000,  # Use subset for faster testing, set to None for full dataset
            'batch_size': 16,  # Smaller batch size for CPU
            'num_workers': 2
        }

        # Model configuration
        self.model_config = {
            'model_type': 'vit_tiny',
            'num_classes': 100,
            'img_size': 32,
            'patch_size': 4,
            'embed_dim': 192,
            'num_heads': 3,
            'num_layers': 12
        }

        # PEFT method configurations
        self.peft_configs = {
            'adapter': {
                'bottleneck_dim': 32,
                'dropout': 0.1,
                'init_weights': True
            },
            'prefix': {
                'prefix_length': 8,
                'embed_dim': 192,
                'num_layers': 12
            },
            'bias': {
                'bias_types': ['attention', 'mlp', 'layernorm']
            },
            'lora': {
                'rank': 8,
                'alpha': 16,
                'dropout': 0.1,
                'target_modules': ['qkv', 'proj', 'fc1', 'fc2']
            }
        }

        # Attack configurations
        self.attack_configs = {
            'adapter': {
                'bottleneck_dim': 32,
                'recovery_method': 'gradient_difference',
                'max_iterations': 500
            },
            'prefix': {
                'prefix_length': 8,
                'recovery_method': 'attention_analysis',
                'attention_threshold': 0.1
            },
            'bias': {
                'recovery_method': 'direct_bias_inversion',
                'activation_threshold': 0.05
            },
            'lora': {
                'lora_rank': 8,
                'lora_alpha': 16,
                'recovery_method': 'low_rank_decomposition',
                'svd_threshold': 0.01
            }
        }

        # Defense configurations
        self.defense_configs = {
            'no_defense': {
                'type': 'none',
                'params': {}
            },
            'mixup': {
                'type': 'mixup',
                'params': {
                    'alpha': 1.0,
                    'mixup_prob': 0.5
                }
            },
            'instahide': {
                'type': 'instahide',
                'params': {
                    'k_mix': 4,
                    'mask_ratio': 0.5,
                    'sign_flip_prob': 0.5
                }
            },
            'differential_privacy': {
                'type': 'differential_privacy',
                'params': {
                    'epsilon': 1.0,
                    'delta': 1e-5,
                    'clipping_threshold': 1.0
                }
            },
            'gradient_pruning': {
                'type': 'gradient_pruning',
                'params': {
                    'pruning_ratio': 0.5,
                    'pruning_strategy': 'magnitude'
                }
            },
            'adaptive_dp': {
                'type': 'adaptive_dp',
                'params': {
                    'initial_epsilon': 1.0,
                    'total_budget': 10.0,
                    'adaptation_strategy': 'linear'
                }
            },
            'comprehensive': {
                'type': 'comprehensive',
                'params': {
                    'defense_weights': {
                        'mixup': 0.25,
                        'instahide': 0.25,
                        'differential_privacy': 0.25,
                        'gradient_pruning': 0.25
                    }
                }
            }
        }

        # Federated learning configuration
        self.fl_config = {
            'num_clients': 5,
            'num_rounds': 10,
            'local_epochs': 1,
            'client_fraction': 1.0,
            'aggregation_method': 'fedavg'
        }

        # Evaluation configuration
        self.evaluation_config = {
            'num_samples': 100,  # Number of samples to evaluate per experiment
            'num_runs': 3,  # Number of runs for statistical significance
            'metrics': ['psnr', 'ssim', 'success_rate'],
            'statistical_tests': ['paired_t_test', 'one_way_anova'],
            'significance_level': 0.05
        }

        # Output configuration
        self.output_config = {
            'save_dir': './experiment_results',
            'save_models': False,  # Set to True to save trained models
            'save_patches': True,  # Save sample recovered patches
            'save_plots': True,  # Save visualization plots
            'generate_report': True  # Generate comprehensive report
        }

        # Quick test configuration (for development)
        self.quick_test_config = {
            'dataset_subset_size': 100,
            'num_samples': 20,
            'num_runs': 1,
            'fl_rounds': 3,
            'fl_clients': 3,
            'peft_methods': ['adapter', 'bias'],  # Test only 2 methods
            'defense_methods': ['no_defense', 'differential_privacy']
        }

    def get_peft_methods(self) -> List[str]:
        """Get list of PEFT methods to evaluate"""
        return list(self.peft_configs.keys())

    def get_defense_methods(self) -> List[str]:
        """Get list of defense methods to evaluate"""
        return list(self.defense_configs.keys())

    def get_quick_config(self) -> 'ExperimentConfig':
        """Get configuration for quick testing"""
        quick_config = ExperimentConfig()

        # Override with quick test settings
        quick_config.dataset_config['subset_size'] = self.quick_test_config['dataset_subset_size']
        quick_config.evaluation_config['num_samples'] = self.quick_test_config['num_samples']
        quick_config.evaluation_config['num_runs'] = self.quick_test_config['num_runs']
        quick_config.fl_config['num_rounds'] = self.quick_test_config['fl_rounds']
        quick_config.fl_config['num_clients'] = self.quick_test_config['fl_clients']

        # Filter PEFT and defense methods
        peft_methods = self.quick_test_config['peft_methods']
        defense_methods = self.quick_test_config['defense_methods']

        quick_config.peft_configs = {k: v for k, v in self.peft_configs.items() if k in peft_methods}
        quick_config.defense_configs = {k: v for k, v in self.defense_configs.items() if k in defense_methods}

        return quick_config

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check device availability
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA not available, switching to CPU")
                self.device = 'cpu'

            # Validate PEFT configs
            for method, config in self.peft_configs.items():
                if method not in ['adapter', 'prefix', 'bias', 'lora']:
                    raise ValueError(f"Unknown PEFT method: {method}")

            # Validate dataset config
            if self.dataset_config['batch_size'] <= 0:
                raise ValueError("Batch size must be positive")

            # Validate evaluation config
            if self.evaluation_config['num_samples'] <= 0:
                raise ValueError("Number of samples must be positive")

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        import json

        config_dict = {
            'device': self.device,
            'dataset_config': self.dataset_config,
            'model_config': self.model_config,
            'peft_configs': self.peft_configs,
            'attack_configs': self.attack_configs,
            'defense_configs': self.defense_configs,
            'fl_config': self.fl_config,
            'evaluation_config': self.evaluation_config,
            'output_config': self.output_config
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {filepath}")


# Global config instance
config = ExperimentConfig()

# Validate configuration on import
if not config.validate_config():
    print("Warning: Configuration validation failed!")
