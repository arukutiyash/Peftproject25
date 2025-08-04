import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from defenses.defense_base import BaseDefense
from defenses.mixup_defense import MixUpDefense, AdaptiveMixUpDefense #defense_factory.py
from defenses.instahide_defense import InstaHideDefense, AdaptiveInstaHideDefense
from defenses.differential_privacy import DifferentialPrivacyDefense, AdaptiveDPDefense, RenyiDPDefense, LocalDPDefense
from defenses.grad_prune_defense import GradientPruningDefense, AdaptiveGradientPruning, LayerWiseGradientPruning
from defenses.combined_defenses import (CombinedDefense, MixUpDPCombination, InstaHidePruningCombination,
                                ComprehensiveDefense, AdaptiveComprehensiveDefense,
                                create_lightweight_defense, create_strong_privacy_defense, create_balanced_defense)


class DefenseFactory:
    """Factory class for creating defense mechanisms"""

    SUPPORTED_DEFENSES = {
        'mixup', 'adaptive_mixup', 'instahide', 'adaptive_instahide',
        'differential_privacy', 'adaptive_dp', 'renyi_dp', 'local_dp',
        'gradient_pruning', 'adaptive_pruning', 'layerwise_pruning',
        'mixup_dp', 'instahide_pruning', 'comprehensive', 'adaptive_comprehensive',
        'lightweight', 'strong_privacy', 'balanced'
    }

    @staticmethod
    def create_defense(defense_type: str, device: str = 'cuda',
                       config: Optional[Dict[str, Any]] = None) -> BaseDefense:
        """
        Create a defense mechanism

        Args:
            defense_type: Type of defense to create
            device: Computing device
            config: Configuration parameters for the defense

        Returns:
            Defense instance
        """

        if defense_type not in DefenseFactory.SUPPORTED_DEFENSES:
            raise ValueError(f"Unsupported defense type: {defense_type}. "
                             f"Supported: {DefenseFactory.SUPPORTED_DEFENSES}")

        config = config or {}

        # Single defenses
        if defense_type == 'mixup':
            return DefenseFactory._create_mixup_defense(device, config)
        elif defense_type == 'adaptive_mixup':
            return DefenseFactory._create_adaptive_mixup_defense(device, config)
        elif defense_type == 'instahide':
            return DefenseFactory._create_instahide_defense(device, config)
        elif defense_type == 'adaptive_instahide':
            return DefenseFactory._create_adaptive_instahide_defense(device, config)
        elif defense_type == 'differential_privacy':
            return DefenseFactory._create_dp_defense(device, config)
        elif defense_type == 'adaptive_dp':
            return DefenseFactory._create_adaptive_dp_defense(device, config)
        elif defense_type == 'renyi_dp':
            return DefenseFactory._create_renyi_dp_defense(device, config)
        elif defense_type == 'local_dp':
            return DefenseFactory._create_local_dp_defense(device, config)
        elif defense_type == 'gradient_pruning':
            return DefenseFactory._create_pruning_defense(device, config)
        elif defense_type == 'adaptive_pruning':
            return DefenseFactory._create_adaptive_pruning_defense(device, config)
        elif defense_type == 'layerwise_pruning':
            return DefenseFactory._create_layerwise_pruning_defense(device, config)

        # Combined defenses
        elif defense_type == 'mixup_dp':
            return DefenseFactory._create_mixup_dp_combination(device, config)
        elif defense_type == 'instahide_pruning':
            return DefenseFactory._create_instahide_pruning_combination(device, config)
        elif defense_type == 'comprehensive':
            return DefenseFactory._create_comprehensive_defense(device, config)
        elif defense_type == 'adaptive_comprehensive':
            return DefenseFactory._create_adaptive_comprehensive_defense(device, config)

        # Preset combinations
        elif defense_type == 'lightweight':
            return create_lightweight_defense(device)
        elif defense_type == 'strong_privacy':
            return create_strong_privacy_defense(device)
        elif defense_type == 'balanced':
            return create_balanced_defense(device)

        else:
            raise ValueError(f"Defense type {defense_type} not implemented")

    @staticmethod
    def _create_mixup_defense(device: str, config: Dict[str, Any]) -> MixUpDefense:
        """Create MixUp defense"""
        default_config = {
            'alpha': 1.0,
            'mixup_prob': 0.5
        }
        default_config.update(config)

        return MixUpDefense(
            device=device,
            alpha=default_config['alpha'],
            mixup_prob=default_config['mixup_prob']
        )

    @staticmethod
    def _create_adaptive_mixup_defense(device: str, config: Dict[str, Any]) -> AdaptiveMixUpDefense:
        """Create Adaptive MixUp defense"""
        default_config = {
            'initial_alpha': 1.0,
            'adaptation_rate': 0.1,
            'mixup_prob': 0.5
        }
        default_config.update(config)

        return AdaptiveMixUpDefense(
            device=device,
            initial_alpha=default_config['initial_alpha'],
            adaptation_rate=default_config['adaptation_rate'],
            mixup_prob=default_config['mixup_prob']
        )

    @staticmethod
    def _create_instahide_defense(device: str, config: Dict[str, Any]) -> InstaHideDefense:
        """Create InstaHide defense"""
        default_config = {
            'k_mix': 4,
            'mask_ratio': 0.5,
            'sign_flip_prob': 0.5
        }
        default_config.update(config)

        return InstaHideDefense(
            device=device,
            k_mix=default_config['k_mix'],
            mask_ratio=default_config['mask_ratio'],
            sign_flip_prob=default_config['sign_flip_prob']
        )

    @staticmethod
    def _create_adaptive_instahide_defense(device: str, config: Dict[str, Any]) -> AdaptiveInstaHideDefense:
        """Create Adaptive InstaHide defense"""
        default_config = {
            'k_mix': 4,
            'mask_ratio': 0.5,
            'sign_flip_prob': 0.5
        }
        default_config.update(config)

        return AdaptiveInstaHideDefense(
            device=device,
            k_mix=default_config['k_mix'],
            mask_ratio=default_config['mask_ratio'],
            sign_flip_prob=default_config['sign_flip_prob']
        )

    @staticmethod
    def _create_dp_defense(device: str, config: Dict[str, Any]) -> DifferentialPrivacyDefense:
        """Create Differential Privacy defense"""
        default_config = {
            'epsilon': 1.0,
            'delta': 1e-5,
            'sensitivity': 1.0,
            'clipping_threshold': 1.0
        }
        default_config.update(config)

        return DifferentialPrivacyDefense(
            device=device,
            epsilon=default_config['epsilon'],
            delta=default_config['delta'],
            sensitivity=default_config['sensitivity'],
            clipping_threshold=default_config['clipping_threshold']
        )

    @staticmethod
    def _create_adaptive_dp_defense(device: str, config: Dict[str, Any]) -> AdaptiveDPDefense:
        """Create Adaptive DP defense"""
        default_config = {
            'initial_epsilon': 1.0,
            'total_budget': 10.0,
            'adaptation_strategy': 'linear',
            'delta': 1e-5
        }
        default_config.update(config)

        return AdaptiveDPDefense(
            device=device,
            initial_epsilon=default_config['initial_epsilon'],
            total_budget=default_config['total_budget'],
            adaptation_strategy=default_config['adaptation_strategy'],
            delta=default_config['delta']
        )

    @staticmethod
    def _create_renyi_dp_defense(device: str, config: Dict[str, Any]) -> RenyiDPDefense:
        """Create Rényi DP defense"""
        default_config = {
            'alpha': 2.0,
            'epsilon': 1.0,
            'delta': 1e-5
        }
        default_config.update(config)

        return RenyiDPDefense(
            device=device,
            alpha=default_config['alpha'],
            epsilon=default_config['epsilon'],
            delta=default_config['delta']
        )

    @staticmethod
    def _create_local_dp_defense(device: str, config: Dict[str, Any]) -> LocalDPDefense:
        """Create Local DP defense"""
        default_config = {
            'epsilon_local': 1.0,
            'mechanism': 'laplace'
        }
        default_config.update(config)

        return LocalDPDefense(
            device=device,
            epsilon_local=default_config['epsilon_local'],
            mechanism=default_config['mechanism']
        )

    @staticmethod
    def _create_pruning_defense(device: str, config: Dict[str, Any]) -> GradientPruningDefense:
        """Create Gradient Pruning defense"""
        default_config = {
            'pruning_ratio': 0.5,
            'pruning_strategy': 'magnitude',
            'adaptive': False
        }
        default_config.update(config)

        return GradientPruningDefense(
            device=device,
            pruning_ratio=default_config['pruning_ratio'],
            pruning_strategy=default_config['pruning_strategy'],
            adaptive=default_config['adaptive']
        )

    @staticmethod
    def _create_adaptive_pruning_defense(device: str, config: Dict[str, Any]) -> AdaptiveGradientPruning:
        """Create Adaptive Gradient Pruning defense"""
        default_config = {
            'pruning_ratio': 0.5,
            'pruning_strategy': 'magnitude'
        }
        default_config.update(config)

        return AdaptiveGradientPruning(
            device=device,
            pruning_ratio=default_config['pruning_ratio'],
            pruning_strategy=default_config['pruning_strategy']
        )

    @staticmethod
    def _create_layerwise_pruning_defense(device: str, config: Dict[str, Any]) -> LayerWiseGradientPruning:
        """Create Layer-wise Gradient Pruning defense"""
        default_config = {
            'pruning_ratio': 0.5,
            'layer_pruning_ratios': {}
        }
        default_config.update(config)

        return LayerWiseGradientPruning(
            device=device,
            pruning_ratio=default_config['pruning_ratio'],
            layer_pruning_ratios=default_config['layer_pruning_ratios']
        )

    @staticmethod
    def _create_mixup_dp_combination(device: str, config: Dict[str, Any]) -> MixUpDPCombination:
        """Create MixUp + DP combination"""
        default_config = {
            'mixup_alpha': 1.0,
            'dp_epsilon': 1.0,
            'combination_weight': 0.5
        }
        default_config.update(config)

        return MixUpDPCombination(
            device=device,
            mixup_alpha=default_config['mixup_alpha'],
            dp_epsilon=default_config['dp_epsilon'],
            combination_weight=default_config['combination_weight']
        )

    @staticmethod
    def _create_instahide_pruning_combination(device: str, config: Dict[str, Any]) -> InstaHidePruningCombination:
        """Create InstaHide + Pruning combination"""
        default_config = {
            'k_mix': 4,
            'pruning_ratio': 0.5
        }
        default_config.update(config)

        return InstaHidePruningCombination(
            device=device,
            k_mix=default_config['k_mix'],
            pruning_ratio=default_config['pruning_ratio']
        )

    @staticmethod
    def _create_comprehensive_defense(device: str, config: Dict[str, Any]) -> ComprehensiveDefense:
        """Create Comprehensive defense"""
        default_config = {
            'defense_weights': {
                'mixup': 0.25,
                'instahide': 0.25,
                'differential_privacy': 0.25,
                'gradient_pruning': 0.25
            }
        }
        default_config.update(config)

        return ComprehensiveDefense(
            device=device,
            defense_weights=default_config['defense_weights']
        )

    @staticmethod
    def _create_adaptive_comprehensive_defense(device: str, config: Dict[str, Any]) -> AdaptiveComprehensiveDefense:
        """Create Adaptive Comprehensive defense"""
        return AdaptiveComprehensiveDefense(device=device, **config)

    @staticmethod
    def create_defense_suite(defense_configs: Dict[str, Dict[str, Any]],
                             device: str = 'cuda') -> Dict[str, BaseDefense]:
        """
        Create multiple defenses from configuration dictionary

        Args:
            defense_configs: Dictionary mapping defense names to their configs
            device: Computing device

        Returns:
            Dictionary of defense instances
        """
        defense_suite = {}

        for defense_name, config in defense_configs.items():
            defense_type = config.pop('type', defense_name)  # Use name as type if not specified

            try:
                defense_instance = DefenseFactory.create_defense(defense_type, device, config)
                defense_suite[defense_name] = defense_instance
            except Exception as e:
                print(f"Warning: Failed to create defense {defense_name}: {e}")

        return defense_suite

    @staticmethod
    def get_defense_recommendations(threat_model: str = 'moderate') -> Dict[str, Dict[str, Any]]:
        """
        Get recommended defense configurations based on threat model

        Args:
            threat_model: Threat level ('low', 'moderate', 'high', 'critical')

        Returns:
            Dictionary of recommended defense configurations
        """

        if threat_model == 'low':
            return {
                'lightweight_pruning': {
                    'type': 'gradient_pruning',
                    'pruning_ratio': 0.3,
                    'pruning_strategy': 'magnitude'
                }
            }

        elif threat_model == 'moderate':
            return {
                'balanced_defense': {
                    'type': 'balanced'
                },
                'mixup_defense': {
                    'type': 'mixup',
                    'alpha': 1.0,
                    'mixup_prob': 0.5
                }
            }

        elif threat_model == 'high':
            return {
                'strong_privacy': {
                    'type': 'strong_privacy'
                },
                'adaptive_dp': {
                    'type': 'adaptive_dp',
                    'initial_epsilon': 0.5,
                    'total_budget': 5.0
                }
            }

        elif threat_model == 'critical':
            return {
                'adaptive_comprehensive': {
                    'type': 'adaptive_comprehensive'
                },
                'renyi_dp': {
                    'type': 'renyi_dp',
                    'alpha': 10.0,
                    'epsilon': 0.1
                }
            }

        else:
            raise ValueError(f"Unknown threat model: {threat_model}")


# Example usage and testing
if __name__ == "__main__":

    # Test creating different defense types
    defense_types = ['mixup', 'differential_privacy', 'gradient_pruning', 'comprehensive']

    for defense_type in defense_types:
        print(f"\nTesting {defense_type.upper()} defense:")

        try:
            defense = DefenseFactory.create_defense(defense_type, device='cpu')
            print(f"  ✓ {defense.__class__.__name__} created successfully")
            print(f"  Config: {defense.defense_config}")

            # Test with dummy gradients
            dummy_gradients = {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10),
                'layer2.weight': torch.randn(1, 10)
            }

            defended_gradients = defense.apply_defense_with_validation(dummy_gradients)
            print(f"  ✓ Defense applied successfully")
            print(f"  Privacy cost: {defense.get_privacy_cost():.4f}")

        except Exception as e:
            print(f"  ✗ Error with {defense_type}: {e}")

    # Test defense suite creation
    print(f"\nTesting defense suite creation:")

    suite_config = {
        'light_defense': {'type': 'lightweight'},
        'strong_defense': {'type': 'strong_privacy'},
        'custom_mixup': {
            'type': 'mixup',
            'alpha': 2.0,
            'mixup_prob': 0.8
        }
    }

    try:
        defense_suite = DefenseFactory.create_defense_suite(suite_config, device='cpu')
        print(f"  ✓ Created {len(defense_suite)} defenses in suite")
        for name, defense in defense_suite.items():
            print(f"    - {name}: {defense.__class__.__name__}")
    except Exception as e:
        print(f"  ✗ Error creating defense suite: {e}")

    print("\nAll defense factory tests completed!")
