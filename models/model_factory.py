import torch
from typing import Optional, Dict, Any, Union
from models.vit_model import VisionTransformer, create_vit_base_patch4_32, create_vit_small_patch4_32
from models.peft_adapters import AdapterConfig, add_adapters_to_vit
from models.peft_prefix import PrefixConfig, add_prefix_to_vit
from models.peft_bias import BiasConfig, add_bias_tuning_to_vit
from models.peft_lora import LoRAConfig, add_lora_to_vit

class ModelFactory:
    """Factory class for creating Vision Transformer models with different PEFT methods"""

    SUPPORTED_PEFT_METHODS = ['adapter', 'prefix', 'bias', 'lora']
    SUPPORTED_MODEL_SIZES = ['base', 'small']

    @staticmethod
    def create_model(model_size: str = 'base',
                     peft_method: str = 'adapter',
                     num_classes: int = 100,
                     peft_config: Optional[Dict[str, Any]] = None,
                     malicious: bool = False) -> VisionTransformer:
        """
        Create a Vision Transformer model with specified PEFT method

        Args:
            model_size: Size of the model ('base' or 'small')
            peft_method: PEFT method to use ('adapter', 'prefix', 'bias', 'lora')
            num_classes: Number of output classes
            peft_config: Configuration for the PEFT method
            malicious: Whether to create malicious version for attacks

        Returns:
            VisionTransformer model with PEFT method applied
        """

        # Validate inputs
        if model_size not in ModelFactory.SUPPORTED_MODEL_SIZES:
            raise ValueError(f"Unsupported model size: {model_size}. Supported: {ModelFactory.SUPPORTED_MODEL_SIZES}")

        if peft_method not in ModelFactory.SUPPORTED_PEFT_METHODS:
            raise ValueError(
                f"Unsupported PEFT method: {peft_method}. Supported: {ModelFactory.SUPPORTED_PEFT_METHODS}")

        # Create base model
        if model_size == 'base':
            model = create_vit_base_patch4_32(num_classes=num_classes)
        elif model_size == 'small':
            model = create_vit_small_patch4_32(num_classes=num_classes)

        # Apply PEFT method
        model = ModelFactory._apply_peft_method(model, peft_method, peft_config, malicious)

        return model

    @staticmethod
    def _apply_peft_method(model: VisionTransformer,
                           peft_method: str,
                           peft_config: Optional[Dict[str, Any]],
                           malicious: bool) -> VisionTransformer:
        """Apply the specified PEFT method to the model"""

        if peft_method == 'adapter':
            config = ModelFactory._create_adapter_config(peft_config, malicious)
            model = add_adapters_to_vit(model, config)

        elif peft_method == 'prefix':
            config = ModelFactory._create_prefix_config(peft_config, malicious)
            model = add_prefix_to_vit(model, config)

        elif peft_method == 'bias':
            config = ModelFactory._create_bias_config(peft_config, malicious)
            model = add_bias_tuning_to_vit(model, config)

        elif peft_method == 'lora':
            config = ModelFactory._create_lora_config(peft_config, malicious)
            model = add_lora_to_vit(model, config)

        return model

    @staticmethod
    def _create_adapter_config(peft_config: Optional[Dict[str, Any]], malicious: bool) -> AdapterConfig:
        """Create adapter configuration"""
        default_config = {
            'bottleneck_dim': 64,
            'dropout': 0.0,
            'init_option': 'bert',
            'adapter_scalar': 1.0,
            'target_modules': ['attn', 'mlp']
        }

        if peft_config:
            default_config.update(peft_config)

        return AdapterConfig(
            bottleneck_dim=default_config['bottleneck_dim'],
            dropout=default_config['dropout'],
            init_option=default_config['init_option'],
            adapter_scalar=default_config['adapter_scalar'],
            target_modules=default_config['target_modules'],
            malicious=malicious
        )

    @staticmethod
    def _create_prefix_config(peft_config: Optional[Dict[str, Any]], malicious: bool) -> PrefixConfig:
        """Create prefix-tuning configuration"""
        default_config = {
            'prefix_length': 10,
            'hidden_dim': None,
            'target_modules': ['attn']
        }

        if peft_config:
            default_config.update(peft_config)

        return PrefixConfig(
            prefix_length=default_config['prefix_length'],
            hidden_dim=default_config['hidden_dim'],
            target_modules=default_config['target_modules'],
            malicious=malicious
        )

    @staticmethod
    def _create_bias_config(peft_config: Optional[Dict[str, Any]], malicious: bool) -> BiasConfig:
        """Create bias-tuning configuration"""
        default_config = {
            'target_modules': ['linear', 'layernorm'],
            'init_value': 0.0
        }

        if peft_config:
            default_config.update(peft_config)

        return BiasConfig(
            target_modules=default_config['target_modules'],
            init_value=default_config['init_value'],
            malicious=malicious
        )

    @staticmethod
    def _create_lora_config(peft_config: Optional[Dict[str, Any]], malicious: bool) -> LoRAConfig:
        """Create LoRA configuration"""
        default_config = {
            'rank': 16,
            'alpha': 16,
            'dropout': 0.0,
            'target_modules': ['qkv', 'proj', 'fc1', 'fc2']
        }

        if peft_config:
            default_config.update(peft_config)

        return LoRAConfig(
            rank=default_config['rank'],
            alpha=default_config['alpha'],
            dropout=default_config['dropout'],
            target_modules=default_config['target_modules'],
            malicious=malicious
        )

    @staticmethod
    def create_federated_models(model_size: str = 'base',
                                peft_method: str = 'adapter',
                                num_classes: int = 100,
                                num_clients: int = 10,
                                peft_config: Optional[Dict[str, Any]] = None,
                                malicious_server: bool = False) -> Dict[str, VisionTransformer]:
        """
        Create models for federated learning setup

        Args:
            model_size: Size of the model
            peft_method: PEFT method to use
            num_classes: Number of output classes
            num_clients: Number of client models to create
            peft_config: Configuration for the PEFT method
            malicious_server: Whether the server model should be malicious

        Returns:
            Dictionary containing 'server' model and 'clients' list of models
        """

        # Create server model (potentially malicious)
        server_model = ModelFactory.create_model(
            model_size=model_size,
            peft_method=peft_method,
            num_classes=num_classes,
            peft_config=peft_config,
            malicious=malicious_server
        )

        # Create client models (not malicious)
        client_models = []
        for i in range(num_clients):
            client_model = ModelFactory.create_model(
                model_size=model_size,
                peft_method=peft_method,
                num_classes=num_classes,
                peft_config=peft_config,
                malicious=False
            )
            client_models.append(client_model)

        return {
            'server': server_model,
            'clients': client_models
        }

    @staticmethod
    def get_peft_parameters(model: VisionTransformer, peft_method: str):
        """Get PEFT-specific parameters from model"""
        if peft_method == 'adapter':
            from .peft_adapters import get_adapter_parameters
            return get_adapter_parameters(model)
        elif peft_method == 'prefix':
            from .peft_prefix import get_prefix_parameters
            return get_prefix_parameters(model)
        elif peft_method == 'bias':
            from .peft_bias import get_bias_parameters
            return get_bias_parameters(model)
        elif peft_method == 'lora':
            from .peft_lora import get_lora_parameters
            return get_lora_parameters(model)
        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")

    @staticmethod
    def get_peft_gradients(model: VisionTransformer, peft_method: str):
        """Get PEFT-specific gradients from model"""
        if peft_method == 'adapter':
            from .peft_adapters import get_adapter_gradients
            return get_adapter_gradients(model)
        elif peft_method == 'prefix':
            from .peft_prefix import get_prefix_gradients
            return get_prefix_gradients(model)
        elif peft_method == 'bias':
            from .peft_bias import get_bias_gradients
            return get_bias_gradients(model)
        elif peft_method == 'lora':
            from .peft_lora import get_lora_gradients
            return get_lora_gradients(model)
        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")


# Example usage and testing
if __name__ == "__main__":

    # Test different PEFT methods
    peft_methods = ['adapter', 'prefix', 'bias', 'lora']

    for method in peft_methods:
        print(f"\nTesting {method.upper()} method:")

        # Create regular model
        model = ModelFactory.create_model(
            model_size='base',
            peft_method=method,
            num_classes=100,
            malicious=False
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        try:
            output, _ = model(dummy_input)
            print(f"  Forward pass successful. Output shape: {output.shape}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")

    print("\nAll tests completed!")
