"""Example script demonstrating all VLM implementations."""

import torch
from librobot.models.vlm import create_vlm, list_vlms


def demo_vlm(model_name: str, img_size: int = 224):
    """Demonstrate a VLM model."""
    print(f"\n{'='*60}")
    print(f"Demonstrating: {model_name}")
    print(f"{'='*60}")

    # Create model (without pretrained weights for demo)
    config = {
        'use_flash_attn': False,  # Disable for compatibility
    }

    print(f"Creating {model_name}...")
    model = create_vlm(model_name, config=config)

    # Print model info
    print(f"✓ Model created successfully")
    print(f"  - Embedding dimension: {model.get_embedding_dim()}")
    print(f"  - Total parameters: {model.get_num_parameters():,}")
    print(f"  - Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")
    print(f"  - Config: {model.config}")

    # Set to eval mode
    model.eval()

    # Test image encoding
    print(f"\nTesting image encoding...")
    batch_size = 2
    images = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        image_features = model.encode_image(images)

    print(f"✓ Image encoding successful")
    print(f"  - Input shape: {images.shape}")
    print(f"  - Output shape: {image_features.shape}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    labels = torch.randint(0, 1000, (batch_size, 20))

    with torch.no_grad():
        outputs = model(images=images, input_ids=input_ids, labels=labels)

    print(f"✓ Forward pass successful")
    print(f"  - Embeddings shape: {outputs['embeddings'].shape}")
    print(f"  - Logits shape: {outputs['logits'].shape}")
    print(f"  - Loss: {outputs['loss'].item():.4f}")

    # Test generation
    print(f"\nTesting text generation...")
    prompt = torch.randint(0, 1000, (1, 5))

    with torch.no_grad():
        generated = model.generate(
            images=images[:1],
            input_ids=prompt,
            max_new_tokens=10,
            temperature=1.0,
        )

    print(f"✓ Generation successful")
    print(f"  - Input length: {prompt.shape[1]}")
    print(f"  - Generated length: {generated.shape[1]}")
    print(f"  - Generated tokens: {generated[0].tolist()}")

    print(f"\n✅ All tests passed for {model_name}!")


def main():
    """Run demonstrations for all VLMs."""
    print("="*60)
    print("VLM (Vision-Language Model) Implementation Demo")
    print("="*60)

    # List all available VLMs
    vlms = list_vlms()
    print(f"\nAvailable VLMs: {len(vlms)}")
    for vlm in sorted(vlms):
        print(f"  - {vlm}")

    # Demo each VLM (using smaller variants for speed)
    models_to_demo = [
        ('qwen2-vl-2b', 224),
        ('florence-2-base', 224),
        ('paligemma-3b', 224),
        ('internvl2-2b', 224),
        ('llava-v1.5-7b', 336),
    ]

    for model_name, img_size in models_to_demo:
        try:
            demo_vlm(model_name, img_size)
        except Exception as e:
            print(f"\n❌ Error demonstrating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
