"""Main test runner for CogVideoX inpainting tests."""

import pytest
from .features.test_conditioning import (
    test_temporal_smoothing,
    test_temporal_smoothing_edge_cases,
    test_text_conditioning
)
from .training.test_components import (
    test_training_components,
    test_loss_temporal,
    test_loss_functions,
    test_metrics_computation,
    test_padding_functions,
    test_vae_temporal_handling,
    test_pipeline_components
)
from .test_shapes import (
    test_vae_shapes,
    test_transformer_shapes,
    test_scheduler_config,
    test_resolution_scaling,
    test_padding_edge_cases,
    test_memory_calculation,
    test_metrics_cpu_fallback
)

def run_all_tests():
    """Run all inpainting tests."""
    print("=== Starting CogVideoX Inpainting Tests ===\n")
    
    print("Running test_vae_shapes...")
    test_vae_shapes()
    print("✓ Passed test_vae_shapes\n")
    
    print("Running test_transformer_shapes...")
    test_transformer_shapes()
    print("✓ Passed test_transformer_shapes\n")
    
    print("Running test_scheduler_config...")
    test_scheduler_config()
    print("✓ Passed test_scheduler_config\n")
    
    print("Running test_resolution_scaling...")
    test_resolution_scaling()
    print("✓ Passed test_resolution_scaling\n")
    
    print("Running test_padding_edge_cases...")
    test_padding_edge_cases()
    print("✓ Passed test_padding_edge_cases\n")
    
    print("Running test_memory_calculation...")
    test_memory_calculation()
    print("✓ Passed test_memory_calculation\n")
    
    print("Running test_metrics_cpu_fallback...")
    test_metrics_cpu_fallback()
    print("✓ Passed test_metrics_cpu_fallback\n")
    
    print("Running test_training_components...")
    test_training_components()
    print("✓ Passed test_training_components\n")
    
    print("Running test_loss_temporal...")
    test_loss_temporal()
    print("✓ Passed test_loss_temporal\n")
    
    print("Running test_text_conditioning...")
    test_text_conditioning()
    print("✓ Passed test_text_conditioning\n")
    
    print("Running test_temporal_smoothing...")
    test_temporal_smoothing()
    print("✓ Passed test_temporal_smoothing\n")
    
    print("Running test_temporal_smoothing_edge_cases...")
    test_temporal_smoothing_edge_cases()
    print("✓ Passed test_temporal_smoothing_edge_cases\n")
    
    print("Running test_loss_functions...")
    test_loss_functions()
    print("✓ Passed test_loss_functions\n")
    
    print("Running test_metrics_computation...")
    test_metrics_computation()
    print("✓ Passed test_metrics_computation\n")
    
    print("Running test_padding_functions...")
    test_padding_functions()
    print("✓ Passed test_padding_functions\n")
    
    print("Running test_vae_temporal_handling...")
    test_vae_temporal_handling()
    print("✓ Passed test_vae_temporal_handling\n")
    
    print("Running test_pipeline_components...")
    test_pipeline_components()
    print("✓ Passed test_pipeline_components\n")

if __name__ == "__main__":
    run_all_tests()
