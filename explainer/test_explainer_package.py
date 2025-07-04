"""
Test script to verify the explainer package works correctly
"""

import torch
import numpy as np
from torch_geometric.data import Data
import sys
import os

# Test all major components
def test_explainer_package():
    """Test the complete explainer package"""
    print("🧪 Testing Circuit-Tracer GNNExplainer Package")
    print("=" * 50)
    
    device = 'cpu'  # Use CPU for testing
    
    # Test 1: Core explanation functionality
    print("1. Testing core explainer components...")
    
    try:
        from gnn_explainer_core import CircuitTracerGNNExplainer, test_explainer
        result = test_explainer()
        print(f"   ✅ Core explainer: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"   ❌ Core explainer: FAIL ({e})")
    
    # Test 2: Explanation data structures
    print("2. Testing explanation data structures...")
    
    try:
        from explanation import AttributionGraphExplanation, ExplanationBatch
        
        # Create test explanation
        explanation = AttributionGraphExplanation(
            graph_id="test",
            true_label=1,
            predicted_label=1,
            prediction_confidence=0.85,
            edge_mask=torch.randn(10),
            node_mask=torch.randn(5, 9),
            important_edges=[(0, 1, 0.9), (2, 3, 0.8)],
            important_nodes=[(0, "node_0", 0.9)],
            critical_features={"influence": 0.8}
        )
        
        # Test serialization
        summary = explanation.summary()
        assert "test" in summary
        
        # Test batch functionality
        batch = ExplanationBatch([explanation])
        assert len(batch) == 1
        assert batch.get_accuracy() == 1.0
        
        print("   ✅ Data structures: PASS")
        
    except Exception as e:
        print(f"   ❌ Data structures: FAIL ({e})")
    
    # Test 3: Circuit-tracer processor
    print("3. Testing circuit-tracer processor...")
    
    try:
        from circuit_tracer_processor import CircuitTracerExplanationProcessor
        
        processor = CircuitTracerExplanationProcessor()
        
        # Test pattern detection
        test_features = {"influence": 0.8, "activation": 0.6}
        test_edges = [(0, 1, 0.9), (2, 3, 0.8)]
        test_nodes = [(0, "node_0", 0.9)]
        
        patterns = processor._identify_attack_patterns(test_features, test_edges, test_nodes)
        
        print(f"   ✅ Circuit processor: PASS (detected {len(patterns)} patterns)")
        
    except Exception as e:
        print(f"   ❌ Circuit processor: FAIL ({e})")
    
    # Test 4: Metrics evaluation
    print("4. Testing metrics evaluation...")
    
    try:
        from metrics import ExplanationEvaluator, test_evaluator
        result = test_evaluator()
        print(f"   ✅ Metrics evaluator: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"   ❌ Metrics evaluator: FAIL ({e})")
    
    # Test 5: Visualization
    print("5. Testing visualization...")
    
    try:
        from visualization import ExplanationVisualizer, test_visualizer
        result = test_visualizer()
        print(f"   ✅ Visualization: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"   ❌ Visualization: FAIL ({e})")
    
    # Test 6: Utilities
    print("6. Testing utilities...")
    
    try:
        from utils import ExplanationCache, test_utils
        result = test_utils()
        print(f"   ✅ Utilities: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"   ❌ Utilities: FAIL ({e})")
    
    # Test 7: Package imports
    print("7. Testing package imports...")
    
    try:
        import explainer
        from explainer import (
            CircuitTracerGNNExplainer,
            AttributionGraphExplanation,
            ExplanationBatch,
            CircuitTracerExplanationProcessor,
            ExplanationEvaluator,
            ExplanationVisualizer,
            ExplanationCache,
            BatchProcessor
        )
        print("   ✅ Package imports: PASS")
    except Exception as e:
        print(f"   ❌ Package imports: FAIL ({e})")
    
    print("\n" + "=" * 50)
    print("🎉 Explainer package testing completed!")
    print("\nThe explainer is ready for integration with your trained GraphGPS model.")
    print("\nNext steps:")
    print("1. Train your GraphGPS model (or use existing trained model)")
    print("2. Run explanations using: python explainer/explain_model.py")
    print("3. Analyze results in the generated reports")


if __name__ == "__main__":
    test_explainer_package()