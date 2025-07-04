# GNNExplainer Implementation Plan for Circuit-Tracer

## Overview

This document outlines the implementation plan for adding GNNExplainer capabilities to the circuit-tracer project for explaining prompt injection detection decisions made by our GraphGPS model.

## Background & Motivation

### Current State
- ✅ **Working GraphGPS Model**: 89.38% accuracy on prompt injection detection
- ✅ **Balanced Dataset**: 6K attribution graphs (3K benign, 3K injected)
- ✅ **Clean Feature Pipeline**: Fixed mixed semantics in node features

### Goals
- **Interpretability**: Understand which graph structures indicate prompt injection
- **Trust**: Provide evidence for model decisions in security-critical applications  
- **Discovery**: Identify novel prompt injection patterns and techniques
- **Debugging**: Validate model is learning meaningful patterns vs spurious correlations

## Literature Review Summary

### GNNExplainer (Ying et al., NeurIPS 2019)
- **Approach**: Model-agnostic explanation via mutual information maximization
- **Output**: Subgraph + feature importance masks
- **Strengths**: Works with any GNN, identifies domain-specific motifs
- **Limitations**: Computationally expensive, single-instance explanations

### Alternative Methods Considered
- **PGExplainer**: More stable but edge-only explanations
- **GraphLIME**: Better for complex local patterns but more complex
- **SubgraphX**: Better for large subgraphs but requires Shapley values

**Decision**: Start with GNNExplainer as the gold standard, expand later if needed.

## Technical Implementation Plan

### Phase 1: Basic GNNExplainer Integration (Week 1-2)

#### 1.1 Setup and Dependencies
```python
# Additional dependencies needed
pip install torch-geometric[full]
pip install matplotlib seaborn  # For visualization
pip install networkx  # For graph manipulation
```

#### 1.2 Core Implementation Structure
```
circuit-tracer/train/
├── explainer/
│   ├── __init__.py
│   ├── gnn_explainer.py      # Main explainer implementation
│   ├── explanation.py        # Explanation data structures
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Explanation visualization
│   └── utils.py              # Helper functions
├── explain_model.py          # Main explanation script
└── evaluate_explanations.py  # Evaluation script
```

#### 1.3 Basic Explainer Implementation
```python
# explainer/gnn_explainer.py
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

class CircuitTracerExplainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(
                epochs=200,          # Optimization epochs
                lr=0.01,            # Learning rate
                edge_size=0.005,    # Sparsity regularization
                edge_ent=1.0,       # Entropy regularization
                node_feat_size=1.0, # Node feature regularization
            ),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',  # Graph-level classification
                return_type='probs'
            )
        )
    
    def explain_graph(self, data):
        """Generate explanation for a single graph"""
        return self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        )
```

### Phase 2: Explanation Analysis (Week 3)

#### 2.1 Explanation Data Structure
```python
# explainer/explanation.py
@dataclass
class GraphExplanation:
    """Container for graph explanation results"""
    graph_id: str
    true_label: int
    predicted_label: int
    confidence: float
    
    # Explanation masks
    edge_mask: torch.Tensor     # Importance of each edge
    node_mask: torch.Tensor     # Importance of each node feature
    
    # Derived explanations
    important_edges: List[Tuple[int, int]]    # Top-k edges
    important_nodes: List[int]                # Top-k nodes
    subgraph: torch.Tensor                    # Minimal explaining subgraph
    
    # Metadata
    explanation_time: float
    fidelity_score: float
    sparsity_score: float
```

#### 2.2 Evaluation Metrics
```python
# explainer/metrics.py
class ExplanationMetrics:
    @staticmethod
    def fidelity_plus(model, original_data, masked_data):
        """Necessity: prediction change when removing explanation"""
        orig_pred = model(original_data)
        masked_pred = model(masked_data)
        return (orig_pred - masked_pred).abs().mean()
    
    @staticmethod  
    def fidelity_minus(model, subgraph_data, empty_data):
        """Sufficiency: prediction with only explanation"""
        sub_pred = model(subgraph_data)
        empty_pred = model(empty_data)
        return (sub_pred - empty_pred).abs().mean()
    
    @staticmethod
    def sparsity(explanation, total_elements):
        """Fraction of elements in explanation"""
        return explanation.sum() / total_elements
    
    @staticmethod
    def stability(explanations_list):
        """Jaccard similarity between explanations"""
        # Implementation for explanation consistency
        pass
```

### Phase 3: Visualization & Interpretation (Week 4)

#### 3.1 Visualization Pipeline
```python
# explainer/visualization.py
class ExplanationVisualizer:
    def __init__(self, feature_names, node_vocab):
        self.feature_names = feature_names
        self.node_vocab = node_vocab
    
    def plot_subgraph_explanation(self, explanation, save_path=None):
        """Visualize important subgraph with edge weights"""
        # Convert to NetworkX for visualization
        # Color edges by importance
        # Annotate nodes with feature importance
        pass
    
    def plot_feature_importance(self, explanation, save_path=None):
        """Bar plot of node feature importance"""
        # Show which features (influence, activation, etc.) matter most
        pass
    
    def create_explanation_report(self, explanation, save_path=None):
        """Generate comprehensive HTML/PDF report"""
        # Combine visualizations with metrics and interpretations
        pass
```

#### 3.2 Domain-Specific Interpretation
```python
# explainer/interpretation.py
class PromptInjectionInterpreter:
    def __init__(self, converter):
        self.converter = converter
        
    def interpret_explanation(self, explanation, original_graph):
        """Convert explanation to domain insights"""
        interpretations = {
            'suspicious_patterns': [],
            'key_features': [],
            'injection_indicators': [],
            'benign_indicators': []
        }
        
        # Analyze important edges for attack patterns
        # Identify suspicious feature combinations
        # Map to circuit-tracer concepts
        
        return interpretations
```

### Phase 4: Evaluation & Validation (Week 5)

#### 4.1 Synthetic Ground Truth Evaluation
```python
# Create synthetic test cases with known injection patterns
def create_synthetic_injection_patterns():
    """Generate graphs with known malicious substructures"""
    patterns = {
        'direct_injection': create_direct_injection_pattern(),
        'indirect_manipulation': create_indirect_pattern(),
        'context_hijacking': create_context_pattern()
    }
    return patterns

def evaluate_on_synthetic(explainer, synthetic_graphs):
    """Test if explainer identifies known patterns"""
    results = {}
    for pattern_name, graphs in synthetic_graphs.items():
        explanations = [explainer.explain_graph(g) for g in graphs]
        accuracy = compute_pattern_detection_accuracy(explanations, pattern_name)
        results[pattern_name] = accuracy
    return results
```

#### 4.2 Real Data Validation
```python
def validate_explanations_on_real_data(explainer, test_dataset):
    """Comprehensive evaluation on real attribution graphs"""
    
    metrics = {
        'fidelity_plus': [],
        'fidelity_minus': [],
        'sparsity': [],
        'stability': [],
        'prediction_consistency': []
    }
    
    # Generate explanations for test set
    # Compute all metrics
    # Analyze patterns across benign vs injected
    
    return metrics
```

### Phase 5: Production Integration (Week 6)

#### 5.1 Explanation API
```python
# explain_model.py - Main explanation script
def main():
    parser = argparse.ArgumentParser(description='Explain GraphGPS predictions')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_path', required=True, help='Path to graph to explain')
    parser.add_argument('--output_dir', default='./explanations', help='Output directory')
    parser.add_argument('--top_k', type=int, default=10, help='Top K elements to highlight')
    
    args = parser.parse_args()
    
    # Load model and data
    # Generate explanation
    # Create visualizations
    # Save results
```

#### 5.2 Batch Explanation Pipeline
```python
def explain_dataset_batch(model_path, dataset_path, output_dir):
    """Generate explanations for entire dataset"""
    
    # Setup explainer
    # Process in batches to manage memory
    # Generate summary statistics
    # Create aggregate reports
    
    pass
```

## Expected Outputs & Deliverables

### Week 1-2: Basic Implementation
- ✅ Working GNNExplainer integration with PyTorch Geometric
- ✅ Basic explanation generation for single graphs
- ✅ Unit tests for core functionality

### Week 3: Analysis Tools
- ✅ Explanation data structures and containers
- ✅ Evaluation metrics implementation (fidelity, sparsity, stability)
- ✅ Basic visualization capabilities

### Week 4: Visualization & Interpretation
- ✅ Subgraph visualization with importance highlighting
- ✅ Feature importance plots
- ✅ Domain-specific interpretation tools
- ✅ HTML/PDF report generation

### Week 5: Validation
- ✅ Synthetic pattern detection evaluation
- ✅ Real data validation results
- ✅ Comprehensive metrics analysis
- ✅ Comparison with baseline explanation methods

### Week 6: Production Ready
- ✅ Command-line explanation tools
- ✅ Batch processing capabilities
- ✅ Integration with existing training pipeline
- ✅ Documentation and examples

## Evaluation Strategy

### Quantitative Metrics
1. **Fidelity+**: Necessity of explanation (target: >0.8)
2. **Fidelity-**: Sufficiency of explanation (target: >0.7)  
3. **Sparsity**: Explanation compactness (target: <0.1)
4. **Stability**: Consistency across similar graphs (target: >0.8)

### Qualitative Analysis
1. **Domain Expert Review**: Manual inspection by security researchers
2. **Pattern Discovery**: Novel injection techniques identified
3. **Interpretability**: Clear mapping to circuit-tracer concepts
4. **Actionability**: Explanations enable defensive measures

### Benchmarking
1. **Comparison with Baselines**: Random explanations, gradient-based methods
2. **Cross-Model Validation**: Test explanations across different architectures
3. **Human Studies**: User comprehension and trust metrics

## Technical Considerations

### Performance Optimization
- **Caching**: Store explanations to avoid recomputation
- **Parallelization**: Batch explanation generation
- **Memory Management**: Handle large graphs efficiently
- **GPU Utilization**: Optimize for CUDA when available

### Scalability Challenges
- **Large Graphs**: Current GNNExplainer doesn't scale well >10K nodes
- **Batch Processing**: Memory constraints for multiple explanations
- **Storage**: Explanation results can be large

### Mitigation Strategies
- **Sampling**: Explain representative subsets for large datasets
- **Approximation**: Use faster explanation methods for initial screening
- **Hierarchical**: Multi-level explanations (coarse-grained → fine-grained)

## Domain-Specific Considerations

### Prompt Injection Context
- **Attack Vectors**: Direct injection, context manipulation, jailbreaking
- **Security Relevance**: High-stakes decisions require explainable AI
- **Adversarial Robustness**: Explanations should be robust to evasion attempts

### Attribution Graph Properties
- **Node Types**: Feature nodes, token nodes, logit nodes, error nodes
- **Edge Semantics**: Influence relationships, activation flows
- **Scale**: Large graphs (1K+ nodes) with complex connectivity

### Interpretation Challenges
- **Circuit Concepts**: Map explanations to interpretable circuit concepts
- **Feature Interactions**: Complex relationships between node features
- **Temporal Aspects**: How injection unfolds through computation graph

## Risk Assessment & Mitigation

### Technical Risks
1. **Explanation Quality**: Poor explanations worse than no explanations
   - *Mitigation*: Comprehensive evaluation on synthetic and real data
   
2. **Computational Cost**: Explanations too slow for practical use
   - *Mitigation*: Optimize implementation, consider faster alternatives
   
3. **Scalability Issues**: Cannot handle large production graphs
   - *Mitigation*: Implement sampling and approximation strategies

### Security Risks
1. **Adversarial Explanations**: Attackers manipulate explanations
   - *Mitigation*: Evaluate robustness, implement defense mechanisms
   
2. **Information Leakage**: Explanations reveal model vulnerabilities  
   - *Mitigation*: Careful analysis of what explanations expose

3. **False Confidence**: Poor explanations increase user trust inappropriately
   - *Mitigation*: Clear uncertainty quantification and limitations

## Success Criteria

### Technical Success
- [ ] Generate explanations for >90% of test graphs within 30 seconds
- [ ] Achieve fidelity+ > 0.8 and sparsity < 0.1 on evaluation dataset
- [ ] Explanation stability > 0.8 across similar graphs

### Interpretability Success  
- [ ] Domain experts can understand explanations without ML background
- [ ] Explanations correctly identify synthetic injection patterns >85% accuracy
- [ ] Novel insights discovered about prompt injection techniques

### Adoption Success
- [ ] Integration with existing circuit-tracer workflow
- [ ] Positive feedback from security researchers and practitioners
- [ ] Published evaluation results and open-source availability

## Future Enhancements

### Short-term (3-6 months)
- **Multi-Model Explanations**: Compare explanations across different architectures
- **Counterfactual Analysis**: "What would make this benign/malicious?"
- **Real-time Explanation**: Optimize for production latency requirements

### Medium-term (6-12 months)
- **Explanation-Guided Training**: Use explanations to improve model training
- **Adaptive Explanations**: Tailor explanations to user expertise level
- **Fairness Analysis**: Ensure explanations don't discriminate unfairly

### Long-term (1+ years)
- **Causal Explanations**: Move beyond correlation to causation
- **Interactive Explanations**: Allow users to probe and refine explanations
- **Multi-modal Explanations**: Combine graph structure with text analysis

## Resources & Timeline

### Personnel (assuming 1 developer)
- **40 hours/week** × **6 weeks** = **240 total hours**
- **Research & Learning**: 40 hours (Week 1)
- **Core Implementation**: 80 hours (Weeks 2-3) 
- **Visualization**: 40 hours (Week 4)
- **Evaluation**: 40 hours (Week 5)
- **Production Integration**: 40 hours (Week 6)

### Computational Resources
- **GPU Requirements**: CUDA-capable GPU for explanation generation
- **Storage**: ~1-10 GB for explanation caches and visualizations
- **Memory**: 16+ GB RAM for large graph processing

### External Dependencies
- PyTorch Geometric latest version
- Matplotlib/Seaborn for visualization  
- NetworkX for graph manipulation
- Optional: Plotly for interactive visualizations

---

This implementation plan provides a comprehensive roadmap for adding explainability to the circuit-tracer project, balancing research rigor with practical deployment considerations. The phased approach allows for iterative development and validation while building towards a production-ready explanation system.