# Circuit-Tracer GNNExplainer

This package provides comprehensive explainability tools for understanding prompt injection detection decisions made by the Circuit-Tracer GraphGPS model.

## Overview

The explainer uses GNNExplainer to generate explanations for graph-level predictions on attribution graphs, helping researchers understand:
- Which graph structures indicate prompt injection attacks
- Key features and patterns in malicious vs benign prompts  
- Model decision boundaries and potential vulnerabilities
- Circuit-tracer specific insights for prompt injection detection

## Installation

The explainer requires the following dependencies:

```bash
# Core dependencies (should already be installed for Circuit-Tracer)
pip install torch torch-geometric
pip install numpy pandas scikit-learn

# Visualization dependencies
pip install matplotlib seaborn networkx

# Optional dependencies for enhanced features
pip install plotly  # Interactive visualizations
```

## Quick Start

### 1. Basic Usage

```python
from explainer import CircuitTracerGNNExplainer, ExplanationVisualizer

# Load your trained GraphGPS model
model = load_your_trained_model()

# Initialize explainer
explainer = CircuitTracerGNNExplainer(model, device='cuda')

# Explain a single graph
explanation = explainer.explain_graph(your_graph_data)

# Visualize results
visualizer = ExplanationVisualizer()
visualizer.plot_subgraph_explanation(explanation, your_graph_data)
```

### 2. Command Line Usage

```bash
# Explain a single graph
python explainer/explain_model.py \
  --model path/to/trained_model.pth \
  --data path/to/dataset.pkl \
  --mode single \
  --graph-index 42

# Batch explain entire dataset
python explainer/explain_model.py \
  --model path/to/trained_model.pth \
  --data path/to/converted_graphs/ \
  --mode batch \
  --max-graphs 100 \
  --use-cache
```

### 3. Batch Processing with Cache

```python
from explainer import ExplanationCache, BatchProcessor

# Setup caching for faster repeated explanations
cache = ExplanationCache("./explanation_cache")

# Process dataset in batches
processor = BatchProcessor(explainer, cache=cache)
explanations = processor.process_dataset(dataset, use_cache=True)

# Analyze results
visualizer.plot_pattern_analysis(explanations.explanations)
```

## Core Components

### 1. CircuitTracerGNNExplainer
Main explainer class that wraps PyTorch Geometric's GNNExplainer with Circuit-Tracer specific configuration.

**Key Features:**
- Graph-level explanation for prompt injection detection
- Optimized hyperparameters for attribution graphs
- Support for both single graph and batch explanation
- Integration with GraphGPS model architecture

### 2. AttributionGraphExplanation
Comprehensive explanation container that stores both raw masks and processed insights.

**Contains:**
- Raw edge and node importance masks
- Top-k important edges, nodes, and features
- Domain-specific suspicious pattern detection
- Circuit-tracer insights (attack depth, manipulation type, etc.)
- Quality metrics (fidelity+, fidelity-, sparsity)

### 3. CircuitTracerExplanationProcessor
Domain-specific processor that converts raw GNNExplainer output to interpretable insights.

**Identifies:**
- High influence concentration (attention hijacking)
- Context position manipulation
- Cross-layer transcoder attacks
- Direct logit manipulation
- Activation anomalies

### 4. ExplanationEvaluator
Comprehensive evaluation metrics for explanation quality assessment.

**Metrics:**
- **Fidelity+** (Necessity): How much prediction changes when removing explanation
- **Fidelity-** (Sufficiency): How well explanation alone predicts
- **Sparsity**: Compactness of explanation
- **Stability**: Consistency across similar graphs

### 5. ExplanationVisualizer
Rich visualization tools for explanation analysis and reporting.

**Generates:**
- Subgraph importance plots with edge/node highlighting
- Feature importance bar charts
- Pattern analysis across multiple explanations
- Comprehensive HTML reports
- Comparative analysis plots

### 6. ExplanationCache & BatchProcessor
Utilities for efficient large-scale explanation generation.

**Features:**
- Automatic caching to avoid recomputation
- Progress tracking for batch processing
- Memory-efficient streaming
- Export to CSV/JSON for further analysis

## Explanation Quality Metrics

The explainer computes several metrics to assess explanation quality:

### Fidelity+ (Necessity)
Measures how much the model's prediction changes when important elements are removed.
- **Range**: 0.0 to 1.0
- **Higher is better**: Indicates explanation captures necessary elements
- **Target**: > 0.8 for high-quality explanations

### Fidelity- (Sufficiency) 
Measures how well the explanation subgraph alone maintains the original prediction.
- **Range**: 0.0 to 1.0  
- **Higher is better**: Indicates explanation is sufficient for prediction
- **Target**: > 0.7 for high-quality explanations

### Sparsity
Fraction of graph elements included in the explanation.
- **Range**: 0.0 to 1.0
- **Lower is better**: Indicates more compact, focused explanations
- **Target**: < 0.1 for interpretable explanations

## Domain-Specific Pattern Detection

The explainer identifies prompt injection attack patterns:

### Attack Patterns
- **High Influence Concentration**: Attention hijacking attacks
- **Context Position Manipulation**: Positional prompt injection
- **Cross-Layer Attack**: Unusual layer activation patterns
- **Direct Logit Manipulation**: Output layer targeting
- **Transcoder Manipulation**: Cross-layer transcoder exploitation
- **Multi-Feature Attack**: Complex attacks using multiple techniques

### Circuit Insights
- **Attack Depth**: Shallow, medium, or deep layer involvement
- **Manipulation Type**: Direct logit, context hijacking, attention steering, etc.
- **Confidence**: Quality of the explanation
- **Anomaly Score**: How unusual the pattern is

## Configuration

### GNNExplainer Parameters

```python
explainer = CircuitTracerGNNExplainer(
    model=model,
    device='cuda',
    epochs=200,          # Optimization epochs
    lr=0.01,            # Learning rate
    edge_size=0.005,    # Edge sparsity regularization
    edge_ent=1.0,       # Edge entropy regularization
    node_feat_size=1.0, # Node feature regularization
    node_feat_ent=0.1   # Node feature entropy
)
```

### Pattern Detection Thresholds

```python
processor = CircuitTracerExplanationProcessor()
processor.update_thresholds({
    'high_influence': 0.5,
    'context_manipulation': 0.3,
    'cross_layer_attack': 0.4,
    'logit_manipulation': 0.2,
    'activation_anomaly': 0.4
})
```

## Output Files

### Single Graph Explanation
- `explanation_report_graph_X.html` - Comprehensive visual report
- `explanation_graph_X.json` - Raw explanation data
- `explanation_summary_graph_X.json` - Summary with metrics
- `explanation_graph_X_subgraph.png` - Subgraph visualization
- `explanation_graph_X_features.png` - Feature importance plot

### Batch Explanation
- `batch_analysis_report.html` - Aggregate analysis report
- `explanations_summary.csv` - Tabular data for analysis
- `explanation_statistics.json` - Detailed statistics
- `pattern_analysis.png` - Pattern frequency visualization
- `individual_explanations/` - Individual explanation files
- `cache/` - Cached explanations for reuse

## Integration with Circuit-Tracer

The explainer integrates seamlessly with the existing Circuit-Tracer pipeline:

1. **Training Phase**: Train GraphGPS model using existing `train.py`
2. **Explanation Phase**: Generate explanations using `explain_model.py`
3. **Analysis Phase**: Analyze patterns and insights in generated reports

### Example Workflow

```bash
# 1. Train model (existing workflow)
python train.py --config config.yaml

# 2. Generate explanations
python explainer/explain_model.py \
  --model checkpoints/best_model.pth \
  --data data/test_graphs/ \
  --mode batch \
  --output-dir explanations/

# 3. Analyze results
open explanations/batch_analysis_report.html
```

## Testing

Test the explainer package:

```bash
cd explainer/
python test_explainer_package.py
```

This will verify all components work correctly and provide feedback on any issues.

## Advanced Usage

### Custom Pattern Detection

```python
# Define custom attack pattern detector
def detect_custom_pattern(features, edges, nodes):
    # Your custom logic here
    return pattern_detected

# Add to processor
processor.custom_patterns['my_pattern'] = detect_custom_pattern
```

### Multi-Model Comparison

```python
# Compare explanations across different models
models = [model1, model2, model3]
explanations = []

for model in models:
    explainer = CircuitTracerGNNExplainer(model)
    exp = explainer.explain_graph(data)
    explanations.append(exp)

# Visualize comparison
visualizer.plot_explanation_comparison(explanations, 
                                     labels=['Model 1', 'Model 2', 'Model 3'])
```

### Synthetic Evaluation

```python
# Create synthetic graphs with known patterns
synthetic_graphs = create_synthetic_injection_patterns()

# Evaluate explainer on synthetic data
results = evaluate_on_synthetic(explainer, synthetic_graphs)
print(f"Pattern detection accuracy: {results}")
```

## Performance Considerations

### Memory Usage
- Large graphs (>10K nodes) may require significant GPU memory
- Use batch processing for memory efficiency
- Consider CPU mode for very large graphs

### Speed Optimization
- Use caching to avoid recomputing explanations
- Reduce explanation epochs for faster (but lower quality) results
- Process in batches for better GPU utilization

### Scalability
- Current implementation handles up to ~1000 graphs efficiently
- For larger datasets, consider sampling representative subsets
- Use distributed processing for very large datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or switch to CPU
   - Use explanation caching to avoid recomputation

2. **Explanation Quality Too Low**
   - Increase number of optimization epochs
   - Check if model is properly trained
   - Verify data preprocessing matches training

3. **Slow Explanation Generation**
   - Use caching for repeated explanations
   - Reduce epochs for faster results
   - Consider parallel processing

### Debug Mode

Enable verbose output for troubleshooting:

```bash
python explainer/explain_model.py --verbose [other args]
```

## Citation

If you use this explainer in your research, please cite:

```bibtex
@software{circuit_tracer_explainer,
  title={Circuit-Tracer GNNExplainer for Prompt Injection Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/circuit-tracer}
}
```

## Contributing

Contributions are welcome! Please see the main Circuit-Tracer repository for contribution guidelines.

## License

This explainer package is part of the Circuit-Tracer project and follows the same license terms.