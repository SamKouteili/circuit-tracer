"""
Main script for generating explanations from trained GraphGPS models
"""

import argparse
import torch
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

from torch_geometric.data import Data, DataLoader
from gnn_explainer_core import CircuitTracerGNNExplainer
from circuit_tracer_processor import CircuitTracerExplanationProcessor
from explanation import AttributionGraphExplanation, ExplanationBatch
from visualization import ExplanationVisualizer
from metrics import ExplanationEvaluator
from utils import ExplanationCache, BatchProcessor, export_explanations_to_csv, compute_explanation_statistics

# Import model and data utilities
try:
    from train.models import GraphGPS
    from train.dataset import create_dataset_from_converted_files
    from train.data_converter import AttributionGraphConverter
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")
    print("Make sure you're running from the correct directory")
    # Try alternative import paths
    try:
        sys.path.append(str(Path(__file__).parent.parent / "train"))
        from models import GraphGPS
        from dataset import create_dataset_from_converted_files
        from data_converter import AttributionGraphConverter
    except ImportError as e2:
        print(f"Could not import with alternative path: {e2}")
        GraphGPS = None


def load_trained_model(model_path: str, device: str = 'cuda') -> GraphGPS:
    """
    Load a trained GraphGPS model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded GraphGPS model
    """
    print(f"📥 Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})
    
    # Default configuration if not stored in checkpoint
    default_config = {
        'num_features': 9,
        'hidden_dim': 64,
        'num_classes': 2,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1
    }
    
    # Merge configurations
    config = {**default_config, **model_config}
    
    # Initialize model
    model = GraphGPS(
        num_features=config['num_features'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Architecture: {config}")
    
    return model, config


def load_dataset_for_explanation(cache_dir: str, 
                                dataset_split: str = 'test',
                                subset_size: Optional[int] = None) -> List[Data]:
    """
    Load cached PyG dataset for explanation generation
    
    Args:
        cache_dir: Path to cache directory containing the PyG dataset files
        dataset_split: Which split to load ('train', 'val', 'test', or 'all')
        subset_size: Optional limit on number of graphs to load
        
    Returns:
        List of PyG Data objects
    """
    print(f"📥 Loading cached dataset from {cache_dir}")
    
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    # Find cached dataset files by pattern matching
    import glob
    import pickle
    
    cache_files = {}
    for split in ['train', 'val', 'test']:
        pattern = os.path.join(cache_dir, f"{split}_dataset_*.pkl")
        matching_files = glob.glob(pattern)
        if matching_files:
            # Use the most recent file if multiple exist
            cache_files[split] = max(matching_files, key=os.path.getmtime)
    
    if not cache_files:
        raise FileNotFoundError(f"No cached dataset files found in {cache_dir}")
    
    print(f"Found cached datasets:")
    for split, path in cache_files.items():
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"  {split}: {os.path.basename(path)} ({file_size:.1f} MB)")
    
    # Load requested dataset split(s)
    datasets_to_load = []
    if dataset_split == 'all':
        datasets_to_load = ['train', 'val', 'test']
    elif dataset_split in cache_files:
        datasets_to_load = [dataset_split]
    else:
        available = list(cache_files.keys())
        raise ValueError(f"Split '{dataset_split}' not available. Available: {available}")
    
    all_data = []
    for split in datasets_to_load:
        if split in cache_files:
            print(f"🔄 Loading {split} dataset...")
            try:
                with open(cache_files[split], 'rb') as f:
                    dataset = pickle.load(f)
                
                # Handle different dataset formats
                if hasattr(dataset, 'data_list'):
                    split_data = dataset.data_list
                elif isinstance(dataset, list):
                    split_data = dataset
                else:
                    raise ValueError(f"Unexpected dataset format: {type(dataset)}")
                
                print(f"  ✅ Loaded {len(split_data)} graphs from {split} split")
                all_data.extend(split_data)
                
            except Exception as e:
                print(f"  ❌ Failed to load {split} dataset: {e}")
                raise
    
    if subset_size is not None and len(all_data) > subset_size:
        print(f"🎯 Using subset of {subset_size} graphs from {len(all_data)} total")
        all_data = all_data[:subset_size]
    
    print(f"✅ Loaded {len(all_data)} graphs for explanation")
    
    return all_data


def explain_single_graph(model_path: str,
                        cache_dir: str,
                        graph_index: int = 0,
                        output_dir: str = "./explanations",
                        device: str = 'cuda',
                        epochs: int = 200,
                        dataset_split: str = 'test',
                        visualize: bool = True) -> str:
    """
    Generate explanation for a single graph
    
    Args:
        model_path: Path to trained model
        cache_dir: Path to cache directory with PyG datasets
        graph_index: Index of graph to explain
        output_dir: Directory to save results
        device: Device for computation
        epochs: Number of explanation epochs
        dataset_split: Which dataset split to use ('train', 'val', 'test')
        visualize: Whether to generate visualizations
        
    Returns:
        Path to generated explanation report
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model and data
    model, model_config = load_trained_model(model_path, device)
    dataset = load_dataset_for_explanation(cache_dir, dataset_split)
    
    if graph_index >= len(dataset):
        raise ValueError(f"Graph index {graph_index} out of range (dataset size: {len(dataset)})")
    
    target_data = dataset[graph_index]
    graph_id = f"graph_{graph_index}"
    
    print(f"🎯 Explaining graph {graph_index} ({graph_id})")
    print(f"   Nodes: {target_data.x.size(0)}, Edges: {target_data.edge_index.size(1)}")
    
    # Initialize explainer
    explainer = CircuitTracerGNNExplainer(
        model=model,
        device=device,
        epochs=epochs
    )
    
    # Generate explanation
    print("🔄 Generating explanation...")
    start_time = time.time()
    
    try:
        raw_explanation = explainer.explain_graph(target_data)
        
        # Process explanation
        processor = CircuitTracerExplanationProcessor()
        explanation = processor.process_explanation(
            raw_explanation, 
            target_data,
            graph_id=graph_id
        )
        
        explanation_time = time.time() - start_time
        explanation.explanation_time = explanation_time
        
        print(f"✅ Explanation generated in {explanation_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Failed to generate explanation: {e}")
        raise
    
    # Evaluate explanation quality
    print("📊 Evaluating explanation quality...")
    evaluator = ExplanationEvaluator(model, device)
    metrics = evaluator.evaluate_explanation(explanation, target_data)
    
    # Update explanation with computed metrics
    explanation.fidelity_plus = metrics.get('fidelity_plus', 0.0)
    explanation.fidelity_minus = metrics.get('fidelity_minus', 0.0)
    
    # Generate visualizations
    if visualize:
        print("🎨 Creating visualizations...")
        visualizer = ExplanationVisualizer()
        
        # Create comprehensive report
        report_path = output_path / f"explanation_report_{graph_id}.html"
        visualizer.create_explanation_report(
            explanation, 
            target_data,
            save_path=str(report_path),
            include_subgraph=True,
            include_features=True
        )
        
        print(f"📄 Report saved to {report_path}")
    
    # Save explanation data
    explanation_json_path = output_path / f"explanation_{graph_id}.json"
    explanation.save_json(str(explanation_json_path))
    
    # Save summary
    summary = {
        "graph_id": graph_id,
        "graph_index": graph_index,
        "model_path": model_path,
        "explanation_time": explanation_time,
        "metrics": metrics,
        "explanation": explanation.to_dict()
    }
    
    summary_path = output_path / f"explanation_summary_{graph_id}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Explanation data saved to {explanation_json_path}")
    print(f"📋 Summary saved to {summary_path}")
    
    # Print explanation summary
    print("\n" + "="*50)
    print(explanation.summary())
    print("="*50)
    
    return str(report_path) if visualize else str(explanation_json_path)


def explain_dataset_batch(model_path: str,
                         cache_dir: str,
                         output_dir: str = "./batch_explanations",
                         device: str = 'cuda',
                         epochs: int = 200,
                         batch_size: int = 32,
                         max_graphs: Optional[int] = None,
                         dataset_split: str = 'test',
                         use_cache: bool = True,
                         generate_report: bool = True) -> str:
    """
    Generate explanations for entire dataset in batches
    
    Args:
        model_path: Path to trained model
        cache_dir: Path to cache directory with PyG datasets
        output_dir: Directory to save results
        device: Device for computation
        epochs: Number of explanation epochs
        batch_size: Batch size for processing
        max_graphs: Maximum number of graphs to explain
        dataset_split: Which dataset split to use ('train', 'val', 'test', 'all')
        use_cache: Whether to use explanation cache
        generate_report: Whether to generate analysis report
        
    Returns:
        Path to batch analysis report
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize cache
    cache = None
    if use_cache:
        explanation_cache_dir = output_path / "explanation_cache"
        cache = ExplanationCache(str(explanation_cache_dir))
        print(f"💾 Using explanation cache: {explanation_cache_dir}")
        print(f"   Cache stats: {cache.get_cache_stats()}")
    
    # Load model and data
    model, model_config = load_trained_model(model_path, device)
    dataset = load_dataset_for_explanation(cache_dir, dataset_split, subset_size=max_graphs)
    
    print(f"🎯 Explaining {len(dataset)} graphs in batches of {batch_size}")
    
    # Initialize explainer and processor
    explainer = CircuitTracerGNNExplainer(
        model=model,
        device=device,
        epochs=epochs
    )
    
    processor = CircuitTracerExplanationProcessor()
    
    # Setup batch processor
    batch_processor = BatchProcessor(
        explainer=explainer,
        cache=cache,
        batch_size=batch_size
    )
    
    # Generate graph IDs
    graph_ids = [f"graph_{i}" for i in range(len(dataset))]
    
    # Process all graphs
    print("🔄 Starting batch explanation generation...")
    start_time = time.time()
    
    try:
        # Custom batch processing with our processor
        explanations = []
        
        for i, (data, graph_id) in enumerate(zip(dataset, graph_ids)):
            explanation = None
            
            # Try cache first
            if cache is not None:
                explanation = cache.load_explanation(
                    graph_id, 
                    model,
                    explainer.config
                )
            
            # Generate if not cached
            if explanation is None:
                try:
                    raw_explanation = explainer.explain_graph(data)
                    explanation = processor.process_explanation(
                        raw_explanation, 
                        data,
                        graph_id=graph_id
                    )
                    
                    # Cache the result
                    if cache is not None:
                        cache.save_explanation(explanation, model, explainer.config)
                        
                except Exception as e:
                    print(f"⚠️ Failed to explain {graph_id}: {e}")
                    explanation = None
            
            explanations.append(explanation)
            
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(dataset) - i - 1) / rate if rate > 0 else 0
                
                print(f"  Progress: {i + 1}/{len(dataset)} ({(i + 1)/len(dataset)*100:.1f}%) "
                      f"- {rate:.1f} graphs/sec - ETA: {eta:.0f}s")
        
        # Filter successful explanations
        successful_explanations = [exp for exp in explanations if exp is not None]
        explanation_batch = ExplanationBatch(successful_explanations)
        
        total_time = time.time() - start_time
        
        print(f"✅ Batch processing completed in {total_time:.1f}s")
        print(f"   Generated {len(successful_explanations)}/{len(dataset)} explanations")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        raise
    
    # Evaluate explanation quality
    print("📊 Evaluating explanation quality...")
    evaluator = ExplanationEvaluator(model, device)
    
    # Update explanations with quality metrics
    for explanation, data in zip(successful_explanations, dataset[:len(successful_explanations)]):
        try:
            metrics = evaluator.evaluate_explanation(explanation, data)
            explanation.fidelity_plus = metrics.get('fidelity_plus', 0.0)
            explanation.fidelity_minus = metrics.get('fidelity_minus', 0.0)
        except Exception as e:
            print(f"⚠️ Failed to evaluate {explanation.graph_id}: {e}")
    
    # Save individual explanations
    explanations_dir = output_path / "individual_explanations"
    explanation_batch.save_batch(str(explanations_dir))
    
    # Export to CSV for analysis
    csv_path = output_path / "explanations_summary.csv"
    export_explanations_to_csv(successful_explanations, str(csv_path))
    
    # Compute comprehensive statistics
    stats = compute_explanation_statistics(successful_explanations)
    
    # Save statistics
    stats_path = output_path / "explanation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate analysis report
    report_path = None
    if generate_report:
        print("📄 Generating batch analysis report...")
        
        visualizer = ExplanationVisualizer()
        
        # Create pattern analysis plot
        pattern_plot_path = output_path / "pattern_analysis.png"
        visualizer.plot_pattern_analysis(
            successful_explanations,
            save_path=str(pattern_plot_path)
        )
        
        # Generate HTML report
        report_path = output_path / "batch_analysis_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Explanation Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .image {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Batch Explanation Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h3>Processing Summary</h3>
        <div class="metric"><strong>Total Graphs:</strong> {len(dataset)}</div>
        <div class="metric"><strong>Successful Explanations:</strong> {len(successful_explanations)}</div>
        <div class="metric"><strong>Success Rate:</strong> {len(successful_explanations)/len(dataset)*100:.1f}%</div>
        <div class="metric"><strong>Processing Time:</strong> {total_time:.1f}s</div>
        <div class="metric"><strong>Average Time per Graph:</strong> {total_time/len(dataset):.2f}s</div>
    </div>
    
    <div class="section">
        <h3>Prediction Accuracy</h3>
        <div class="metric"><strong>Overall Accuracy:</strong> {stats['summary']['accuracy']:.3f}</div>
        <div class="metric"><strong>Benign Graphs:</strong> {stats['summary']['label_distribution']['benign']}</div>
        <div class="metric"><strong>Injected Graphs:</strong> {stats['summary']['label_distribution']['injected']}</div>
    </div>
    
    <div class="section">
        <h3>Explanation Quality Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Median</th></tr>
            <tr><td>Fidelity+ (Necessity)</td><td>{stats['quality_metrics']['fidelity_plus']['mean']:.3f}</td><td>{stats['quality_metrics']['fidelity_plus']['std']:.3f}</td><td>{stats['quality_metrics']['fidelity_plus']['median']:.3f}</td></tr>
            <tr><td>Fidelity- (Sufficiency)</td><td>{stats['quality_metrics']['fidelity_minus']['mean']:.3f}</td><td>{stats['quality_metrics']['fidelity_minus']['std']:.3f}</td><td>{stats['quality_metrics']['fidelity_minus']['median']:.3f}</td></tr>
            <tr><td>Sparsity</td><td>{stats['quality_metrics']['sparsity']['mean']:.3f}</td><td>{stats['quality_metrics']['sparsity']['std']:.3f}</td><td>{stats['quality_metrics']['sparsity']['median']:.3f}</td></tr>
            <tr><td>Prediction Confidence</td><td>{stats['quality_metrics']['confidence']['mean']:.3f}</td><td>{stats['quality_metrics']['confidence']['std']:.3f}</td><td>{stats['quality_metrics']['confidence']['median']:.3f}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h3>Pattern Analysis</h3>
        <p><strong>Total Patterns Detected:</strong> {stats['patterns']['total_patterns']}</p>
        <p><strong>Unique Pattern Types:</strong> {stats['patterns']['unique_patterns']}</p>
        
        <h4>Most Common Patterns:</h4>
        <table>
            <tr><th>Pattern</th><th>Frequency</th></tr>
            {chr(10).join(f'<tr><td>{pattern.replace("_", " ").title()}</td><td>{count}</td></tr>' 
                         for pattern, count in stats['patterns']['most_common'][:10])}
        </table>
    </div>
    
    <div class="section">
        <h3>Pattern Analysis Visualization</h3>
        <div class="image">
            <img src="pattern_analysis.png" alt="Pattern Analysis" style="max-width: 100%; height: auto;">
        </div>
    </div>
    
    <div class="section">
        <h3>Files Generated</h3>
        <ul>
            <li><a href="explanations_summary.csv">explanations_summary.csv</a> - Tabular data for analysis</li>
            <li><a href="explanation_statistics.json">explanation_statistics.json</a> - Detailed statistics</li>
            <li><a href="individual_explanations/">individual_explanations/</a> - Individual explanation files</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Analysis report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH EXPLANATION SUMMARY")
    print("="*60)
    print(f"Processed: {len(successful_explanations)}/{len(dataset)} graphs")
    print(f"Accuracy: {stats['summary']['accuracy']:.3f}")
    print(f"Average Fidelity+: {stats['quality_metrics']['fidelity_plus']['mean']:.3f}")
    print(f"Average Sparsity: {stats['quality_metrics']['sparsity']['mean']:.3f}")
    print(f"Total Patterns: {stats['patterns']['total_patterns']}")
    print(f"Processing Time: {total_time:.1f}s")
    print("="*60)
    
    return str(report_path) if report_path else str(output_path)


def main():
    """Main entry point for explanation generation"""
    parser = argparse.ArgumentParser(
        description='Generate explanations for Circuit-Tracer GraphGPS model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain a single graph from test set
  python explain_model.py --model trained_model.pth --cache-dir ./cache --mode single --graph-index 42
  
  # Explain test dataset in batch
  python explain_model.py --model trained_model.pth --cache-dir ./cache --mode batch --max-graphs 100
  
  # Fast batch processing with explanation cache
  python explain_model.py --model trained_model.pth --cache-dir ./cache --mode batch --use-cache --epochs 100
  
  # Explain all splits (train+val+test)
  python explain_model.py --model trained_model.pth --cache-dir ./cache --mode batch --split all
        """
    )
    
    # Required arguments
    parser.add_argument('--model', required=True, type=str,
                       help='Path to trained GraphGPS model checkpoint')
    parser.add_argument('--cache-dir', required=True, type=str,
                       help='Path to cache directory containing PyG dataset files')
    
    # Mode selection
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Explanation mode: single graph or batch processing')
    
    # Dataset options
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'val', 'test', 'all'],
                       help='Which dataset split to use for explanation')
    
    # Single graph options
    parser.add_argument('--graph-index', type=int, default=0,
                       help='Index of graph to explain (single mode)')
    
    # Batch processing options
    parser.add_argument('--max-graphs', type=int, default=None,
                       help='Maximum number of graphs to explain (batch mode)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    
    # Explanation parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of optimization epochs for explanation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for computation (cuda/cpu)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./explanations',
                       help='Directory to save explanation results')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation (batch mode)')
    
    # Cache options
    parser.add_argument('--use-cache', action='store_true',
                       help='Use explanation cache to avoid recomputation')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear explanation cache before processing')
    
    # Debugging options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.cache_dir):
        print(f"❌ Cache directory not found: {args.cache_dir}")
        sys.exit(1)
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"🚀 Starting explanation generation")
    print(f"   Model: {args.model}")
    print(f"   Cache Dir: {args.cache_dir}")
    print(f"   Split: {args.split}")
    print(f"   Mode: {args.mode}")
    print(f"   Device: {args.device}")
    print(f"   Output: {args.output_dir}")
    
    try:
        if args.mode == 'single':
            result_path = explain_single_graph(
                model_path=args.model,
                cache_dir=args.cache_dir,
                graph_index=args.graph_index,
                output_dir=args.output_dir,
                device=args.device,
                epochs=args.epochs,
                dataset_split=args.split,
                visualize=not args.no_visualize
            )
            print(f"🎉 Single graph explanation completed: {result_path}")
            
        elif args.mode == 'batch':
            result_path = explain_dataset_batch(
                model_path=args.model,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_graphs=args.max_graphs,
                dataset_split=args.split,
                use_cache=args.use_cache,
                generate_report=not args.no_report
            )
            print(f"🎉 Batch explanation completed: {result_path}")
    
    except Exception as e:
        print(f"❌ Explanation generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()