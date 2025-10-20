"""
Visualization utilities for the License Plate Recognition System.

This module provides visualization functions for detection results,
analytics, and system performance monitoring.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LicensePlateVisualizer:
    """Visualization utilities for license plate detection results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def visualize_detection_result(self, 
                                  original_image: np.ndarray,
                                  plate_text: str,
                                  bounding_box: Tuple[int, int, int, int],
                                  confidence: float,
                                  plate_image: Optional[np.ndarray] = None,
                                  method: str = "Detection") -> plt.Figure:
        """
        Create a comprehensive visualization of detection results.
        
        Args:
            original_image: Original input image
            plate_text: Detected license plate text
            bounding_box: Bounding box coordinates (x, y, w, h)
            confidence: Detection confidence score
            plate_image: Extracted license plate image
            method: Detection method name
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2 if plate_image is not None else 1, 
                                figsize=self.figsize)
        
        if plate_image is not None:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = [axes]
        
        # Plot original image with bounding box
        ax1 = axes[0]
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        # Draw bounding box
        x, y, w, h = bounding_box
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                               edgecolor='green', facecolor='none')
        ax1.add_patch(rect)
        
        # Add text annotation
        ax1.text(x, y-10, f'{plate_text} ({confidence:.2f})', 
                fontsize=12, color='green', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax1.set_title(f'{method} Result', fontsize=14, weight='bold')
        ax1.axis('off')
        
        # Plot extracted license plate if available
        if plate_image is not None and len(axes) > 1:
            ax2 = axes[1]
            if len(plate_image.shape) == 3:
                ax2.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            else:
                ax2.imshow(plate_image, cmap='gray')
            
            ax2.set_title(f'Extracted Plate: {plate_text}', fontsize=14, weight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_confidence_histogram(self, confidences: List[float], 
                                   title: str = "Confidence Score Distribution") -> plt.Figure:
        """
        Create a histogram of confidence scores.
        
        Args:
            confidences: List of confidence scores
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(confidences):.2f}')
        ax.axvline(np.median(confidences), color='green', linestyle='--', 
                  label=f'Median: {np.median(confidences):.2f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_processing_time_chart(self, processing_times: List[float],
                                   methods: Optional[List[str]] = None,
                                   title: str = "Processing Time Analysis") -> plt.Figure:
        """
        Create a chart showing processing times.
        
        Args:
            processing_times: List of processing times
            methods: Optional list of method names
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(processing_times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.axvline(np.mean(processing_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(processing_times):.2f}s')
        ax1.set_xlabel('Processing Time (seconds)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Processing Time Distribution', fontsize=12, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot if methods are provided
        if methods:
            df = pd.DataFrame({'time': processing_times, 'method': methods})
            sns.boxplot(data=df, x='method', y='time', ax=ax2)
            ax2.set_title('Processing Time by Method', fontsize=12, weight='bold')
            ax2.set_xlabel('Detection Method', fontsize=12)
            ax2.set_ylabel('Processing Time (seconds)', fontsize=12)
            plt.setp(ax2.get_xticklabels(), rotation=45)
        else:
            ax2.boxplot(processing_times)
            ax2.set_title('Processing Time Box Plot', fontsize=12, weight='bold')
            ax2.set_ylabel('Processing Time (seconds)', fontsize=12)
        
        plt.suptitle(title, fontsize=14, weight='bold')
        plt.tight_layout()
        return fig
    
    def create_method_comparison(self, results_data: List[Dict[str, Any]]) -> plt.Figure:
        """
        Create a comparison chart between different detection methods.
        
        Args:
            results_data: List of result dictionaries
            
        Returns:
            Matplotlib figure object
        """
        df = pd.DataFrame(results_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confidence comparison
        sns.boxplot(data=df, x='method', y='confidence', ax=axes[0, 0])
        axes[0, 0].set_title('Confidence Score by Method', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('Detection Method', fontsize=10)
        axes[0, 0].set_ylabel('Confidence Score', fontsize=10)
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45)
        
        # Processing time comparison
        sns.boxplot(data=df, x='method', y='processing_time', ax=axes[0, 1])
        axes[0, 1].set_title('Processing Time by Method', fontsize=12, weight='bold')
        axes[0, 1].set_xlabel('Detection Method', fontsize=10)
        axes[0, 1].set_ylabel('Processing Time (seconds)', fontsize=10)
        plt.setp(axes[0, 1].get_xticklabels(), rotation=45)
        
        # Success rate
        success_rate = df.groupby('method').size() / len(df) * 100
        axes[1, 0].bar(success_rate.index, success_rate.values, color='lightgreen')
        axes[1, 0].set_title('Detection Success Rate by Method', fontsize=12, weight='bold')
        axes[1, 0].set_xlabel('Detection Method', fontsize=10)
        axes[1, 0].set_ylabel('Success Rate (%)', fontsize=10)
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
        
        # Average confidence
        avg_confidence = df.groupby('method')['confidence'].mean()
        axes[1, 1].bar(avg_confidence.index, avg_confidence.values, color='lightblue')
        axes[1, 1].set_title('Average Confidence by Method', fontsize=12, weight='bold')
        axes[1, 1].set_xlabel('Detection Method', fontsize=10)
        axes[1, 1].set_ylabel('Average Confidence', fontsize=10)
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45)
        
        plt.suptitle('Detection Method Comparison', fontsize=16, weight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, results_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            results_data: List of result dictionaries
            
        Returns:
            Plotly figure object
        """
        df = pd.DataFrame(results_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Distribution', 'Processing Time Distribution',
                          'Method Comparison', 'Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Confidence histogram
        fig.add_trace(
            go.Histogram(x=df['confidence'], name='Confidence', nbinsx=20),
            row=1, col=1
        )
        
        # Processing time histogram
        fig.add_trace(
            go.Histogram(x=df['processing_time'], name='Processing Time', nbinsx=20),
            row=1, col=2
        )
        
        # Method comparison
        method_stats = df.groupby('method').agg({
            'confidence': 'mean',
            'processing_time': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(x=method_stats['method'], y=method_stats['confidence'], 
                  name='Avg Confidence'),
            row=2, col=1
        )
        
        # Timeline (if timestamp available)
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            timeline_data = df.groupby([df['datetime'].dt.date, 'method']).size().reset_index()
            timeline_data.columns = ['date', 'method', 'count']
            
            for method in timeline_data['method'].unique():
                method_data = timeline_data[timeline_data['method'] == method]
                fig.add_trace(
                    go.Scatter(x=method_data['date'], y=method_data['count'], 
                             mode='lines+markers', name=f'{method} Timeline'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="License Plate Detection Analytics Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def save_visualization(self, fig: plt.Figure, filename: str, 
                          output_dir: str = "output/visualizations") -> None:
        """
        Save visualization to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Output filename
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {file_path}")


class PerformanceMonitor:
    """Performance monitoring and visualization utilities."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_history = []
    
    def log_metric(self, metric_name: str, value: float, 
                   timestamp: Optional[float] = None) -> None:
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history.append({
            'timestamp': timestamp,
            'metric': metric_name,
            'value': value
        })
    
    def create_performance_dashboard(self) -> go.Figure:
        """
        Create a performance monitoring dashboard.
        
        Returns:
            Plotly figure object
        """
        if not self.metrics_history:
            logger.warning("No metrics data available")
            return None
        
        df = pd.DataFrame(self.metrics_history)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Group metrics by type
        metric_types = df['metric'].unique()
        
        fig = make_subplots(
            rows=len(metric_types), cols=1,
            subplot_titles=[f'{metric} Over Time' for metric in metric_types],
            vertical_spacing=0.05
        )
        
        for i, metric in enumerate(metric_types, 1):
            metric_data = df[df['metric'] == metric]
            fig.add_trace(
                go.Scatter(x=metric_data['datetime'], y=metric_data['value'],
                          mode='lines+markers', name=metric),
                row=i, col=1
            )
        
        fig.update_layout(
            title_text="Performance Monitoring Dashboard",
            height=200 * len(metric_types),
            showlegend=True
        )
        
        return fig


def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    visualizer = LicensePlateVisualizer()
    
    # Sample data
    confidences = np.random.beta(2, 1, 100)  # Skewed towards higher values
    processing_times = np.random.exponential(1.5, 100)
    methods = np.random.choice(['Traditional CV', 'Advanced (YOLO + OCR)'], 100)
    
    results_data = [
        {
            'method': method,
            'confidence': conf,
            'processing_time': time,
            'timestamp': time.time() - np.random.uniform(0, 3600)
        }
        for method, conf, time in zip(methods, confidences, processing_times)
    ]
    
    # Create visualizations
    fig1 = visualizer.create_confidence_histogram(confidences)
    fig2 = visualizer.create_processing_time_chart(processing_times, methods)
    fig3 = visualizer.create_method_comparison(results_data)
    
    # Save visualizations
    visualizer.save_visualization(fig1, "confidence_distribution.png")
    visualizer.save_visualization(fig2, "processing_time_analysis.png")
    visualizer.save_visualization(fig3, "method_comparison.png")
    
    # Create interactive dashboard
    interactive_fig = visualizer.create_interactive_dashboard(results_data)
    interactive_fig.write_html("output/visualizations/interactive_dashboard.html")
    
    logger.info("Sample visualizations created successfully")


def main():
    """Main function for testing visualization utilities."""
    import time
    
    # Create sample visualizations
    create_sample_visualizations()
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    # Simulate some metrics
    for i in range(50):
        monitor.log_metric('confidence', np.random.beta(2, 1))
        monitor.log_metric('processing_time', np.random.exponential(1.5))
        time.sleep(0.1)
    
    # Create performance dashboard
    perf_fig = monitor.create_performance_dashboard()
    if perf_fig:
        perf_fig.write_html("output/visualizations/performance_dashboard.html")
        logger.info("Performance dashboard created")


if __name__ == "__main__":
    main()
