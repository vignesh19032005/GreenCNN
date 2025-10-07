"""
Visualization tools for carbon tracking and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class GreenCNNVisualizer:
    """Visualization tools for GreenCNN training results"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_training_metrics(self, training_history: List[Dict], 
                            save_path: Optional[str] = None):
        """Plot training and validation metrics"""
        
        epochs = [h['epoch'] for h in training_history]
        train_loss = [h['loss'] for h in training_history]
        train_acc = [h['accuracy'] for h in training_history]
        val_loss = [h['val_loss'] for h in training_history]
        val_acc = [h['val_accuracy'] for h in training_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training/Validation Loss
        ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training/Validation Accuracy
        ax2.plot(epochs, train_acc, label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Carbon Efficiency
        carbon_efficiency = [h['carbon_data']['carbon_efficiency'] 
                           for h in training_history 
                           if h.get('carbon_data')]
        if carbon_efficiency:
            ax3.plot(epochs[:len(carbon_efficiency)], carbon_efficiency, 
                    color='green', linewidth=2, marker='o', markersize=4)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Carbon Efficiency (acc/g CO2)')
            ax3.set_title('Carbon Efficiency Over Time')
            ax3.grid(True, alpha=0.3)
        
        # Learning Rate
        lr_values = [h['optimizer_stats']['current_lr'] 
                    for h in training_history 
                    if h.get('optimizer_stats')]
        if lr_values:
            ax4.semilogy(epochs[:len(lr_values)], lr_values, 
                        color='orange', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate (log scale)')
            ax4.set_title('Learning Rate Schedule')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_carbon_tracking(self, carbon_data_path: str, 
                           save_path: Optional[str] = None):
        """Plot detailed carbon tracking data"""
        
        with open(carbon_data_path, 'r') as f:
            data = json.load(f)
        
        energy_data = data['energy_data']
        
        if not energy_data:
            print("No energy data available for plotting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(energy_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['cumulative_emissions'] = df['carbon_emissions_kg'].cumsum() * 1000  # Convert to grams
        df['cumulative_energy'] = df['energy_kwh'].cumsum()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Power consumption over time
        ax1.plot(df['timestamp'], df['estimated_power_watts'], 
                color='red', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Power (Watts)')
        ax1.set_title('Power Consumption Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative carbon emissions
        ax2.plot(df['timestamp'], df['cumulative_emissions'], 
                color='green', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative CO2 Emissions (g)')
        ax2.set_title('Cumulative Carbon Emissions')
        ax2.grid(True, alpha=0.3)
        
        # CPU and Memory usage
        ax3.plot(df['timestamp'], df['cpu_percent'], 
                label='CPU %', linewidth=1, alpha=0.8)
        ax3.plot(df['timestamp'], df['memory_percent'], 
                label='Memory %', linewidth=1, alpha=0.8)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Usage (%)')
        ax3.set_title('System Resource Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Energy consumption rate
        ax4.plot(df['timestamp'], df['energy_kwh'] * 1000, 
                color='orange', linewidth=1, alpha=0.7)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Energy Rate (Wh)')
        ax4.set_title('Energy Consumption Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, training_history: List[Dict], 
                                   carbon_data_path: str) -> go.Figure:
        """Create interactive Plotly dashboard"""
        
        # Load carbon data
        with open(carbon_data_path, 'r') as f:
            carbon_data = json.load(f)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Metrics', 'Carbon Efficiency', 
                          'Power Consumption', 'System Resources',
                          'Cumulative Emissions', 'Learning Rate'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = [h['epoch'] for h in training_history]
        
        # Training metrics
        fig.add_trace(
            go.Scatter(x=epochs, y=[h['accuracy'] for h in training_history],
                      name='Train Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=[h['val_accuracy'] for h in training_history],
                      name='Val Accuracy', line=dict(color='red')),
            row=1, col=1
        )
        
        # Carbon efficiency
        carbon_efficiency = [h['carbon_data']['carbon_efficiency'] 
                           for h in training_history 
                           if h.get('carbon_data')]
        if carbon_efficiency:
            fig.add_trace(
                go.Scatter(x=epochs[:len(carbon_efficiency)], y=carbon_efficiency,
                          name='Carbon Efficiency', line=dict(color='green')),
                row=1, col=2
            )
        
        # Power consumption (if available)
        if carbon_data.get('energy_data'):
            energy_df = pd.DataFrame(carbon_data['energy_data'])
            timestamps = pd.to_datetime(energy_df['timestamp'], unit='s')
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=energy_df['estimated_power_watts'],
                          name='Power (W)', line=dict(color='orange')),
                row=2, col=1
            )
            
            # System resources
            fig.add_trace(
                go.Scatter(x=timestamps, y=energy_df['cpu_percent'],
                          name='CPU %', line=dict(color='purple')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=timestamps, y=energy_df['memory_percent'],
                          name='Memory %', line=dict(color='brown')),
                row=2, col=2
            )
            
            # Cumulative emissions
            cumulative_emissions = energy_df['carbon_emissions_kg'].cumsum() * 1000
            fig.add_trace(
                go.Scatter(x=timestamps, y=cumulative_emissions,
                          name='Cumulative CO2 (g)', line=dict(color='darkgreen')),
                row=3, col=1
            )
        
        # Learning rate
        lr_values = [h['optimizer_stats']['current_lr'] 
                    for h in training_history 
                    if h.get('optimizer_stats')]
        if lr_values:
            fig.add_trace(
                go.Scatter(x=epochs[:len(lr_values)], y=lr_values,
                          name='Learning Rate', line=dict(color='magenta')),
                row=3, col=2
            )
        
        fig.update_layout(
            title="GreenCNN Training Dashboard",
            height=900,
            showlegend=True
        )
        
        return fig
    
    def plot_efficiency_comparison(self, results: List[Dict], 
                                 labels: List[str],
                                 save_path: Optional[str] = None):
        """Compare carbon efficiency across different runs"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Carbon efficiency comparison
        efficiencies = [r['avg_carbon_efficiency'] for r in results]
        accuracies = [r['best_val_accuracy'] for r in results]
        emissions = [r['carbon_summary']['total_emissions_kg'] * 1000 for r in results]
        
        bars1 = ax1.bar(labels, efficiencies, color='green', alpha=0.7)
        ax1.set_ylabel('Carbon Efficiency (acc/g CO2)')
        ax1.set_title('Carbon Efficiency Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, eff in zip(bars1, efficiencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{eff:.2f}', ha='center', va='bottom')
        
        # Accuracy vs Emissions scatter
        scatter = ax2.scatter(emissions, accuracies, 
                            c=efficiencies, cmap='RdYlGn', 
                            s=100, alpha=0.7)
        
        for i, label in enumerate(labels):
            ax2.annotate(label, (emissions[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Total CO2 Emissions (g)')
        ax2.set_ylabel('Best Validation Accuracy')
        ax2.set_title('Accuracy vs Carbon Emissions')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Carbon Efficiency (acc/g CO2)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, training_summary: Dict, 
                       output_path: str = "green_training_report.html"):
        """Generate comprehensive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GreenCNN Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }}
                .green {{ color: #4CAF50; font-weight: bold; }}
                .red {{ color: #f44336; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌱 GreenCNN Training Report</h1>
                <p>Carbon-Efficient Deep Learning Results</p>
            </div>
            
            <div class="section">
                <h2>Training Summary</h2>
                <div class="metric">
                    <strong>Total Epochs:</strong> {training_summary['total_epochs']}
                </div>
                <div class="metric">
                    <strong>Best Validation Accuracy:</strong> <span class="green">{training_summary['best_val_accuracy']:.4f}</span>
                </div>
                <div class="metric">
                    <strong>Final Validation Accuracy:</strong> {training_summary['final_val_accuracy']:.4f}
                </div>
            </div>
            
            <div class="section">
                <h2>🌱 Carbon Footprint</h2>
                <div class="metric">
                    <strong>Total CO2 Emissions:</strong> <span class="red">{training_summary['carbon_summary']['total_emissions_kg']*1000:.2f}g</span>
                </div>
                <div class="metric">
                    <strong>Total Energy:</strong> {training_summary['carbon_summary']['total_energy_kwh']:.4f} kWh
                </div>
                <div class="metric">
                    <strong>Best Carbon Efficiency:</strong> <span class="green">{training_summary['best_carbon_efficiency']:.2f} acc/g CO2</span>
                </div>
                <div class="metric">
                    <strong>Average Carbon Efficiency:</strong> {training_summary['avg_carbon_efficiency']:.2f} acc/g CO2
                </div>
            </div>
            
            <div class="section">
                <h2>Model Statistics</h2>
                <div class="metric">
                    <strong>Total Parameters:</strong> {training_summary['model_stats']['total_parameters']:,}
                </div>
                <div class="metric">
                    <strong>Model Size:</strong> {training_summary['model_stats']['model_size_mb']:.2f} MB
                </div>
                <div class="metric">
                    <strong>Estimated FLOPs:</strong> {training_summary['model_stats']['estimated_flops']:,}
                </div>
            </div>
            
            <div class="section">
                <h2>Environmental Impact</h2>
                <p>This training session consumed <strong>{training_summary['carbon_summary']['total_energy_kwh']:.4f} kWh</strong> 
                of energy and produced <strong>{training_summary['carbon_summary']['total_emissions_kg']*1000:.2f}g</strong> 
                of CO2 emissions.</p>
                
                <p>The carbon efficiency of <strong>{training_summary['best_carbon_efficiency']:.2f} accuracy points per gram of CO2</strong> 
                demonstrates the effectiveness of green optimization techniques.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Training report saved to {output_path}")


# Example usage function
def visualize_training_results(training_summary: Dict, carbon_data_path: str):
    """Convenience function to generate all visualizations"""
    
    visualizer = GreenCNNVisualizer()
    
    # Plot training metrics
    visualizer.plot_training_metrics(
        training_summary['training_history'],
        save_path='training_metrics.png'
    )
    
    # Plot carbon tracking
    visualizer.plot_carbon_tracking(
        carbon_data_path,
        save_path='carbon_tracking.png'
    )
    
    # Create interactive dashboard
    dashboard = visualizer.create_interactive_dashboard(
        training_summary['training_history'],
        carbon_data_path
    )
    dashboard.write_html('interactive_dashboard.html')
    
    # Generate report
    visualizer.generate_report(training_summary)
    
    print("All visualizations generated successfully!")