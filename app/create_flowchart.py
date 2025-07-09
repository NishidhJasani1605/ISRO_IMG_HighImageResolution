import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import FancyArrowPatch, Rectangle

def create_flowchart():
    """Create a flowchart image for the ISRO SR project"""
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define box properties
    box_width = 1.5
    box_height = 0.6
    box_spacing = 1.2
    
    # Define box positions
    positions = {
        'input1': (1, 5),
        'input2': (1, 3),
        'align': (3, 4),
        'preprocess': (5, 4),
        'encoder1': (7, 5),
        'encoder2': (7, 3),
        'fusion': (9, 4),
        'decoder': (11, 4),
        'sr': (13, 4),
        'eval': (15, 4),
    }
    
    # Define box labels
    labels = {
        'input1': 'LR Image 1',
        'input2': 'LR Image 2',
        'align': 'Image\nAlignment',
        'preprocess': 'Preprocessing',
        'encoder1': 'Encoder 1',
        'encoder2': 'Encoder 2',
        'fusion': 'Feature\nFusion',
        'decoder': 'Decoder',
        'sr': 'SR Image',
        'eval': 'Quality\nEvaluation',
    }
    
    # Define box colors
    colors = {
        'input1': '#3498db',
        'input2': '#3498db',
        'align': '#9b59b6',
        'preprocess': '#9b59b6',
        'encoder1': '#e74c3c',
        'encoder2': '#e74c3c',
        'fusion': '#e74c3c',
        'decoder': '#e74c3c',
        'sr': '#2ecc71',
        'eval': '#f39c12',
    }
    
    # Draw boxes
    boxes = {}
    for key, pos in positions.items():
        x, y = pos
        box = Rectangle((x - box_width / 2, y - box_height / 2), 
                       box_width, box_height, 
                       facecolor=colors[key],
                       edgecolor='black',
                       alpha=0.8,
                       linewidth=2,
                       zorder=1)
        ax.add_patch(box)
        boxes[key] = box
        ax.text(x, y, labels[key], ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white', zorder=2)
    
    # Define arrows
    arrows = [
        ('input1', 'align'),
        ('input2', 'align'),
        ('align', 'preprocess'),
        ('preprocess', 'encoder1'),
        ('preprocess', 'encoder2'),
        ('encoder1', 'fusion'),
        ('encoder2', 'fusion'),
        ('fusion', 'decoder'),
        ('decoder', 'sr'),
        ('sr', 'eval'),
    ]
    
    # Draw arrows
    for start, end in arrows:
        start_pos = positions[start]
        end_pos = positions[end]
        
        if start == 'preprocess' and end == 'encoder1':
            # Curve up
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                connectionstyle="arc3,rad=0.3",
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        elif start == 'preprocess' and end == 'encoder2':
            # Curve down
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                connectionstyle="arc3,rad=-0.3",
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        elif start == 'input1' and end == 'align':
            # Curve from input1 to align
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                connectionstyle="arc3,rad=-0.3",
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        elif start == 'input2' and end == 'align':
            # Curve from input2 to align
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                connectionstyle="arc3,rad=0.3",
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        elif start == 'encoder1' and end == 'fusion':
            # Curve from encoder1 to fusion
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                connectionstyle="arc3,rad=-0.3",
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        elif start == 'encoder2' and end == 'fusion':
            # Curve from encoder2 to fusion
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                connectionstyle="arc3,rad=0.3",
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        else:
            # Straight arrow
            arrow = FancyArrowPatch(
                (start_pos[0] + box_width / 2, start_pos[1]),
                (end_pos[0] - box_width / 2, end_pos[1]),
                arrowstyle='-|>',
                color='black',
                linewidth=1.5,
                zorder=0
            )
        
        ax.add_patch(arrow)
    
    # Add title
    ax.text(8, 6.5, 'Dual Image Super-Resolution Pipeline', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Add credits
    ax.text(8, 1.5, 'ISRO Satellite Image Super-Resolution Project', 
            ha='center', fontsize=10, color='gray')
    
    # Set axis limits
    ax.set_xlim(0, 16)
    ax.set_ylim(1, 7)
    
    # Remove axes
    ax.axis('off')
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), 'flowchart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    
    print(f"Flowchart saved to {output_path}")

if __name__ == "__main__":
    create_flowchart() 