import yaml
import numpy as np
import os
import math
import torch

def create_polygon_config(n_agents, radius=15.0):
    """
    Create a configuration with agents arranged in an n-sided polygon
    with goals at diagonally opposite points.
    
    Args:
        n_agents: Number of agents (vertices of the polygon)
        radius: Radius of the polygon
    
    Returns:
        config: Configuration dictionary
    """
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "agents": {},
        "device": device,
        "dt": 0.1,
        "control": {
            "u_max": [6, 6],
            "u_min": [-6, -6],
        },
        "objective": {
            "max_speed": 0.3,
        },
        "simulator": {
            "mode": "thrust",
            "render": True,
            "steps": 700,
            "urdf": "quarter_roboat.urdf"
        }
    }
    
    # Calculate positions around the polygon
    angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
    
    for i in range(n_agents):
        # Current agent position
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        
        # Goal position (diagonally opposite)
        opposite_idx = (i + n_agents // 2) % n_agents
        goal_x = radius * np.cos(angles[opposite_idx])
        goal_y = radius * np.sin(angles[opposite_idx])
        
        # Calculate heading (pointing toward center initially)
        heading = np.arctan2(-y, -x)
        
        # Add agent to config
        config["agents"][f"agent{i}"] = {
            "initial_pose": [float(x), float(y), float(heading)],
            "initial_goal": [float(goal_x), float(goal_y), 0.0],
            "urdf": "aritra.urdf",
            "type": "agent"  # All agents are the same type
        }
    
    return config

if __name__ == "__main__":
    # Create configuration for 8 agents in an octagon
    n_agents = 8
    config = create_polygon_config(n_agents)
    
    # Save configuration to file
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = f"{abs_path}/cfg_polygon.yml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created configuration for {n_agents} agents in a polygon.")
    print(f"Configuration saved to {config_path}")
    print(f"Using device: {config['device']}") 