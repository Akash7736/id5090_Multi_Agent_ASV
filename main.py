from dynamics import UnderactuatedQuarterRoboatDynamics
import traceback
from tqdm import tqdm
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from velocity_obstacle import VelocityObstaclePlanner

class MultiAgentSimulator:
    def __init__(self, config, dynamics):
        self.config = config
        self.dynamics = dynamics
        self.dt = config["dt"]
        self.device = config["device"]
        
        # Initialize agents
        self.agents = {}
        self.planners = {}
        
        for agent_id, agent_info in config["agents"].items():
            # Extract agent ID number
            agent_num = int(agent_id.replace("agent", ""))
            
            # Initialize agent state
            initial_pose = torch.tensor(agent_info["initial_pose"], device=self.device)
            initial_vel = torch.zeros(3, device=self.device)
            state = torch.cat([initial_pose, initial_vel])
            
            # Store initial position and goal for visualization
            initial_position = initial_pose[:2].cpu().detach().numpy()
            goal_position = torch.tensor(agent_info["initial_goal"][:2], device="cpu").numpy()
            
            self.agents[agent_id] = {
                "state": state,
                "initial_position": initial_position,
                "goal_position": goal_position
            }
            
            # Create planner for this agent
            self.planners[agent_id] = VelocityObstaclePlanner(
                config, agent_num, dynamics
            )
        
        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup the visualization"""
        # Set axis limits
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Create agent markers
        self.agent_markers = {}
        for agent_id in self.agents.keys():
            marker, = self.ax.plot([], [], 'o', color='blue', markersize=10)
            self.agent_markers[agent_id] = marker
            
        # Create trajectory lines
        self.traj_lines = {}
        for agent_id in self.agents.keys():
            line, = self.ax.plot([], [], '-', alpha=0.5)
            self.traj_lines[agent_id] = line
        
        # Plot initial positions and goals
        for agent_id, agent_data in self.agents.items():
            # Plot initial position (green star)
            initial_pos = agent_data["initial_position"]
            self.ax.plot(initial_pos[0], initial_pos[1], '*', color='green', markersize=12)
            
            # Plot goal position (red star)
            goal_pos = agent_data["goal_position"]
            self.ax.plot(goal_pos[0], goal_pos[1], '*', color='red', markersize=12)
            
            # Add agent ID labels
            self.ax.text(initial_pos[0], initial_pos[1] + 1, f"Agent {agent_id}", 
                        fontsize=8, ha='center')
            self.ax.text(goal_pos[0], goal_pos[1] + 1, f"Goal {agent_id}", 
                        fontsize=8, ha='center')
            
            # Draw a line connecting start and goal
            self.ax.plot([initial_pos[0], goal_pos[0]], [initial_pos[1], goal_pos[1]], 
                        '--', color='gray', alpha=0.3)
            
        # Add legend
        self.ax.plot([], [], '*', color='green', markersize=12, label='Start')
        self.ax.plot([], [], '*', color='red', markersize=12, label='Goal')
        self.ax.plot([], [], 'o', color='blue', markersize=10, label='Agent')
        self.ax.legend(loc='upper right')
            
        plt.ion()  # Turn on interactive mode
        
    def step(self, actions=None):
        """
        Step the simulation forward
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observation: Dictionary of agent states
        """
        if actions is None:
            actions = {}
            
        # Process each agent
        for agent_id, agent_data in self.agents.items():
            # Get action for this agent
            if agent_id in actions:
                action = actions[agent_id]
            else:
                # Default action (no thrust)
                action = torch.zeros(2, device=self.device)
                
            # Reshape state and action for dynamics
            state_batch = agent_data["state"].unsqueeze(0).unsqueeze(0)
            action_batch = action.unsqueeze(0).unsqueeze(0)
            
            # Step dynamics
            new_state_batch, _ = self.dynamics.step(state_batch, action_batch)
            
            # Update agent state
            self.agents[agent_id]["state"] = new_state_batch[0, 0]
            
        # Return current observation
        return self.get_observation()
    
    def get_observation(self):
        """Get the current observation of all agents"""
        observation = {}
        for agent_id, agent_data in self.agents.items():
            observation[agent_id] = {
                "state": agent_data["state"].clone()
            }
        return observation
    
    def plot_trajectories(self, trajectories=None):
        """
        Update the visualization
        
        Args:
            trajectories: Dictionary of planned trajectories
        """
        # Update agent positions
        for agent_id, agent_data in self.agents.items():
            state = agent_data["state"]
            marker = self.agent_markers[agent_id]
            
            # Convert tensor to CPU for plotting
            state_cpu = state.cpu().detach()
            marker.set_data(state_cpu[0].item(), state_cpu[1].item())
            
            # Draw heading
            heading = state_cpu[2].item()
            length = 1.0
            dx = length * np.cos(heading)
            dy = length * np.sin(heading)
            self.ax.arrow(state_cpu[0].item(), state_cpu[1].item(), dx, dy, 
                         head_width=0.3, head_length=0.5, fc='black', ec='black')
            
            # Update trajectory if available
            if trajectories is not None and agent_id in trajectories:
                traj = trajectories[agent_id]
                line = self.traj_lines[agent_id]
                
                # Convert trajectory to CPU for plotting
                traj_cpu = traj.cpu().detach()
                line.set_data(traj_cpu[:, 0], traj_cpu[:, 1])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

def run_simulation():
    # Create simulator
    dynamics = UnderactuatedQuarterRoboatDynamics(cfg=CONFIG)
    aritrasim = MultiAgentSimulator(CONFIG, dynamics)
    
    # Get initial observation
    observation = aritrasim.get_observation()
    
    for step in tqdm(range(CONFIG['simulator']['steps'])):
        try:
            # Update planners and get actions
            actions = {}
            plans = {}
            
            for agent_id, planner in aritrasim.planners.items():
                planner.make_plan(observation)
                actions[agent_id] = planner.get_command()
                plans[agent_id] = planner.get_planned_traj()
            
            # Visualize plans
            aritrasim.plot_trajectories(plans)
            
            # Step simulation
            observation = aritrasim.step(actions)
            
        except KeyboardInterrupt:
            print("\nUser interrupted the simulation. Saving data and exiting...")
            break

        except Exception as e:
            print(f"\nError in simulation step {step}: {str(e)}")
            print(traceback.format_exc())

            print("Attempting to continue simulation...")
            continue

if __name__ == "__main__":
    # Load the config file
    abs_path = os.path.dirname(os.path.abspath(__file__))
    
    # Use the polygon configuration
    config_path = f"{abs_path}/cfg_polygon.yml"
    
    # If the polygon config doesn't exist, create it
    if not os.path.exists(config_path):
        from create_polygon_config import create_polygon_config
        config = create_polygon_config(8)  # Create an octagon
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created new polygon configuration at {config_path}")
    else:
        config = yaml.safe_load(open(config_path))
    
    CONFIG = config
    
    # Set device - use CUDA if available
    if "device" not in CONFIG:
        CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {CONFIG['device']}")
    
    run_simulation()