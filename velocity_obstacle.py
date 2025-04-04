import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import time

class VelocityObstaclePlanner:
    def __init__(self, cfg, agent_id, dynamics):
        """
        Initialize the Velocity Obstacle Planner
        
        Args:
            cfg: Configuration dictionary
            agent_id: ID of the agent this planner controls
            dynamics: Dynamics model of the agent
        """
        self.cfg = cfg
        self.agent_id = agent_id
        self.dynamics = dynamics
        self.device = cfg["device"]
        self.dt = cfg["dt"]
        
        # Get agent info
        self.agent_info = cfg["agents"][f"agent{agent_id}"]
        self.goal = torch.tensor(self.agent_info["initial_goal"][:2], device=self.device, dtype=torch.float32)
        
        # Parameters
        self.safety_radius = 1.0  # Safety radius around each agent
        self.time_horizon = 5.0   # Time horizon for velocity obstacles
        self.max_speed = 0.5      # Increased from default for faster movement
        self.num_samples = 150    # Increased number of velocity samples
        
        # Control limits
        self.u_min = torch.tensor(cfg["control"]["u_min"], device=self.device, dtype=torch.float32)
        self.u_max = torch.tensor(cfg["control"]["u_max"], device=self.device, dtype=torch.float32)
        
        # Store planned trajectory
        self.planned_traj = None
        self.current_command = torch.zeros(2, device=self.device, dtype=torch.float32)
        self.other_agents = {}
        
        # Goal-reaching parameters
        self.goal_threshold = 0.5  # Distance threshold to consider goal reached
        self.goal_weight = 2.0     # Increased weight for goal-directed behavior
        self.avoid_weight = 0.5    # Reduced weight for collision avoidance
        
        # Debug info
        self.debug_info = {}
        
    def make_plan(self, observation):
        """
        Create a plan based on current observation
        
        Args:
            observation: Dictionary containing agent states
        """
        # Extract own state
        self.state = observation[f"agent{self.agent_id}"]["state"]
        self.position = self.state[:2]
        self.heading = self.state[2]
        self.velocity = self.state[3:5]
        
        # Extract other agents' states
        self.other_agents = {}
        for agent_key, agent_data in observation.items():
            if agent_key != f"agent{self.agent_id}":
                agent_id = int(agent_key.replace("agent", ""))
                self.other_agents[agent_id] = {
                    "state": agent_data["state"]
                }
        
        # Check if we're at the goal
        direction_to_goal = self.goal - self.position
        distance_to_goal = torch.norm(direction_to_goal)
        
        if distance_to_goal < self.goal_threshold:
            # At goal, stop
            self.current_command = torch.zeros(2, device=self.device, dtype=torch.float32)
            
            # Generate a simple trajectory at the goal
            traj_length = 20
            traj = torch.zeros((traj_length, 2), device=self.device, dtype=torch.float32)
            for i in range(traj_length):
                traj[i] = self.position
            
            self.planned_traj = traj
            return
        
        # Plan with velocity obstacles to reach goal while avoiding collisions
        self._plan_with_velocity_obstacles(self.goal)
    
    def _plan_with_velocity_obstacles(self, goal):
        """
        Plan using velocity obstacles to avoid collisions
        
        Args:
            goal: Goal position tensor
        """
        # Direction and distance to goal
        direction_to_goal = goal - self.position
        distance_to_goal = torch.norm(direction_to_goal)
        
        # Normalize direction
        if distance_to_goal > 0:
            direction_to_goal = direction_to_goal / distance_to_goal
        
        # Preferred velocity (directly toward goal)
        preferred_speed = min(self.max_speed, distance_to_goal / 2.0)
        preferred_vel = direction_to_goal * preferred_speed
        
        # Generate velocity samples
        velocities = self._generate_velocity_samples(preferred_vel, preferred_speed)
        
        # Evaluate each velocity
        best_vel = None
        best_score = float('-inf')
        
        for vel in velocities:
            # Check for collisions with other agents
            collision = False
            for agent_id, agent_data in self.other_agents.items():
                other_pos = agent_data["state"][:2]
                other_vel = agent_data["state"][3:5]
                
                # Ensure same data type
                vel_float = vel.to(dtype=torch.float32)
                other_pos_float = other_pos.to(dtype=torch.float32)
                other_vel_float = other_vel.to(dtype=torch.float32)
                
                if self._check_collision(vel_float, other_pos_float, other_vel_float):
                    collision = True
                    break
            
            # Skip this velocity if it leads to collision
            if collision:
                continue
            
            # Score this velocity
            # Higher score for velocities closer to preferred velocity
            vel_diff = vel - preferred_vel
            goal_score = -torch.norm(vel_diff) * self.goal_weight
            
            # Higher score for higher speeds when far from goal
            speed_score = torch.norm(vel) * (distance_to_goal / 10.0)
            
            # Total score
            score = goal_score + speed_score
            
            if score > best_score:
                best_score = score
                best_vel = vel
        
        # If no valid velocity found, use emergency strategy
        if best_vel is None:
            print(f"Agent {self.agent_id}: No valid velocity found, using emergency strategy")
            
            # Try to move slowly toward goal
            best_vel = direction_to_goal * 0.1
            
            # If still in collision, try stopping
            for agent_id, agent_data in self.other_agents.items():
                other_pos = agent_data["state"][:2].to(dtype=torch.float32)
                other_vel = agent_data["state"][3:5].to(dtype=torch.float32)
                
                if self._check_collision(best_vel.to(dtype=torch.float32), other_pos, other_vel):
                    best_vel = torch.zeros(2, device=self.device, dtype=torch.float32)
                    break
        
        # Store for debugging
        self.debug_info["best_vel"] = best_vel
        self.debug_info["preferred_vel"] = preferred_vel
        
        # Generate trajectory with this velocity
        traj_length = 20
        traj = torch.zeros((traj_length, 2), device=self.device, dtype=torch.float32)
        
        for i in range(traj_length):
            t = i * self.dt
            traj[i] = self.position + best_vel * t
        
        self.planned_traj = traj
        
        # Convert velocity to control inputs
        self._convert_velocity_to_control(best_vel)
    
    def _generate_velocity_samples(self, preferred_vel, preferred_speed):
        """
        Generate velocity samples around the preferred velocity
        
        Args:
            preferred_vel: Preferred velocity vector
            preferred_speed: Preferred speed
            
        Returns:
            List of velocity samples
        """
        velocities = []
        
        # Add preferred velocity
        velocities.append(preferred_vel)
        
        # Add zero velocity
        velocities.append(torch.zeros(2, device=self.device, dtype=torch.float32))
        
        # Add velocities in a circle around preferred velocity
        num_angles = 16
        for i in range(num_angles):
            angle = 2 * np.pi * i / num_angles
            direction = torch.tensor([np.cos(angle), np.sin(angle)], device=self.device, dtype=torch.float32)
            
            # Add velocities at different speeds
            for speed_factor in [0.25, 0.5, 0.75, 1.0]:
                speed = preferred_speed * speed_factor
                velocities.append(direction * speed)
        
        # Add more samples in the direction of the goal
        goal_dir = preferred_vel / (torch.norm(preferred_vel) + 1e-6)
        for speed_factor in [0.3, 0.6, 0.9, 1.2]:
            speed = preferred_speed * speed_factor
            velocities.append(goal_dir * speed)
        
        # Add current velocity
        velocities.append(self.velocity)
        
        return velocities
    
    def _convert_velocity_to_control(self, desired_vel):
        """
        Convert desired velocity to control inputs
        
        Args:
            desired_vel: Desired velocity in world frame
        """
        # Convert to body frame
        heading_tensor = torch.tensor([self.heading], device=self.device, dtype=torch.float32)
        cos_h = torch.cos(heading_tensor)
        sin_h = torch.sin(heading_tensor)
        
        # Rotation matrix from world to body
        R = torch.tensor([
            [cos_h, sin_h],
            [-sin_h, cos_h]
        ], device=self.device, dtype=torch.float32).squeeze()
        
        vel_body = torch.matmul(R, desired_vel)
        
        # Stronger control for more responsive behavior
        u1 = vel_body[0] * 10.0  # Forward thrust - increased gain
        u2 = vel_body[1] * 15.0  # Turning - increased gain
        
        # Clamp control inputs
        u1 = torch.clamp(u1, self.u_min[0], self.u_max[0])
        u2 = torch.clamp(u2, self.u_min[1], self.u_max[1])
        
        self.current_command = torch.tensor([u1, u2], device=self.device, dtype=torch.float32)
        
        # Debug info
        print(f"Agent {self.agent_id}: pos={self.position.cpu().numpy()}, " +
              f"heading={self.heading.cpu().item():.2f}, " +
              f"desired_vel={desired_vel.cpu().numpy()}, " +
              f"vel_body={vel_body.cpu().numpy()}, " +
              f"control=[{u1.cpu().item():.2f}, {u2.cpu().item():.2f}]")
    
    def _check_collision(self, vel, other_pos, other_vel):
        """
        Check if velocity leads to collision with another agent
        
        Args:
            vel: Velocity to check
            other_pos: Position of other agent
            other_vel: Velocity of other agent
            
        Returns:
            True if collision, False otherwise
        """
        # Ensure all tensors have the same dtype
        vel = vel.to(dtype=torch.float32)
        other_pos = other_pos.to(dtype=torch.float32)
        other_vel = other_vel.to(dtype=torch.float32)
        position = self.position.to(dtype=torch.float32)
        
        # Relative position and velocity
        rel_pos = other_pos - position
        rel_vel = other_vel - vel
        
        # Distance between agents
        dist = torch.norm(rel_pos)
        
        # Combined radius (safety margin)
        combined_radius = 2 * self.safety_radius
        
        # If agents are already too close
        if dist < combined_radius:
            return True
        
        # If relative velocity is very small, no collision
        if torch.norm(rel_vel) < 1e-6:
            return False
            
        # Project relative position onto relative velocity
        t_closest = -torch.dot(rel_pos, rel_vel) / torch.dot(rel_vel, rel_vel)
        
        # If closest approach is in the past, no collision
        if t_closest < 0:
            return False
            
        # If closest approach is too far in the future, ignore
        if t_closest > self.time_horizon:
            return False
            
        # Position at closest approach
        closest_pos = rel_pos + t_closest * rel_vel
        closest_dist = torch.norm(closest_pos)
        
        # Check if closest approach is within combined radius
        return closest_dist < combined_radius
    
    def get_command(self):
        """Get the current control command"""
        return self.current_command
    
    def get_planned_traj(self):
        """Get the planned trajectory"""
        return self.planned_traj
    
    def visualize_velocity_obstacles(self, ax):
        """
        Visualize velocity obstacles for debugging
        
        Args:
            ax: Matplotlib axis
        """
        # Convert tensors to CPU for visualization
        position_cpu = self.position.cpu().detach()
        velocity_cpu = self.velocity.cpu().detach()
        
        # Draw agent
        agent_circle = Circle((position_cpu[0].item(), position_cpu[1].item()), 
                             self.safety_radius, color='blue', alpha=0.3)
        ax.add_patch(agent_circle)
        
        # Draw velocity vector
        ax.arrow(position_cpu[0].item(), position_cpu[1].item(), 
                velocity_cpu[0].item(), velocity_cpu[1].item(), 
                head_width=0.3, head_length=0.5, fc='blue', ec='blue')
        
        # Draw other agents and velocity obstacles
        for agent_id, agent_data in self.other_agents.items():
            other_pos = agent_data["state"][:2].cpu().detach()
            other_vel = agent_data["state"][3:5].cpu().detach()
            
            # Skip if too far away
            dist_to_other = np.linalg.norm(other_pos.numpy() - position_cpu.numpy())
            if dist_to_other > 15.0:
                continue
                
            # Draw other agent
            other_circle = Circle((other_pos[0].item(), other_pos[1].item()), 
                                 self.safety_radius, color='blue', alpha=0.3)
            ax.add_patch(other_circle)
            
            # Draw velocity vector
            ax.arrow(other_pos[0].item(), other_pos[1].item(), 
                    other_vel[0].item(), other_vel[1].item(), 
                    head_width=0.3, head_length=0.5, fc='blue', ec='blue')
            
            # Draw velocity obstacle (simplified as a cone)
            # This is a simplified visualization
            rel_pos_np = (other_pos - position_cpu).numpy()
            dist = np.linalg.norm(rel_pos_np)
            
            # Combined radius
            combined_radius = 2 * self.safety_radius
            
            # Angle to tangent points
            if dist > combined_radius:
                # Angle to center
                angle_to_center = np.arctan2(rel_pos_np[1], rel_pos_np[0])
                
                # Half angle of the cone
                half_angle = np.arcsin(combined_radius / dist)
                
                # Draw the cone
                wedge = Wedge((position_cpu[0].item(), position_cpu[1].item()), 
                              15.0,  # Radius of wedge
                              np.degrees(angle_to_center - half_angle),
                              np.degrees(angle_to_center + half_angle),
                              alpha=0.2, color='blue')
                ax.add_patch(wedge)
