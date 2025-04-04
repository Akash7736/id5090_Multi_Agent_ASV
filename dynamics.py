import torch

class UnderactuatedQuarterRoboatDynamics:
    def __init__(self, cfg) -> None:
        self.aa = 0.45  # Distance between thrusters (half-width)
        self.bb = 0.90  # Not used in differential thrust

        # Inertia parameters
        self.m11 = 12    # Surge mass
        self.m22 = 24    # Sway mass
        self.m33 = 1.5   # Yaw moment of inertia
        
        # Damping parameters
        self.d11 = 6     # Surge damping
        self.d22 = 8     # Sway damping
        self.d33 = 1.35  # Yaw damping

        self.cfg = cfg
        self.dt = cfg["dt"]
        self.device = cfg["device"]

        # Dynamics matrices
        self.D = torch.tensor([
            [self.d11, 0, 0],
            [0, self.d22, 0],
            [0, 0, self.d33]
        ], device=self.device)

        self.M = torch.tensor([
            [self.m11, 0, 0],
            [0, self.m22, 0],
            [0, 0, self.m33]
        ], device=self.device)
        
        # Modified B matrix for differential thrust (2 inputs)
        # [left_thrust, right_thrust]
        self.B = torch.tensor([
            [1, 1],           # Both thrusters contribute to surge
            [0, 0],           # No direct sway control
            [self.aa/2, -self.aa/2]  # Differential thrust for yaw
        ], device=self.device)

        # Inverse of inertia matrix
        self.Minv = torch.inverse(self.M)

    def rot_matrix(self, heading):
        cos = torch.cos(heading).to(self.device)
        sin = torch.sin(heading).to(self.device)
        self.zeros = torch.zeros_like(heading, device=self.device)
        ones = torch.ones_like(heading, device=self.device)

        stacked = torch.stack(
            [cos, -sin, self.zeros, sin, cos, self.zeros, self.zeros, self.zeros, ones],
            dim=1
        ).reshape(heading.size(0), 3, 3, heading.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)

    def coriolis(self, vel):
        stacked = torch.stack([
            self.zeros, self.zeros, -self.m22 * vel[:, :, 1],
            self.zeros, self.zeros, self.m11 * vel[:, :, 0],
            self.m22 * vel[:, :, 1], -self.m11 * vel[:, :, 0], self.zeros
        ], dim=1).reshape(vel.size(0), 3, 3, vel.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)
        
    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int = -1) -> torch.Tensor:
        # Ensure all tensors are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Other tensor initializations...
        pose_enu = states[:, :, 0:3]
        pose = torch.zeros_like(pose_enu).to(self.device)
        pose[:, :, 0] = pose_enu[:, :, 1]
        pose[:, :, 1] = pose_enu[:, :, 0]
        pose[:, :, 2] = torch.pi / 2 - pose_enu[:, :, 2]

        vel_enu = states[:, :, 3:6]
        vel = torch.zeros_like(vel_enu).to(self.device)
        vel[:, :, 0] = vel_enu[:, :, 1]
        vel[:, :, 1] = vel_enu[:, :, 0]
        vel[:, :, 2] = -vel_enu[:, :, 2]

        self.zeros = torch.zeros(vel.size(0), vel.size(1), device=self.device)

        # Rotate velocity to the body frame
        vel_body = torch.bmm(
            self.rot_matrix(-pose[:, :, 2]).reshape(-1, 3, 3),
            vel.reshape(-1, 3).unsqueeze(2),
        ).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Ensure other tensors are on the correct device
        Minv_batch = self.Minv.repeat(vel.size(0) * vel.size(1), 1, 1).to(self.device)
        B_batch = self.B.repeat(vel.size(0) * vel.size(1), 1, 1).to(self.device)
        D_batch = self.D.repeat(vel.size(0) * vel.size(1), 1, 1).to(self.device)
        C_batch = self.coriolis(vel_body).reshape(-1, 3, 3).to(self.device)

        # Key change: reshape actions to match the 2-input B matrix
        new_vel_body = torch.bmm(
            Minv_batch,
            (
                torch.bmm(B_batch, actions.reshape(-1, 2).unsqueeze(2))
                - torch.bmm(C_batch, vel_body.reshape(-1, 3).unsqueeze(2))
                - torch.bmm(D_batch, vel_body.reshape(-1, 3).unsqueeze(2))
            ),
        ).reshape(vel.size(0), vel.size(1), vel.size(2)) * self.dt + vel_body

        # Rotate velocity to the world frame
        vel = torch.bmm(
            self.rot_matrix(pose[:, :, 2]).reshape(-1, 3, 3),
            new_vel_body.reshape(-1, 3).unsqueeze(2),
        ).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Compute new pose
        pose += self.dt * vel

        # Convert from NED to ENU
        new_pose = torch.zeros_like(pose).to(self.device)
        new_pose[:, :, 0] = pose[:, :, 1]
        new_pose[:, :, 1] = pose[:, :, 0]
        new_pose[:, :, 2] = torch.pi / 2 - pose[:, :, 2]
        new_vel = torch.zeros_like(vel).to(self.device)
        new_vel[:, :, 0] = vel[:, :, 1]
        new_vel[:, :, 1] = vel[:, :, 0]
        new_vel[:, :, 2] = -vel[:, :, 2]

        # Set new state
        new_states = torch.concatenate((new_pose, new_vel), 2)

        return new_states, actions