"""Low-level cat-style motion via Adversarial Motion Priors (AMP).

The motion module is responsible for the *look* of how the robot moves.
A small set of retargeted cat motion clips trains a style discriminator that
biases the RL policy toward cat-like gait, posture, weight-shifting, and
balance recovery. We are matching the *distribution* of cat motion, not any
specific trajectory.
"""
