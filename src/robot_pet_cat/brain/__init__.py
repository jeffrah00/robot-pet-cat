"""High-level cat brain — picks which skill to invoke given scene + mood.

This is where cat *personality* lives. The policy is RL-trained with three
intrinsic rewards:

  - curiosity   — visit unfamiliar locations, attend to novel objects
                  (Pathak et al. 2017 'Curiosity-Driven Exploration')
  - comfort     — prefer elevated, warm, near-window, soft spots
  - play        — engagement reward when small movable objects are nearby

A latent *mood* vector (4-8 dim, slow-drifting over minutes) biases the
weights of the three rewards. Sleepy mood weighs comfort highly; alert mood
weighs curiosity. Mood drift gives session-to-session behavioral variety.

The policy outputs a structured action: (skill_id, target). Sampled with a
nonzero temperature so behavior is unpredictable but coherent.
"""
