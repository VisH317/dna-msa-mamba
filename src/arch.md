# ARCHITECTURE EXPLANATION

**Axial Attention but with selective scan**
- Horizontal operation - selective scan
  - need to run in its own block separately with all norms, convs, etc.
- Vertical operation - gated attention (with mask on unknown base pairs)
  - vertical attention operation will be its own block
- Previous method - axial attention
  - hard to scale to large context lengths
- Other locations
  - MSA transformer, retrieved sequence augmentation in proteins use a similar tactic
- 
