"""AdaptDNS — hybrid two-stage DNS-tunneling detection over encrypted (DoH/DoT) flows.

Architecture (see case study document, Section 3):
  Stage 1  — always-on passive profiling of encrypted-flow metadata
  Gate     — suspicion score vs. escalation-budget threshold
  Stage 2  — selective blackhole probing + retry-distribution analysis,
             parallel resolver-side traditional-DNS feature extraction
  Fusion   — combined feature vector -> Random Forest -> benign / tunneling
  Feedback — analyst-confirmed benign updates only the benign baseline
"""

__version__ = "0.1.0"
