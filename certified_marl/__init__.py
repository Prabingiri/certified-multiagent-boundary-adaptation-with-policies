r"""Certified multi-agent collaboration via boundary adaptation.

Reference implementation for the paper. The default path is
certification-first:

* ``env`` implements the CSG-RAG environment, arrivals, and CS-LSTF service.
* ``shield`` implements the seven-clause feasibility kernel and matching.
* ``metrics`` reports safety counters, response summaries, load CV, and
  admission/rejection throughput.
* ``models`` and ``trainers`` implement the masked PPO CPAC layer.
* ``objectives`` implements the pairwise CPAC reward and reward normalizer.

Safety is enforced by CS-LSTF and the feasibility kernel. CPAC only chooses
among masked feasible boundary actions.
"""

__version__ = "0.2.0"