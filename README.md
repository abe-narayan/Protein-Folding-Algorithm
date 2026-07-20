\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{enumitem}

\title{Protein Folding Algorithm\\
\large Project 3}
\author{}
\date{}

\begin{document}

\maketitle

\section{Overview}

This project implements a hybrid quantum-classical algorithm for predicting the
lowest-energy conformation of a short protein using a lattice-based protein
folding model. The implementation follows the methods described in Robert et al.
(2021), using a Variational Quantum Eigensolver (VQE) with Conditional Value at
Risk (CVaR) optimization.

The project also benchmarks the quantum approach against classical optimization
methods such as brute-force search and simulated annealing.

\section{Objectives}

\begin{itemize}[leftmargin=*]
    \item Encode protein folds on a tetrahedral lattice.
    \item Construct the protein folding cost Hamiltonian.
    \item Implement a parameterized quantum circuit.
    \item Optimize using CVaR-VQE.
    \item Compare performance with classical algorithms.
\end{itemize}

\section{Project Structure}

\begin{verbatim}
protein-folding/
│
├── README.tex
├── requirements.txt
├── main.py
│
├── data/
├── docs/
├── notebooks/
├── src/
│   ├── encoding.py
│   ├── hamiltonian.py
│   ├── ansatz.py
│   ├── vqe.py
│   ├── classical.py
│   ├── benchmark.py
│   └── utils.py
│
├── tests/
└── results/
\end{verbatim}

\section{References}

\begin{enumerate}
    \item Robert et al., \emph{Resource-Efficient Quantum Algorithm for Protein Folding}, 2021.
    \item Hybrid quantum-classical approaches for HP lattice protein folding.
\end{enumerate}

\end{document}
