from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import null_space

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency.
    torch = None
    _TORCH_AVAILABLE = False

try:  # Keep the classical MPS utilities usable even in environments without Qiskit.
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit_aer import AerSimulator

    _QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the execution environment.
    ClassicalRegister = None
    QuantumCircuit = None
    QuantumRegister = None
    AerSimulator = None
    transpile = None
    _QISKIT_AVAILABLE = False


FitMethod = Literal["tt_svd", "gradient"]


@dataclass(frozen=True)
class MpsTrainingResult:
    """Summary returned by :meth:`MLMpsCircuit.fit_target`.

    Attributes
    ----------
    method:
        Fitting method used. Currently ``"tt_svd"``.
    n_iters:
        Number of effective training iterations. TT-SVD is a one-shot classical
        fit, so this is always one after a successful call.
    converged:
        Whether the fitting routine completed successfully.
    bond_dim:
        Requested maximum MPS bond dimension.
    effective_bond_dim:
        Maximum bond dimension actually used by the compressed MPS.
    circuit_bond_dim:
        Power-of-two bond dimension used by the circuit embedding.
    truncation_error:
        Sum of squared discarded singular values across TT-SVD cuts.
    loss_history:
        Cross-entropy objective values. For TT-SVD this contains the final value.
    metrics_history:
        Distribution metrics after each effective training step.
    """

    method: str
    n_iters: int
    converged: bool
    bond_dim: int
    effective_bond_dim: int
    circuit_bond_dim: int
    truncation_error: float
    loss_history: list[float]
    metrics_history: list[dict[str, float]]


class MLMpsCircuit:
    r"""
    Matrix Product State (MPS) state-preparation wrapper for CVA/QCBM-style
    distribution learning.

    This class is meant as the MPS analogue of ``MLQcbmCircuit``:

    * it learns a target probability vector ``ptg`` over ``2**n_qubits`` states;
    * it represents the target quantum state with amplitudes ``sqrt(ptg)`` as an
      open-boundary MPS with bounded bond dimension;
    * it converts the MPS into a sequence of local isometric unitaries following
      the Alcazar et al. / Schon et al. construction;
    * it exposes QCBM-like helpers: ``probabilities``, ``cost_value``,
      ``cost_fn`` and ``metrics``.

    Important conventions
    ---------------------
    Probability-vector indexing matches Qiskit's ``Statevector.probabilities``:

    ``index = sum(bit(q) * 2**q for q in range(n_qubits))``.

    Internally, tensors are stored as open-boundary left-canonical tensors
    ``A[k]`` with shape ``(D_left, 2, D_right)`` such that

    ``psi[b0, ..., b_{n-1}] = A[0][0,b0,a1] ... A[n-1][a_{n-1},b_{n-1},0]``.

    Circuit construction
    --------------------
    The circuit is built from right to left. At site ``k`` the right bond is
    encoded on the qubits immediately to the left of the already-prepared suffix;
    a local unitary embeds the isometry

    ``|right_bond> -> sum_{left_bond, bit} A[k][left_bond, bit, right_bond]
       |left_bond>|bit>``.

    This is the operational version of the SVD/isometry recipe in Appendix B of
    Alcazar et al. It uses overlapping windows of at most
    ``ceil(log2(circuit_bond_dim)) + 1`` qubits. The windowed unitaries are added
    as dense ``UnitaryGate`` objects; Qiskit transpilation can decompose them to
    the desired backend basis.

    Notes
    -----
    ``fit_target`` uses TT-SVD. This is a deterministic classical compression of
    the target amplitude vector, not a stochastic variational optimizer. It gives
    you a trainable/benchmarkable MPS state preparation with the same loss and
    metric interface as the QCBM, but it does not require a parameter vector.
    """

    def __init__(
        self,
        n_qubits: int,
        *,
        bond_dim: int = 2,
        name: str = "G_p_mps",
        backend=None,
        transpile_backend=None,
        noise_model=None,
        coupling_map=None,
        basis_gates: list[str] | None = None,
        simulation_method: str = "automatic",
        optimization_level: int = 1,
        initial_layout: list[int] | None = None,
        layout_method: str | None = None,
        routing_method: str | None = None,
        seed_transpiler: int | None = None,
        build_circuit_on_fit: bool = True,
    ) -> None:
        if int(n_qubits) <= 0:
            raise ValueError("n_qubits must be positive.")
        if int(bond_dim) <= 0:
            raise ValueError("bond_dim must be positive.")

        self._n_qubits = int(n_qubits)
        self._bond_dim = int(bond_dim)
        self._name = str(name)
        self._build_circuit_on_fit = bool(build_circuit_on_fit)

        self._transpile_backend = transpile_backend
        self._layout_method = layout_method
        self._routing_method = routing_method
        self._seed_transpiler = seed_transpiler
        self._optimization_level = int(optimization_level)
        self._noise_model = noise_model
        self._coupling_map = coupling_map
        self._basis_gates = (
            list(basis_gates)
            if basis_gates is not None
            else ["cz", "id", "rx", "rz", "rzz", "sx", "x"]
        )
        self._simulation_method = str(simulation_method)
        self._initial_layout = list(initial_layout) if initial_layout is not None else None

        if backend is None and _QISKIT_AVAILABLE:
            self._backend = AerSimulator(
                method=self._simulation_method,
                noise_model=self._noise_model,
                coupling_map=self._coupling_map,
                basis_gates=self._basis_gates,
            )
        else:
            self._backend = backend

        self.tensors: list[np.ndarray] | None = None
        self._statevector: np.ndarray | None = None
        self._last_target: np.ndarray | None = None
        self._last_training_result: MpsTrainingResult | None = None
        self._effective_bond_dim = 1
        self._circuit_bond_dim = 1
        self._d_qubits = 0

        self.qc = None
        self._qc_meas = None
        self._tqc = None
        self._tqc_meas = None
        self._n_clbits = self._n_qubits
        self._clbit_indices = list(range(self._n_qubits))

    # =========================================================
    # Basic properties
    # =========================================================
    @property
    def n_qubits(self) -> int:
        """Number of physical qubits in the state-preparation register."""
        return self._n_qubits

    @property
    def dim(self) -> int:
        """Hilbert-space dimension, equal to ``2**n_qubits``."""
        return 2**self._n_qubits

    @property
    def bond_dim(self) -> int:
        """Requested maximum MPS bond dimension."""
        return self._bond_dim

    @property
    def effective_bond_dim(self) -> int:
        """Maximum bond dimension actually present after compression."""
        return self._effective_bond_dim

    @property
    def circuit_bond_dim(self) -> int:
        """Power-of-two bond dimension used in the circuit embedding."""
        return self._circuit_bond_dim

    @property
    def d_qubits(self) -> int:
        """Number of qubits needed to encode the circuit bond dimension."""
        return self._d_qubits

    @property
    def n_params(self) -> int:
        """Number of classical MPS tensor parameters for gradient training."""
        if self.tensors is not None:
            return int(sum(np.asarray(t).size for t in self.tensors))
        bonds = self._default_bond_dims(self.n_qubits, self.bond_dim)
        return int(sum(bonds[i] * 2 * bonds[i + 1] for i in range(self.n_qubits)))

    @property
    def last_training_result(self) -> MpsTrainingResult | None:
        """Most recent training result, or ``None`` before fitting."""
        return self._last_training_result

    # =========================================================
    # Fitting / training
    # =========================================================
    @classmethod
    def from_target(
        cls,
        ptg: np.ndarray,
        *,
        bond_dim: int = 2,
        name: str = "G_p_mps",
        cutoff: float = 0.0,
        build_circuit_on_fit: bool = True,
        **kwargs,
    ) -> "MLMpsCircuit":
        """Construct an MPS circuit and immediately fit it to ``ptg``."""
        p = np.asarray(ptg, dtype=float).ravel()
        n_qubits = cls._infer_n_qubits_from_dim(p.shape[0])
        obj = cls(
            n_qubits,
            bond_dim=bond_dim,
            name=name,
            build_circuit_on_fit=build_circuit_on_fit,
            **kwargs,
        )
        obj.fit_target(p, cutoff=cutoff, rebuild_circuit=build_circuit_on_fit)
        return obj

    def train(
        self,
        ptg: np.ndarray,
        *,
        method: FitMethod = "gradient",
        cutoff: float = 0.0,
        rebuild_circuit: bool | None = None,
        init: Literal["random", "tt_svd", "current"] = "tt_svd",
        optimizer: Literal["adam", "lbfgs"] = "adam",
        maxiter: int = 1000,
        lr: float = 5e-2,
        tol: float = 1e-10,
        miniter: int = 25,
        seed: int | None = 42,
        init_scale: float = 1e-2,
        eps: float = 1e-12,
        callback: Callable[[int, float, dict[str, float]], None] | None = None,
    ) -> MpsTrainingResult:
        """Alias for :meth:`fit_target`, matching QCBM training language."""
        return self.fit_target(
            ptg,
            method=method,
            cutoff=cutoff,
            rebuild_circuit=rebuild_circuit,
            init=init,
            optimizer=optimizer,
            maxiter=maxiter,
            lr=lr,
            tol=tol,
            miniter=miniter,
            seed=seed,
            init_scale=init_scale,
            eps=eps,
            callback=callback,
        )

    def fit_target(
        self,
        ptg: np.ndarray,
        *,
        method: FitMethod = "gradient",
        cutoff: float = 0.0,
        rebuild_circuit: bool | None = None,
        init: Literal["random", "tt_svd", "current"] = "tt_svd",
        optimizer: Literal["adam", "lbfgs"] = "adam",
        maxiter: int = 1000,
        lr: float = 5e-2,
        tol: float = 1e-10,
        miniter: int = 25,
        seed: int | None = 42,
        init_scale: float = 1e-2,
        eps: float = 1e-12,
        callback: Callable[[int, float, dict[str, float]], None] | None = None,
    ) -> MpsTrainingResult:
        r"""
        Fit the MPS to a target probability distribution.

        Parameters
        ----------
        ptg:
            Target probability vector of length ``2**n_qubits``.
        method:
            ``"gradient"`` performs iterative classical training of the MPS
            tensors by minimizing cross-entropy/KL with PyTorch autograd.
            ``"tt_svd"`` keeps the previous deterministic tensor-train SVD
            compression baseline.
        init:
            Initialization for gradient training:
            ``"random"`` uses random open-boundary tensors;
            ``"tt_svd"`` initializes from the deterministic TT-SVD fit;
            ``"current"`` starts from the currently stored tensors.
        optimizer:
            ``"adam"`` or ``"lbfgs"`` for gradient training.
        maxiter:
            Maximum gradient iterations.
        lr:
            Optimizer learning rate.
        tol:
            Stop when the absolute loss improvement is below this value after
            ``miniter`` iterations.
        rebuild_circuit:
            Whether to rebuild/transpile the Qiskit circuit after fitting.
        callback:
            Optional function called as ``callback(iter_idx, loss, metrics)``.

        Notes
        -----
        The gradient stage optimizes unconstrained real MPS tensors. After the
        final step, the learned statevector is canonicalized by a TT-SVD with the
        same ``bond_dim`` so that the Alcazar/Schon isometry-to-circuit
        construction remains valid. This canonicalization is not used as the
        fitting objective; it is only the final gauge/isometry conversion step.
        """
        if cutoff < 0.0 or not np.isfinite(float(cutoff)):
            raise ValueError("cutoff must be a finite non-negative number.")
        if method == "tt_svd":
            return self._fit_target_ttsvd(
                ptg,
                cutoff=cutoff,
                rebuild_circuit=rebuild_circuit,
            )
        if method == "gradient":
            return self._fit_target_gradient(
                ptg,
                cutoff=cutoff,
                rebuild_circuit=rebuild_circuit,
                init=init,
                optimizer=optimizer,
                maxiter=maxiter,
                lr=lr,
                tol=tol,
                miniter=miniter,
                seed=seed,
                init_scale=init_scale,
                eps=eps,
                callback=callback,
            )
        raise NotImplementedError("method must be 'gradient' or 'tt_svd'.")

    def _fit_target_ttsvd(
        self,
        ptg: np.ndarray,
        *,
        cutoff: float = 0.0,
        rebuild_circuit: bool | None = None,
    ) -> MpsTrainingResult:
        """Deterministic TT-SVD compression baseline."""
        ptg_v = self._validate_target(ptg, dim=self.dim)
        psi = np.sqrt(ptg_v).astype(complex)

        tensors, truncation_error = self._tt_svd_statevector(
            psi,
            n_qubits=self.n_qubits,
            max_bond_dim=self.bond_dim,
            cutoff=float(cutoff),
        )
        tensors = self._normalize_tensors(tensors)
        state = self.mps_to_statevector(tensors)
        state = state / float(np.linalg.norm(state))

        self._store_fitted_tensors(
            tensors=tensors,
            state=state,
            ptg=ptg_v,
            rebuild_circuit=rebuild_circuit,
        )

        p_model = np.abs(self._statevector) ** 2
        p_model = p_model / float(p_model.sum())
        ce = self.cross_entropy(ptg_v, p_model)
        metrics = self.metrics(ptg_v, p_model)

        result = MpsTrainingResult(
            method="tt_svd",
            n_iters=1,
            converged=True,
            bond_dim=self.bond_dim,
            effective_bond_dim=self.effective_bond_dim,
            circuit_bond_dim=self.circuit_bond_dim,
            truncation_error=float(truncation_error),
            loss_history=[float(ce)],
            metrics_history=[metrics],
        )
        self._last_training_result = result
        return result

    def _fit_target_gradient(
        self,
        ptg: np.ndarray,
        *,
        cutoff: float = 0.0,
        rebuild_circuit: bool | None = None,
        init: Literal["random", "tt_svd", "current"] = "tt_svd",
        optimizer: Literal["adam", "lbfgs"] = "adam",
        maxiter: int = 1000,
        lr: float = 5e-2,
        tol: float = 1e-10,
        miniter: int = 25,
        seed: int | None = 42,
        init_scale: float = 1e-2,
        eps: float = 1e-12,
        callback: Callable[[int, float, dict[str, float]], None] | None = None,
    ) -> MpsTrainingResult:
        """Gradient-based MPS fitting by exact cross-entropy minimization."""
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "Gradient MPS training requires PyTorch. Install torch or use method='tt_svd'."
            )
        if int(maxiter) <= 0:
            raise ValueError("maxiter must be positive.")
        if float(lr) <= 0.0:
            raise ValueError("lr must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        ptg_v = self._validate_target(ptg, dim=self.dim)
        target_t = torch.tensor(ptg_v, dtype=torch.float64)

        init_tensors_np = self._initial_mps_tensors(
            ptg_v,
            init=init,
            cutoff=cutoff,
            seed=seed,
            init_scale=init_scale,
        )
        params = [
            torch.tensor(np.real(t), dtype=torch.float64, requires_grad=True)
            for t in init_tensors_np
        ]

        loss_history: list[float] = []
        metrics_history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_params: list[np.ndarray] | None = None
        converged = False

        def closure_loss():
            psi = self._torch_mps_to_statevector(params)
            prob = psi.square()
            prob = prob / torch.clamp(prob.sum(), min=eps)
            return -torch.sum(target_t * torch.log(torch.clamp(prob, min=eps)))

        def record(iter_idx: int, loss_value: float) -> None:
            nonlocal best_loss, best_params
            with torch.no_grad():
                psi_now = self._torch_mps_to_statevector(params).detach().cpu().numpy()
                p_now = np.square(np.asarray(psi_now, dtype=float))
                p_now = p_now / float(p_now.sum())
                metrics = self.metrics(ptg_v, p_now, eps=eps)
            loss_history.append(float(loss_value))
            metrics_history.append(metrics)
            if loss_value < best_loss:
                best_loss = float(loss_value)
                best_params = [p.detach().cpu().numpy().copy() for p in params]
            if callback is not None:
                callback(iter_idx, float(loss_value), metrics)

        opt_name = str(optimizer).lower()
        previous_loss: float | None = None

        if opt_name == "adam":
            opt = torch.optim.Adam(params, lr=float(lr))
            for it in range(1, int(maxiter) + 1):
                opt.zero_grad()
                loss_t = closure_loss()
                loss_t.backward()
                opt.step()
                loss_value = float(loss_t.detach().cpu().item())
                record(it, loss_value)
                if previous_loss is not None and it >= int(miniter):
                    if abs(previous_loss - loss_value) <= float(tol):
                        converged = True
                        break
                previous_loss = loss_value
        elif opt_name == "lbfgs":
            opt = torch.optim.LBFGS(
                params,
                lr=float(lr),
                max_iter=1,
                line_search_fn="strong_wolfe",
            )
            for it in range(1, int(maxiter) + 1):
                def lbfgs_closure():
                    opt.zero_grad()
                    loss_inner = closure_loss()
                    loss_inner.backward()
                    return loss_inner

                loss_t = opt.step(lbfgs_closure)
                loss_value = float(loss_t.detach().cpu().item())
                record(it, loss_value)
                if previous_loss is not None and it >= int(miniter):
                    if abs(previous_loss - loss_value) <= float(tol):
                        converged = True
                        break
                previous_loss = loss_value
        else:
            raise ValueError("optimizer must be 'adam' or 'lbfgs'.")

        if best_params is None:
            best_params = [p.detach().cpu().numpy().copy() for p in params]

        # Build the learned statevector from the best observed tensors.
        best_tensors = [np.asarray(t, dtype=float).astype(complex) for t in best_params]
        learned_state = self.mps_to_statevector(best_tensors)
        learned_norm = float(np.linalg.norm(learned_state))
        if not np.isfinite(learned_norm) or learned_norm <= 0.0:
            raise RuntimeError("Gradient training produced a non-finite or zero state.")
        learned_state = learned_state / learned_norm

        # Canonicalize the learned state to obtain valid isometries for the circuit.
        tensors, truncation_error = self._tt_svd_statevector(
            learned_state,
            n_qubits=self.n_qubits,
            max_bond_dim=self.bond_dim,
            cutoff=float(cutoff),
        )
        tensors = self._normalize_tensors(tensors)
        state = self.mps_to_statevector(tensors)
        state = state / float(np.linalg.norm(state))

        self._store_fitted_tensors(
            tensors=tensors,
            state=state,
            ptg=ptg_v,
            rebuild_circuit=rebuild_circuit,
        )

        p_model = np.abs(self._statevector) ** 2
        p_model = p_model / float(p_model.sum())
        final_ce = self.cross_entropy(ptg_v, p_model, eps=eps)
        final_metrics = self.metrics(ptg_v, p_model, eps=eps)

        # Ensure the stored final/canonical point is represented in histories.
        if not loss_history or abs(loss_history[-1] - final_ce) > 1e-14:
            loss_history.append(float(final_ce))
            metrics_history.append(final_metrics)

        result = MpsTrainingResult(
            method=f"gradient:{opt_name}:init={init}",
            n_iters=len(loss_history),
            converged=bool(converged),
            bond_dim=self.bond_dim,
            effective_bond_dim=self.effective_bond_dim,
            circuit_bond_dim=self.circuit_bond_dim,
            truncation_error=float(truncation_error),
            loss_history=loss_history,
            metrics_history=metrics_history,
        )
        self._last_training_result = result
        return result

    def _store_fitted_tensors(
        self,
        *,
        tensors: list[np.ndarray],
        state: np.ndarray,
        ptg: np.ndarray,
        rebuild_circuit: bool | None,
    ) -> None:
        """Store fitted tensors, update bond metadata and optionally rebuild the circuit."""
        self.tensors = [np.asarray(t, dtype=complex).copy() for t in tensors]
        self._statevector = np.asarray(state, dtype=complex).copy()
        self._last_target = np.asarray(ptg, dtype=float).copy()
        self._effective_bond_dim = max(
            max(t.shape[0], t.shape[2]) for t in self.tensors
        )
        self._circuit_bond_dim = self._next_power_of_two(self._effective_bond_dim)
        self._d_qubits = self._ceil_log2(self._circuit_bond_dim)

        do_rebuild = self._build_circuit_on_fit if rebuild_circuit is None else bool(rebuild_circuit)
        if do_rebuild:
            self.rebuild_circuit()

    def _initial_mps_tensors(
        self,
        ptg: np.ndarray,
        *,
        init: Literal["random", "tt_svd", "current"],
        cutoff: float,
        seed: int | None,
        init_scale: float,
    ) -> list[np.ndarray]:
        """Return initial tensors for gradient training."""
        init_l = str(init).lower()
        if init_l == "current":
            if self.tensors is None:
                raise RuntimeError("init='current' requires already fitted/stored MPS tensors.")
            return [np.real(np.asarray(t, dtype=complex)).copy() for t in self.tensors]

        if init_l == "tt_svd":
            tensors, _ = self._tt_svd_statevector(
                np.sqrt(ptg).astype(complex),
                n_qubits=self.n_qubits,
                max_bond_dim=self.bond_dim,
                cutoff=float(cutoff),
            )
            # A tiny perturbation breaks flat-gauge degeneracies without moving
            # far from the deterministic compression.
            if init_scale > 0.0:
                rng = np.random.default_rng(seed)
                tensors = [
                    np.real(t) + float(init_scale) * rng.standard_normal(t.shape)
                    for t in tensors
                ]
            return [np.asarray(np.real(t), dtype=float) for t in tensors]

        if init_l == "random":
            rng = np.random.default_rng(seed)
            bonds = self._default_bond_dims(self.n_qubits, self.bond_dim)
            tensors = []
            scale = float(init_scale) if init_scale > 0.0 else 1e-2
            for i in range(self.n_qubits):
                tensors.append(scale * rng.standard_normal((bonds[i], 2, bonds[i + 1])))
            return tensors

        raise ValueError("init must be 'random', 'tt_svd' or 'current'.")

    @staticmethod
    def _default_bond_dims(n_qubits: int, max_bond_dim: int) -> list[int]:
        """Open-boundary MPS bond dimensions capped by entanglement rank."""
        n = int(n_qubits)
        D = int(max_bond_dim)
        return [min(D, 2**min(i, n - i)) for i in range(n + 1)]

    def _torch_mps_to_statevector(self, tensors):
        """Contract real torch MPS tensors to a little-endian statevector."""
        arr = tensors[0][0, :, :]
        for tensor in tensors[1:]:
            arr = torch.tensordot(arr, tensor, dims=([-1], [0]))
        arr = arr[..., 0]
        if self.n_qubits == 1:
            return arr.reshape(-1)
        perm = tuple(reversed(range(self.n_qubits)))
        return arr.permute(perm).reshape(-1)

    def fit_statevector(
        self,
        psi: np.ndarray,
        *,
        cutoff: float = 0.0,
        rebuild_circuit: bool | None = None,
    ) -> MpsTrainingResult:
        """
        Fit directly to a statevector instead of a probability vector.

        This is useful if you later want phases. For CVA/QCBM probability
        loading, use :meth:`fit_target` so that amplitudes are ``sqrt(ptg)``.
        """
        psi = np.asarray(psi, dtype=complex).ravel()
        if psi.shape[0] != self.dim:
            raise ValueError(f"psi must have length {self.dim}; got {psi.shape[0]}.")
        norm = float(np.linalg.norm(psi))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("psi has non-finite or zero norm.")
        psi = psi / norm
        ptg = np.abs(psi) ** 2
        ptg = ptg / float(ptg.sum())

        tensors, truncation_error = self._tt_svd_statevector(
            psi,
            n_qubits=self.n_qubits,
            max_bond_dim=self.bond_dim,
            cutoff=float(cutoff),
        )
        tensors = self._normalize_tensors(tensors)
        state = self.mps_to_statevector(tensors)
        state = state / float(np.linalg.norm(state))

        self.tensors = tensors
        self._statevector = state
        self._last_target = ptg
        self._effective_bond_dim = max(
            max(t.shape[0], t.shape[2]) for t in self.tensors
        )
        self._circuit_bond_dim = self._next_power_of_two(self._effective_bond_dim)
        self._d_qubits = self._ceil_log2(self._circuit_bond_dim)

        do_rebuild = self._build_circuit_on_fit if rebuild_circuit is None else bool(rebuild_circuit)
        if do_rebuild:
            self.rebuild_circuit()

        p_model = np.abs(self._statevector) ** 2
        p_model = p_model / float(p_model.sum())
        ce = self.cross_entropy(ptg, p_model)
        metrics = self.metrics(ptg, p_model)
        result = MpsTrainingResult(
            method="tt_svd",
            n_iters=1,
            converged=True,
            bond_dim=self.bond_dim,
            effective_bond_dim=self.effective_bond_dim,
            circuit_bond_dim=self.circuit_bond_dim,
            truncation_error=float(truncation_error),
            loss_history=[float(ce)],
            metrics_history=[metrics],
        )
        self._last_training_result = result
        return result

    # =========================================================
    # Circuit construction
    # =========================================================
    def rebuild_circuit(self):
        """Rebuild and transpile the MPS state-preparation circuit."""
        self._require_qiskit()
        self._require_fitted()

        q = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(q, name=self._name)

        for site in range(self.n_qubits - 1, -1, -1):
            U, start, stop = self._local_unitary_for_site(site)
            qargs = [q[i] for i in range(start, stop + 1)]
            qc.unitary(U, qargs, label=f"MPS_U[{site}]")

        self.qc = qc
        self._qc_meas = self._build_measured_circuit(qc)
        self._tqc = self._transpile_for_backend(self.qc)
        self._tqc_meas = self._transpile_for_backend(self._qc_meas)
        return self.qc

    def bind(self, x: np.ndarray | None = None, *, measured: bool = False):
        """
        Return the fitted MPS circuit.

        The MPS circuit has no variational parameter vector after classical
        fitting. Therefore ``x`` must be ``None`` or an empty vector.
        """
        self._require_fitted()
        if x is not None and np.asarray(x).size != 0:
            raise ValueError("The fitted MPS circuit has no variational parameters; pass x=None or an empty vector.")
        if measured:
            if self._qc_meas is None:
                self.rebuild_circuit()
            return self._qc_meas.copy()
        if self.qc is None:
            self.rebuild_circuit()
        return self.qc.copy()

    def to_instruction(self, *, measured: bool = False):
        """Return the fitted MPS circuit as a Qiskit instruction."""
        return self.bind(measured=measured).to_instruction()

    def _build_measured_circuit(self, qc):
        c = ClassicalRegister(self.n_qubits, "c")
        q = QuantumRegister(self.n_qubits, "q")
        out = QuantumCircuit(q, c, name=f"{self._name}_meas")
        out.compose(qc, qubits=list(out.qubits)[: self.n_qubits], inplace=True)
        for i in range(self.n_qubits):
            out.measure(q[i], c[i])

        self._n_clbits = len(out.clbits)
        self._clbit_indices = [out.clbits.index(c[i]) for i in range(self.n_qubits)]
        return out

    def _local_unitary_for_site(self, site: int) -> tuple[np.ndarray, int, int]:
        """
        Build the local Alcazar/Schon unitary for a single MPS site.

        Returns ``(U, start, stop)`` where ``U`` acts on qubits
        ``start, start+1, ..., stop`` in Qiskit's little-endian qarg order.
        """
        self._require_fitted()
        assert self.tensors is not None

        A = self.tensors[site]
        d_left, phys_dim, d_right = A.shape
        if phys_dim != 2:
            raise RuntimeError("Only qubit MPS tensors with physical dimension 2 are supported.")

        d = self.d_qubits
        start = max(0, site - d)
        stop = site
        width = stop - start + 1
        left_width = width - 1

        # For bulk sites, the right bond occupies the window except for the
        # fresh zero qubit at the left. For left-boundary sites, the right bond
        # already occupies the whole remaining prefix window.
        input_shift = 1 if (d > 0 and site >= d) else 0
        dim = 2**width

        cols = np.zeros((dim, d_right), dtype=complex)
        for alpha in range(d_left):
            for bit in range(2):
                row = alpha + (bit << left_width)
                for beta in range(d_right):
                    cols[row, beta] = A[alpha, bit, beta]

        selected_columns = [beta << input_shift for beta in range(d_right)]
        U = self._complete_unitary_from_selected_columns(
            cols,
            selected_columns=selected_columns,
            dim=dim,
        )
        return U, start, stop

    def _transpile_for_backend(self, circuit):
        self._require_qiskit()
        compile_backend = (
            self._transpile_backend
            if self._transpile_backend is not None
            else self._backend
        )

        kwargs: dict[str, object] = {
            "optimization_level": self._optimization_level,
        }
        if compile_backend is not None:
            kwargs["backend"] = compile_backend
        else:
            kwargs["basis_gates"] = self._basis_gates
            if self._coupling_map is not None:
                kwargs["coupling_map"] = self._coupling_map

        if self._initial_layout is not None:
            kwargs["initial_layout"] = self._initial_layout
        if self._layout_method is not None:
            kwargs["layout_method"] = self._layout_method
        if self._routing_method is not None:
            kwargs["routing_method"] = self._routing_method
        if self._seed_transpiler is not None:
            kwargs["seed_transpiler"] = self._seed_transpiler

        return transpile(circuit, **kwargs)

    # =========================================================
    # Probabilities
    # =========================================================
    def statevector(self) -> np.ndarray:
        """Return the fitted MPS statevector in Qiskit little-endian order."""
        self._require_fitted()
        assert self._statevector is not None
        return self._statevector.copy()

    def probabilities(
        self,
        x: np.ndarray | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
        use_backend: bool = True,
    ) -> np.ndarray:
        """
        Compute probabilities from the fitted MPS.

        Parameters
        ----------
        x:
            Ignored empty parameter vector, accepted for QCBM-like API
            compatibility.
        shots:
            ``None`` returns exact MPS Born probabilities. A positive integer
            returns sampled probabilities. If a backend and circuit are available
            and ``use_backend=True``, samples are obtained from the backend;
            otherwise they are sampled classically from exact probabilities.
        seed:
            Random seed for either simulator/backend or classical sampling.
        use_backend:
            Whether shot-based calls should use the Qiskit backend if possible.
        """
        self._require_fitted()
        if x is not None and np.asarray(x).size != 0:
            raise ValueError("The fitted MPS circuit has no variational parameters; pass x=None or an empty vector.")

        p_exact = np.abs(self._statevector) ** 2
        p_exact = np.maximum(p_exact, 0.0)
        p_exact = p_exact / float(p_exact.sum())

        if shots is None:
            # Exact ideal path is cheaper and avoids Qiskit ordering surprises.
            if self._noise_model is None or not use_backend:
                return p_exact.copy()
            return self._backend_probabilities(seed=seed)

        shots_i = int(shots)
        if shots_i <= 0:
            raise ValueError("shots must be a positive integer.")

        if use_backend and _QISKIT_AVAILABLE and self._backend is not None:
            if self._tqc_meas is None:
                self.rebuild_circuit()
            run_kwargs: dict[str, object] = {"shots": shots_i}
            if seed is not None:
                run_kwargs["seed_simulator"] = int(seed)
            counts = self._backend.run(self._tqc_meas, **run_kwargs).result().get_counts()
            return self._counts_to_probabilities(counts, shots=shots_i)

        rng = np.random.default_rng(seed)
        samples = rng.choice(self.dim, size=shots_i, p=p_exact)
        counts = np.bincount(samples, minlength=self.dim).astype(float)
        return counts / float(shots_i)

    def _backend_probabilities(self, *, seed: int | None = None) -> np.ndarray:
        self._require_qiskit()
        self._require_fitted()
        if self._backend is None:
            raise RuntimeError("No backend is available for backend probabilities.")
        if self._tqc is None:
            self.rebuild_circuit()

        qc_run = self._tqc.copy()
        active_qubits = (
            self._initial_layout
            if self._initial_layout is not None
            else list(range(self.n_qubits))
        )
        qc_run.save_probabilities(qubits=active_qubits)

        run_kwargs: dict[str, object] = {}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        result = self._backend.run(qc_run, **run_kwargs).result()
        p = np.asarray(result.data(0)["probabilities"], dtype=float)
        p = np.maximum(p, 0.0)
        p /= float(p.sum())
        return p

    def _counts_to_probabilities(
        self,
        counts: dict[str, int],
        *,
        shots: int,
    ) -> np.ndarray:
        """Convert Qiskit count strings into Statevector-compatible ordering."""
        p = np.zeros(self.dim, dtype=float)

        for raw_bs, c in counts.items():
            bs = raw_bs.replace(" ", "")
            if len(bs) != self._n_clbits:
                raise RuntimeError(
                    f"Unexpected bitstring length {len(bs)} (expected {self._n_clbits})."
                )

            def bit_at_clbit_index(cl_idx: int) -> int:
                pos_from_left = (self._n_clbits - 1) - cl_idx
                return 1 if bs[pos_from_left] == "1" else 0

            x_val = 0
            for q in range(self.n_qubits):
                b = bit_at_clbit_index(self._clbit_indices[q])
                x_val |= b << q

            p[x_val] += float(c)

        p /= float(shots)
        return p

    # =========================================================
    # Costs and metrics
    # =========================================================
    @staticmethod
    def _validate_target(ptg: np.ndarray, *, dim: int) -> np.ndarray:
        """Validate and normalize a probability vector."""
        ptg = np.asarray(ptg, dtype=float).ravel()
        if ptg.shape[0] != dim:
            raise ValueError(f"ptg must have length {dim}; got {ptg.shape[0]}.")
        if np.any(ptg < 0.0):
            raise ValueError("ptg contains negative entries.")
        s = float(ptg.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("ptg has non-finite or non-positive sum.")
        if abs(s - 1.0) > 1e-12:
            ptg = ptg / s
        return ptg

    @staticmethod
    def entropy(ptg: np.ndarray, *, eps: float = 1e-12) -> float:
        """Compute Shannon entropy ``H(ptg) = -sum ptg log(ptg)``."""
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        ptg = np.asarray(ptg, dtype=float).ravel()
        s = float(ptg.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("ptg has non-finite or non-positive sum.")
        ptg = ptg / s
        return float(-np.sum(ptg * np.log(np.maximum(ptg, eps))))

    @staticmethod
    def cross_entropy(ptg: np.ndarray, p: np.ndarray, *, eps: float = 1e-12) -> float:
        """Compute ``CE(ptg, p) = -sum ptg log(p)``."""
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        ptg = np.asarray(ptg, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()
        if ptg.shape != p.shape:
            raise ValueError(f"Shape mismatch: ptg{ptg.shape} vs p{p.shape}.")
        ptg = ptg / float(ptg.sum())
        p = p / float(p.sum())
        return float(-np.sum(ptg * np.log(np.maximum(p, eps))))

    def cost_value(
        self,
        x: np.ndarray | None,
        ptg: np.ndarray | None = None,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
        rescaled: bool = False,
        use_backend: bool = True,
    ) -> float:
        """
        Evaluate QCBM-style cross-entropy cost.

        Supports both ``cost_value(x, ptg)`` and ``cost_value(ptg)``. Since the
        fitted MPS has no variational parameters, ``x`` is ignored if empty.
        """
        if ptg is None:
            ptg = x
            x = None
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        ptg_v = self._validate_target(ptg, dim=self.dim)
        p = self.probabilities(x, shots=shots, seed=seed, use_backend=use_backend)
        ce = self.cross_entropy(ptg_v, p, eps=eps)
        if rescaled:
            ce -= self.entropy(ptg_v, eps=eps)
        return float(ce)

    def cost_value_rescaled(
        self,
        x: np.ndarray | None,
        ptg: np.ndarray | None = None,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
        use_backend: bool = True,
    ) -> float:
        """Evaluate ``CE(ptg, p_mps) - H(ptg)``."""
        return self.cost_value(
            x,
            ptg,
            eps=eps,
            shots=shots,
            seed=seed,
            rescaled=True,
            use_backend=use_backend,
        )

    def cost_fn(
        self,
        ptg: np.ndarray,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
        rescaled: bool = False,
        use_backend: bool = True,
    ) -> Callable[[np.ndarray], float]:
        """Return an optimizer-compatible objective ``f(x)``."""
        ptg_v = self._validate_target(ptg, dim=self.dim)

        def cost(x: np.ndarray | None = None) -> float:
            return self.cost_value(
                x,
                ptg_v,
                eps=eps,
                shots=shots,
                seed=seed,
                rescaled=rescaled,
                use_backend=use_backend,
            )

        return cost

    @staticmethod
    def metrics(
        ptg: np.ndarray,
        p: np.ndarray,
        *,
        eps: float = 1e-12,
    ) -> dict[str, float]:
        """
        Compute distribution metrics: KL, L1, total variation and L-infinity.
        """
        ptg = np.asarray(ptg, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()
        if ptg.shape != p.shape:
            raise ValueError(f"Shape mismatch: ptg{ptg.shape} vs p{p.shape}.")
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")

        s_ptg = float(ptg.sum())
        s_p = float(p.sum())
        if s_ptg <= 0.0 or not np.isfinite(s_ptg):
            raise ValueError("ptg has non-finite or non-positive sum.")
        if s_p <= 0.0 or not np.isfinite(s_p):
            raise ValueError("p has non-finite or non-positive sum.")

        ptg = ptg / s_ptg
        p = p / s_p
        ptg_c = np.maximum(ptg, eps)
        p_c = np.maximum(p, eps)
        diff = p - ptg

        return {
            "kl": float(np.sum(ptg_c * np.log(ptg_c / p_c))),
            "l1": float(np.sum(np.abs(diff))),
            "tv": float(0.5 * np.sum(np.abs(diff))),
            "linf": float(np.max(np.abs(diff))),
        }

    def fitted_metrics(self, ptg: np.ndarray | None = None, *, eps: float = 1e-12) -> dict[str, float]:
        """Metrics of the fitted MPS against ``ptg`` or the last fitted target."""
        self._require_fitted()
        if ptg is None:
            if self._last_target is None:
                raise ValueError("No target was provided and no previous target is stored.")
            ptg = self._last_target
        return self.metrics(ptg, self.probabilities(), eps=eps)

    # =========================================================
    # Classical MPS utilities
    # =========================================================
    @staticmethod
    def mps_to_statevector(tensors: Sequence[np.ndarray]) -> np.ndarray:
        """Contract open-boundary tensors to a statevector in Qiskit order."""
        if len(tensors) == 0:
            raise ValueError("tensors must be non-empty.")
        arr = np.asarray(tensors[0])[0, :, :]
        for tensor in tensors[1:]:
            arr = np.tensordot(arr, np.asarray(tensor), axes=([-1], [0]))
        arr = arr[..., 0]
        return np.asarray(arr.reshape(-1, order="F"), dtype=complex)

    @staticmethod
    def _tt_svd_statevector(
        psi: np.ndarray,
        *,
        n_qubits: int,
        max_bond_dim: int,
        cutoff: float = 0.0,
    ) -> tuple[list[np.ndarray], float]:
        """Tensor-train SVD of a statevector in Qiskit little-endian order."""
        psi = np.asarray(psi, dtype=complex).ravel()
        if psi.shape[0] != 2**n_qubits:
            raise ValueError(
                f"psi length must be 2**n_qubits={2**n_qubits}; got {psi.shape[0]}."
            )
        if max_bond_dim <= 0:
            raise ValueError("max_bond_dim must be positive.")

        tensor = psi.reshape((2,) * n_qubits, order="F")
        current = tensor.reshape(1, 2, *([2] * (n_qubits - 1)), order="F")
        left_rank = 1
        tensors: list[np.ndarray] = []
        truncation_error = 0.0

        for site in range(n_qubits - 1):
            matrix = current.reshape((left_rank * 2, -1), order="F")
            U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)

            if cutoff > 0.0:
                rank_cutoff = int(np.sum(singular_values > cutoff))
                rank_cutoff = max(rank_cutoff, 1)
            else:
                rank_cutoff = len(singular_values)

            rank = min(int(max_bond_dim), rank_cutoff, len(singular_values))
            discarded = singular_values[rank:]
            truncation_error += float(np.sum(np.abs(discarded) ** 2))

            U = U[:, :rank]
            S = singular_values[:rank]
            Vh = Vh[:rank, :]

            tensors.append(U.reshape((left_rank, 2, rank), order="F"))
            current = (S[:, None] * Vh).reshape(
                rank,
                2,
                *([2] * (n_qubits - site - 2)),
                order="F",
            )
            left_rank = rank

        tensors.append(current.reshape((left_rank, 2, 1), order="F"))
        return tensors, float(truncation_error)

    @staticmethod
    def _normalize_tensors(tensors: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize the MPS by absorbing the global norm into the last tensor."""
        out = [np.asarray(t, dtype=complex).copy() for t in tensors]
        psi = MLMpsCircuit.mps_to_statevector(out)
        norm = float(np.linalg.norm(psi))
        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("Cannot normalize an MPS with non-finite or zero norm.")
        out[-1] = out[-1] / norm
        return out

    @staticmethod
    def _complete_unitary_from_selected_columns(
        columns: np.ndarray,
        *,
        selected_columns: Sequence[int],
        dim: int,
        orth_tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Complete an isometry to a square unitary with columns placed at selected indices.
        """
        columns = np.asarray(columns, dtype=complex)
        if columns.ndim != 2 or columns.shape[0] != dim:
            raise ValueError("columns must have shape (dim, k).")
        selected = list(map(int, selected_columns))
        if len(selected) != columns.shape[1]:
            raise ValueError("selected_columns length must match the number of columns.")
        if len(set(selected)) != len(selected):
            raise ValueError("selected_columns contains duplicates.")
        if any(idx < 0 or idx >= dim for idx in selected):
            raise ValueError("selected_columns has an index outside [0, dim).")

        k = columns.shape[1]
        gram_err = float(np.linalg.norm(columns.conj().T @ columns - np.eye(k)))
        if gram_err > orth_tol:
            # This should not happen for TT-SVD tensors, but QR makes the method
            # robust to roundoff or externally supplied tensors.
            q_cols, _ = np.linalg.qr(columns, mode="reduced")
            columns = q_cols[:, :k]
            gram_err = float(np.linalg.norm(columns.conj().T @ columns - np.eye(k)))
            if gram_err > 10.0 * orth_tol:
                raise RuntimeError(
                    f"Could not orthonormalize isometry columns; residual={gram_err:.3e}."
                )

        complement = null_space(columns.conj().T)
        if complement.shape[1] != dim - k:
            raise RuntimeError("Failed to compute a full orthogonal complement for the isometry.")

        U = np.zeros((dim, dim), dtype=complex)
        remaining = [idx for idx in range(dim) if idx not in selected]
        for j, idx in enumerate(selected):
            U[:, idx] = columns[:, j]
        for j, idx in enumerate(remaining):
            U[:, idx] = complement[:, j]

        unitary_err = float(np.linalg.norm(U.conj().T @ U - np.eye(dim)))
        if unitary_err > 1e-7:
            raise RuntimeError(f"Local MPS unitary is not unitary; residual={unitary_err:.3e}.")
        return U

    # =========================================================
    # Diagnostics and resource estimates
    # =========================================================
    def circuit_summary(self) -> dict[str, object]:
        """Return basic diagnostics of the fitted MPS and its circuit embedding."""
        self._require_fitted()
        summary: dict[str, object] = {
            "n_qubits": self.n_qubits,
            "bond_dim_requested": self.bond_dim,
            "bond_dim_effective": self.effective_bond_dim,
            "bond_dim_circuit": self.circuit_bond_dim,
            "d_qubits": self.d_qubits,
            "max_local_window_qubits": min(self.n_qubits, self.d_qubits + 1),
            "n_raw_local_unitaries": self.n_qubits,
            "n_alcazar_bulk_unitaries": max(self.n_qubits - self.d_qubits, 0),
        }
        if self.qc is not None:
            summary["ops"] = dict(self.qc.count_ops())
            summary["depth"] = int(self.qc.depth())
        if self._tqc is not None:
            summary["transpiled_ops"] = dict(self._tqc.count_ops())
            summary["transpiled_depth"] = int(self._tqc.depth())
        return summary

    def alcazar_resource_estimate(self) -> dict[str, object]:
        """
        Lightweight Alcazar-style resource estimate.

        The exact decomposition of a generic ``(d+1)``-qubit unitary depends on
        the compiler/synthesis method. This routine therefore reports the exact
        closed form for ``D=2`` and a scaling summary otherwise.
        """
        self._require_fitted()
        n = self.n_qubits
        D = self.circuit_bond_dim
        d = self.d_qubits
        out: dict[str, object] = {
            "n_qubits": n,
            "D": D,
            "d": d,
            "max_window_qubits": min(n, d + 1),
            "large_unitaries": max(n - d, 0),
            "two_qubit_scaling": "O(n D^2)",
        }
        if D == 1:
            out["cnot"] = 0
            out["one_qubit"] = n
        elif D == 2:
            out["cnot"] = 3 * (n - 1)
            out["one_qubit"] = 6 * n - 5
        else:
            out["cnot"] = "depends on generic unitary synthesis; use transpiled circuit counts"
            out["one_qubit"] = "depends on generic unitary synthesis; use transpiled circuit counts"
        return out

    # =========================================================
    # Small helpers
    # =========================================================
    @staticmethod
    def _infer_n_qubits_from_dim(dim: int) -> int:
        dim_i = int(dim)
        if dim_i <= 0 or (dim_i & (dim_i - 1)) != 0:
            raise ValueError("The distribution length must be a positive power of two.")
        return int(np.log2(dim_i))

    @staticmethod
    def _ceil_log2(value: int) -> int:
        value_i = int(value)
        if value_i <= 1:
            return 0
        return int(np.ceil(np.log2(value_i)))

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        value_i = int(value)
        if value_i <= 1:
            return 1
        return 1 << (value_i - 1).bit_length()

    def _require_fitted(self) -> None:
        if self.tensors is None or self._statevector is None:
            raise RuntimeError("The MPS has not been fitted yet. Call fit_target(...) first.")

    @staticmethod
    def _require_qiskit() -> None:
        if not _QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for circuit construction/execution. "
                "The classical MPS utilities still work without Qiskit."
            )