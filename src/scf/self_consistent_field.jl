include("scf_callbacks.jl")

# Struct to store some options for forward-diff / reverse-diff response
# (unused in primal calculations)
@kwdef struct ResponseOptions
    verbose = false
end

function default_n_bands(model)
    n_spin = model.n_spin_components
    min_n_bands = div(model.n_electrons, n_spin * filled_occupation(model), RoundUp)
    n_extra = model.temperature == 0 ? 0 : ceil(Int, 0.1 * min_n_bands)
    min_n_bands + n_extra
end
default_occupation_threshold() = 1e-6

"""
Obtain new density ρ by diagonalizing `ham`. Converges as many bands as needed
to ensure that the orbitals of lowest `occupation` are occupied to
at most `occupation_threshold`. To obtain rapid convergence

`n_bands` is only an initial guess and a lower
bound for this number.
"""
function next_density(ham::Hamiltonian;
                      eigensolver=lobpcg_hyper,
                      ψ=nothing,            # as an initial guess
                      eigenvalues=nothing,  # to determine number of bands to compute
                      occupation=nothing,   # to determine number of bands to converge
                      occupation_threshold=default_occupation_threshold(),
                      n_bands=default_n_bands(ham.basis.model),  # Min. bands to converge
                      kwargs...)
    # Determine number of bands to be actually converged
    autoadjusted_bands = false
    if !isnothing(occupation)
        n_bands_occ = maximum(occupation) do occk
            something(findlast(onk -> onk ≥ occupation_threshold, occk), length(occk) + 1)
        end
        n_bands = max(n_bands, n_bands_occ)
        autoadjusted_bands = true
    end

    # Determine number of bands to be computed
    n_bands_compute = n_bands + 3  # At least have 3 unconverged bands
    if !isnothing(eigenvalues)
        n_bands_compute_ε = maximum(eigenvalues) do εk
            ε_threshold = εk[n_bands] + 1e-2  # Ensure gap of 1e-2 to last unconverged band
            something(findlast(εnk -> εnk ≥ ε_threshold, εk), length(εk) + 1)
        end
        @show n_bands n_bands_compute_ε
        n_bands_compute = max(n_bands_compute, n_bands_compute_ε)
    end
    if !isnothing(ψ)
        @assert length(ψ) == length(ham.basis.kpoints)
        n_bands_compute = max(n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
    end
    @show n_bands n_bands_compute

    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute; ψguess=ψ,
                                     n_conv_check=n_bands, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Update density from new ψ
    occupation, εF = compute_occupation(ham.basis, eigres.λ)

    # Check maximal occupation of the unconverged bands is sensible.
    minocc = maximum(minimum, occupation)
    model = ham.basis.model
    if model.temperature > 0 && minocc > occupation_threshold && autoadjusted_bands
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try increasing n_bands beyond the default of $(default_n_bands(model))")
    end

    # TODO Set occupation values below occupation_threshold explicitly to zero
    #      and ignore these bands in density computation ?

    ρout = compute_density(ham.basis, eigres.X, occupation)
    (ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres, n_bands)
end


"""
Solve the Kohn-Sham equations with a SCF algorithm, starting at `ρ`.

- `n_bands`: Minimal number of bands which are fully converged.
"""
@timing function self_consistent_field(basis::PlaneWaveBasis;
                                       ρ=guess_density(basis),
                                       ψ=nothing,
                                       n_bands=default_n_bands(basis.model),
                                       tol=1e-6,
                                       maxiter=100,
                                       solver=scf_nlsolve_solver(),
                                       eigensolver=lobpcg_hyper,
                                       determine_diagtol=ScfDiagtol(),
                                       occupation_threshold=default_occupation_threshold(),
                                       damping=0.8,  # Damping parameter
                                       mixing=LdosMixing(),
                                       is_converged=ScfConvergenceEnergy(tol),
                                       callback=ScfDefaultCallback(; show_damping=false),
                                       compute_consistent_energies=true,
                                       response=ResponseOptions(),  # Dummy here, only for AD
                                      )
    T = eltype(basis)

    # All these variables will get updated by fixpoint_map
    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end
    occupation = nothing
    eigenvalues = nothing
    ρout = ρ
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing
    info = (n_iter=0, ρin=ρ)   # Populate info with initial values
    converged = false
    n_bands_actual = n_bands  # number of actually converged bands might increase

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin)
        converged && return ρin  # No more iterations if convergence flagged
        n_iter += 1

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin, eigenvalues, εF)

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham; eigensolver, ψ, eigenvalues,
                                 occupation, occupation_threshold, n_bands,
                                 miniter=1, tol=determine_diagtol(info))
        (; ψ, eigenvalues, occupation, εF, ρout) = nextstate
        n_bands_actual = nextstate.n_bands

        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, occupation_threshold,
                nextstate..., diagonalization=[nextstate.diagonalization])

        # Compute the energy of the new state
        if compute_consistent_energies
            energies, _ = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF)
        end
        info = merge(info, (energies=energies, ))

        # Apply mixing and pass it the full info as kwargs
        δρ = mix_density(mixing, basis, ρout - ρin; info...)
        ρnext = ρin .+ T(damping) .* δρ
        info = merge(info, (; ρnext=ρnext))

        callback(info)
        is_converged(info) && (converged = true)

        ρnext
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map. Also we do not use the return value of fpres but rather the
    # one that got updated by fixpoint_map
    fpres = solver(fixpoint_map, ρout, maxiter; tol=eps(T))

    # We do not use the return value of fpres but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform
    # a last energy computation to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged, occupation_threshold,
            ρ, α=damping, eigenvalues, occupation, εF,
            n_iter, ψ, info.diagonalization, stage=:finalize,
            algorithm="SCF", norm_Δρ, n_bands=n_bands_actual)
    callback(info)
    info
end
