using ChainRulesCore
using Zygote: @adjoint  # TODO remove, once ChainRules rrules can overrule Zygote
import AbstractFFTs

function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, f_real::AbstractArray)
    f_fourier = r_to_G(basis, f_real)
    function r_to_G_pullback(Δf_fourier)
        ∂f_real = r_to_G_normalization * real(basis.opBFFT * Δf_fourier)
        ∂normalization = sum(Δf_fourier .* f_fourier)
        ∂basis = Tangent{typeof(basis)}(;r_to_G_normalization=∂normalization)
        return NoTangent(), ∂basis, ∂f_real
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, f_fourier::AbstractArray; kwargs...)
    f_real = G_to_r(basis, f_fourier; kwargs...)
    function G_to_r_pullback(Δf_real)
        ∂f_fourier = (basis.opFFT * Δf_real) * (length(basis.opFFT) / (basis.model.unit_cell_volume))
        ∂normalization = sum(Δf_real .* f_real)
        ∂basis = Tangent{typeof(basis)}(;G_to_r_normalization=∂normalization)
        return NoTangent(), ∂basis, ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

# function ChainRulesCore.rrule(::typeof(*), P::AbstractFFTs.ScaledPlan, x)
#     y = P * x
#     function mul_pullback(Δ)
#       N = prod(size(x)[[P.p.region...]])
#       ∂x = N * (P \ Δ)
#       ∂P = Tangent{typeof(P)}(;scale=sum(Δ .* y))
#       return NoTangent(), ∂P, ∂x 
#     end
#     return y, mul_pullback
# end

# # explicit Zygote.adjoint definitions are here bc Zygote.adjoint rules 
# # take precedence over ChainRulesCore.rrule, even if rrule is more specific.
# # Currently, Zygote defines an adjoint for *(P::AbstractFFTs.Plan, xs)
# # that fails for ScaledPlan (which has no field `region`).
# # TODO fix upstream and delete here
# @adjoint function *(P::AbstractFFTs.ScaledPlan, xs)
#     @warn "custom adjoint *(P::AbstractFFTs.ScaledPlan, xs)"
#     ys = P * xs
#     return ys, function(Δ)
#         N = prod(size(xs)[[P.p.region...]])
#         return ((;scale=sum(Δ .* ys)), N * (P \ Δ)) # or dot(Δ, ys) (complex conj) ?
#     end
# end
# @adjoint function \(P::AbstractFFTs.ScaledPlan, xs)
#     @warn "custom adjoint \\(P::AbstractFFTs.ScaledPlan, xs)"
#     ys = P \ xs
#     return ys, function(Δ)
#         N = prod(size(xs)[[P.p.region...]])
#         return ((;scale=-sum(Δ .* ys)/P.scale^2), (P * Δ) / N) # TODO check scale part (complex deriv?)
#     end
# end

# workaround rrules for mpi: treat as noop
function ChainRulesCore.rrule(::typeof(mpi_sum), arr, comm)
    function mpi_sum_pullback(Δy)
        return NoTangent(), Δy, NoTangent()
    end
    return arr, mpi_sum_pullback
end

ChainRulesCore.@non_differentiable ElementPsp(::Any...)
ChainRulesCore.@non_differentiable r_vectors(::Any...)
ChainRulesCore.@non_differentiable G_vectors(::Any...)

# TODO delete
@adjoint (T::Type{<:SArray})(x...) = T(x...), y->(y,)

# TODO delete, or understand why this is necessary
function ChainRulesCore.rrule(T::Type{Vector{Kpoint{Float64}}}, x)
    @warn "strange Vector{Kpoint{Float64}} rrule triggered"
    return T(x), ΔTx -> (NoTangent(), ΔTx)
end


# # simplified version of the Model constructor to
# # help reverse mode AD to only differentiate the relevant computations.
# # this excludes assertions (try-catch), and symmetries
function _autodiff_Model_namedtuple(lattice)
    T = eltype(lattice)
    recip_lattice = 2T(π)*inv(lattice')
    unit_cell_volume = abs(det(lattice))
    recip_cell_volume = abs(det(recip_lattice))
    (;lattice=lattice, recip_lattice=recip_lattice, unit_cell_volume=unit_cell_volume, recip_cell_volume=recip_cell_volume)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{Model}, lattice; kwargs...)
    @warn "simplified Model rrule triggered."
    model = T(lattice; kwargs...)
    _model, Model_pullback = rrule_via_ad(config, _autodiff_Model_namedtuple, lattice)
    # TODO add some assertion that model and _model agree
    # TODO treat atoms (figure out how to deal with differentiable kwargs)
    return model, Model_pullback
end

# a workaround to manually updating immutable Tangent fields
function _update_tangent(t::Tangent{P,T}, nt::NamedTuple) where {P,T}
    return Tangent{P,T}(merge(ChainRulesCore.backing(t), nt))
end

function ChainRulesCore.rrule(::typeof(build_kpoints), model::Model{T}, fft_size, kcoords, Ecut; variational=true) where T
    @warn "build_kpoints rrule triggered"
    kpoints = build_kpoints(model, fft_size, kcoords, Ecut; variational=variational)
    function build_kpoints_pullback(Δkpoints)
        ∂model = sum(Δkpoints).model # TODO double-check.
        ∂recip_lattice = ∂model.recip_lattice + sum([Δkp.coordinate_cart * kp.coordinate' for (kp, Δkp) in zip(kpoints, Δkpoints) if !(Δkp isa NoTangent)])
        # ∂model.recip_lattice += ∂recip_lattice # Tangents are immutable
        ∂model = _update_tangent(∂model, (;recip_lattice=∂recip_lattice))
        ∂kcoords = @not_implemented("TODO")
        return NoTangent(), ∂model, NoTangent(), ∂kcoords, NoTangent()
    end
    return kpoints, build_kpoints_pullback
end

# simplified version of PlaneWaveBasis constructor to
# help reverse mode AD to only differentiate the relevant computations.
# this excludes assertions (try-catch), MPI handling, and other things
function _autodiff_PlaneWaveBasis_namedtuple(model::Model{T}, basis::PlaneWaveBasis) where {T <: Real}
    dvol = model.unit_cell_volume ./ prod(basis.fft_size)
    G_to_r_normalization = 1 / sqrt(model.unit_cell_volume)
    r_to_G_normalization = sqrt(model.unit_cell_volume) / length(basis.ipFFT)

    # Create dummy terms array for _basis to handle
    terms = Vector{Any}(undef, length(model.term_types))

    # kpoints have differentiable components (inside model and coordinate_cart)
    kcoords_thisproc = basis.kcoords_global[basis.krange_thisproc] # TODO which kcoords?
    kpoints = build_kpoints(model, basis.fft_size, kcoords_thisproc, basis.Ecut; basis.variational)

    # cicularity is getting complicated...
    # To correctly instantiate term types, we do need a full PlaneWaveBasis struct;
    # so we need to interleave re-computed differentiable params, and fixed params in basis
    _basis = PlaneWaveBasis{T}( # this shouldn't hit the rrule below a second time due to more args
        model, basis.fft_size, dvol, 
        basis.Ecut, basis.variational,
        basis.opFFT, basis.ipFFT, basis.opBFFT, basis.ipBFFT,
        r_to_G_normalization, G_to_r_normalization,
        kpoints, basis.kweights, basis.ksymops, basis.kgrid, basis.kshift,
        basis.kcoords_global, basis.ksymops_global, basis.comm_kpts, basis.krange_thisproc, basis.krange_allprocs,
        basis.symmetries, terms)

    # terms = Any[t(_basis) for t in model.term_types]
    terms = vcat([], [t(_basis) for t in model.term_types]) # hack: enforce Vector{Any} without causing reverse mutation

    (;model=model, kpoints=kpoints, dvol=dvol, terms=terms, G_to_r_normalization=G_to_r_normalization, r_to_G_normalization=r_to_G_normalization)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{PlaneWaveBasis}, model::Model, Ecut; kwargs...)
    @warn "simplified PlaneWaveBasis rrule triggered."
    basis = T(model, Ecut; kwargs...)
    _basis, PlaneWaveBasis_pullback = rrule_via_ad(config, _autodiff_PlaneWaveBasis_namedtuple, model, basis)
    return basis, PlaneWaveBasis_pullback
end


# TODO delete (once fixed upstream in Zygote, "difftype_warn not defined")
Zygote.z2d(x::Union{AbstractZero, Tangent}, ::Any) = x

# convert generators into arrays (needed for Zygote here)
function _G_vectors_cart(basis::PlaneWaveBasis)
    [basis.model.recip_lattice * G for G in G_vectors(basis.fft_size)]
end
_G_vectors_cart(kpt::Kpoint) = [kpt.model.recip_lattice * G for G in G_vectors(kpt)]

function _autodiff_TermKinetic_namedtuple(basis; scaling_factor=1)
    kinetic_energies = [[scaling_factor * sum(abs2, G + kpt.coordinate_cart) / 2
                         for G in _G_vectors_cart(kpt)]
                        for kpt in basis.kpoints]
    (;basis=basis, kinetic_energies=kinetic_energies)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{TermKinetic}, basis::PlaneWaveBasis; kwargs...)
    @warn "simplified TermKinetic rrule triggered."
    term = T(basis; kwargs...)
    T_simple = (args...) -> _autodiff_TermKinetic_namedtuple(args...; kwargs...)
    _term, TermKinetic_pullback = rrule_via_ad(config, T_simple, basis)
    return term, TermKinetic_pullback
end

Base.zero(::ElementPsp) = ZeroTangent() # TODO 

function ChainRulesCore.rrule(T::Type{TermAtomicLocal}, basis, potential)
    TermAtomicLocal_pullback(Δ) = NoTangent(), Δ.basis, Δ.potential
    return T(basis, potential), TermAtomicLocal_pullback
end

function _autodiff_AtomicLocal(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model

    pot_fourier = map(G_vectors(basis)) do G
        pot = zero(T)
        for (elem, positions) in model.atoms
            form_factor::T = local_potential_fourier(elem, norm(model.recip_lattice * G))
            for r in positions
                pot += cis(-2T(π) * dot(G, r)) * form_factor
            end
        end
        pot / sqrt(model.unit_cell_volume)
    end

    pot_real = G_to_r(basis, pot_fourier)
    TermAtomicLocal(basis, real(pot_real))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, E::AtomicLocal, basis::PlaneWaveBasis{T}) where {T}
    @warn "simplified AtomicLocal rrule triggered."
    term = E(basis)
    _term, AtomicLocal_pullback = rrule_via_ad(config, _autodiff_AtomicLocal, basis)
    return term, AtomicLocal_pullback
end

