"""Silicon area and IO perimeter estimation for NPU components.

All constants are calibrated for TSMC N5 (5nm), the approximate process
node for TPU v5p/v6p-class designs. Results for other nodes scale
proportionally to the node area-scaling factor (~0.5x per generation).

Area sources:
- SA: Jouppi et al. "In-Datacenter Performance Analysis of a Tensor Processing Unit"
  (ISCA 2017) -- 256x256 MXU ~83 mm2 at 28nm -> ~8 mm2 at N5 (10x node scaling);
  x2 for local column FIFOs and activation pipeline registers -> 0.0005 mm2/MAC.
- Routing overhead: 25% of functional area added for power delivery network,
  global clock distribution, pad ring, and control/scheduling logic.
- SRAM: TSMC N5 6T cell 0.021 um2/bit -> 0.176 mm2/MB (raw cells);
  x1.4 for sense amps, decoders, ECC -> ~0.25 mm2/MB.
- ICI SerDes: 112 Gbps lane at N5 ~0.4 mm2 (Cadence 112G ELR IP);
  0.4 / (112 Gbps / 8 bits) = 0.029 mm2/GBps -> rounded to 0.03.
- HBM MC: digital memory controller only -- the HBM PHY (SerDes, equalization,
  PLLs, termination) lives on the HBM base/logic die, not on the accelerator die.
  HBM3 digital MC ~5 mm2 per stack at N5, ~925 GB/s per stack ->
  0.0054 mm2/GBps -> rounded to 0.005.
- VU: estimated at ~10-20% of a comparably-sized SA; for 8x128 SIMD
  lanes (BF16 FMA), ~0.004 mm2/lane gives ~4 mm2 per VU.

Perimeter sources:
- HBM microbump pitch: 55 um (HBM3e JEDEC spec); 1024 data pins per stack
  arranged in 4 rows along the die edge -> 14.1 mm of edge per stack.
  HBM3e: 12.8 Gbps/pin = 1.6 GB/s/pin -> 1638 GB/s per stack.
- ICI SerDes: 112G bidirectional lane uses 2 differential pairs (TX + RX)
  = 4 bumps at 130 um C4 pitch = 0.52 mm per 14 GB/s lane.
- Aspect ratio: 2:1 (width:height) is the best-case rectangle that remains
  fabricable within a single reticle field (~26 mm x 33 mm at N5).

Cost sources:
- Die: TSMC N5 wafer ~$17,000 / ~70,686 mm2 usable area -> ~$0.24/mm2 gross;
  with typical packaging and test overhead -> ~$0.35/mm2 effective.
- HBM: ~$10/GB for HBM3/HBM3e modules (Yole, 2024 market data).
- ICI: short-reach 112G optical transceivers ~$50/link at 14 GB/s per link
  -> ~$3.50/GB/s for the ICI fabric.
"""

import math
from dataclasses import dataclass

from neusim.configs.chips.ChipConfig import ChipConfig

# ── Area constants (TSMC N5 reference) ─────────────────────────────────────

# mm2 per MAC cell in a systolic array.
# Covers compute cells, local column FIFOs, and activation pipeline registers.
# Large weight caches (CMEM/SPMEM) are captured by _K_UNMODELED_MM2 instead.
# Basis: TPU v1 MXU ~83 mm2 at 28nm -> ~8 mm2 at N5; x2 for local staging -> 0.0005.
_K_SA_MM2_PER_MAC: float = 0.0005

# mm2 per BF16 FMA SIMD lane in a vector unit.
# Each VU has _VU_SIMD_LANES lanes (architecture constant, matches power_model.py).
_K_VU_MM2_PER_LANE: float = 0.004

# mm2 per MB of on-chip SRAM (VMEM).
# N5 raw cell area + peripheral overhead (sense amps, decoders, ECC).
_K_SRAM_MM2_PER_MB: float = 0.25

# mm2 per GB/s of ICI bandwidth (SerDes PHY, both TX and RX).
_K_ICI_MM2_PER_GBPS: float = 0.03

# mm2 per GB/s of HBM bandwidth (digital memory controller only).
# The HBM PHY resides on the HBM base/logic die, not the accelerator die.
# Basis: HBM3 digital MC ~5 mm2/stack at N5, ~925 GB/s/stack -> 0.0054 mm2/GBps.
_K_HBM_MM2_PER_GBPS: float = 0.005

# SIMD lanes per VU -- architecture constant (8 banks x 128 elements).
_VU_SIMD_LANES: int = 8 * 128  # 1024

# Routing overhead multiplier applied to the sum of all functional components.
# Covers power delivery network (PDN) stripes (~5-8%), global clock tree (~2-4%),
# and signal routing whitespace (~5-8%). 1.15 = 15% overhead on top of functional area.
_ROUTING_OVERHEAD_FACTOR: float = 1.15

# Fixed control logic area (mm2) present on every chip regardless of compute config.
# Covers PCIe controller, power management unit, fuse arrays, JTAG/debug logic,
# thermal sensors, chip-level reset/boot circuitry, and ICI router digital logic.
# Estimate: ~55 mm2 at N5 for these fixed blocks.
_K_CTRL_MM2: float = 55.0

# Fixed area (mm2) for microarchitectural features not captured by the performance
# model: weight/activation caches (CMEM, SPMEM), sparse cores, and similar blocks
# present in current-generation TPUs. Estimated at ~100 mm2 at N5.
_K_UNMODELED_MM2: float = 125.0

# ── Perimeter/IO constants ──────────────────────────────────────────────────

# HBM microbump geometry (HBM3e class).
# 1024 data pins per stack arranged in 4 rows along the die edge facing each stack.
# Edge consumed per stack: (1024 / 4) * 0.055 mm = 14.08 mm.
# Bandwidth per stack: 1024 pins * 1.6 GB/s/pin = 1638.4 GB/s.
_HBM_BUMP_PITCH_MM: float = 0.055
_HBM_ROWS_PER_STACK: int = 4
_HBM_PINS_PER_STACK: int = 1024
_HBM_GBPS_PER_PIN: float = 1.6      # 12.8 Gbps/pin (HBM3e) / 8 bits

# ICI SerDes geometry (112G bidirectional lane, C4 bumps).
# Each lane: 2 differential pairs (TX + RX) = 4 bumps at 130 um pitch.
_ICI_BUMP_PITCH_MM: float = 0.130
_ICI_BUMPS_PER_LANE: int = 4
_ICI_GBPS_PER_LANE: float = 14.0    # 112 Gbps / 8 bits

# Maximum aspect ratio (width:height) considered fabricable within one reticle field.
# 2:1 is a conservative upper bound; it maximises perimeter for a given area.
_MAX_ASPECT_RATIO: float = 2.0

# Fraction of the die perimeter available for HBM + ICI bumps.
# The remaining half is reserved for power delivery (VDD/VSS C4 bumps),
# PCIe, and other chip-level IO.
_IO_PERIMETER_FRACTION: float = 0.7


# ── Result types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AreaBreakdownMM2:
    """Per-component silicon area estimate in mm2."""
    sa_mm2: float        # all systolic arrays combined (compute + weight staging)
    vu_mm2: float        # all vector units combined
    vmem_mm2: float      # on-chip SRAM (VMEM)
    ici_mm2: float       # ICI serializer/deserializer PHY
    hbm_mc_mm2: float    # HBM digital memory controllers (PHY is on HBM base die)
    routing_mm2: float     # PDN, clock distribution, pad ring (25% of functional area)
    ctrl_mm2: float        # fixed control logic (PCIe, PMU, JTAG, ICI router, etc.)
    unmodeled_mm2: float   # unmodeled microarch features (CMEM, SPMEM, sparse cores)

    @property
    def total_mm2(self) -> float:
        return (self.sa_mm2 + self.vu_mm2 + self.vmem_mm2 + self.ici_mm2
                + self.hbm_mc_mm2 + self.routing_mm2 + self.ctrl_mm2 + self.unmodeled_mm2)


# ── Public API ───────────────────────────────────────────────────────────────

def estimate_area(chip: ChipConfig) -> AreaBreakdownMM2:
    """Return a per-component silicon area estimate for *chip*."""
    sa_mm2 = chip.num_sa * _K_SA_MM2_PER_MAC * (chip.sa_dim ** 2)
    vu_mm2 = chip.num_vu * _K_VU_MM2_PER_LANE * _VU_SIMD_LANES
    vmem_mm2 = _K_SRAM_MM2_PER_MB * chip.vmem_size_MB
    ici_mm2 = _K_ICI_MM2_PER_GBPS * chip.ici_bw_GBps
    hbm_mc_mm2 = _K_HBM_MM2_PER_GBPS * chip.hbm_bw_GBps

    functional_mm2 = sa_mm2 + vu_mm2 + vmem_mm2 + ici_mm2 + hbm_mc_mm2
    routing_mm2 = functional_mm2 * (_ROUTING_OVERHEAD_FACTOR - 1.0)

    return AreaBreakdownMM2(
        sa_mm2=sa_mm2,
        vu_mm2=vu_mm2,
        vmem_mm2=vmem_mm2,
        ici_mm2=ici_mm2,
        hbm_mc_mm2=hbm_mc_mm2,
        routing_mm2=routing_mm2,
        ctrl_mm2=_K_CTRL_MM2,
        unmodeled_mm2=_K_UNMODELED_MM2,
    )


@dataclass(frozen=True)
class PerimeterBudget:
    """IO perimeter feasibility check for HBM and ICI connections."""
    die_perimeter_mm: float   # total estimated die perimeter (2:1 aspect ratio best case)
    hbm_edge_mm: float        # die-edge length consumed by HBM microbump arrays
    ici_edge_mm: float        # die-edge length consumed by ICI SerDes bumps

    @property
    def usable_perimeter_mm(self) -> float:
        """Perimeter available for HBM + ICI; the other half is reserved for
        power delivery bumps, PCIe, and other chip-level IO."""
        return self.die_perimeter_mm * _IO_PERIMETER_FRACTION

    @property
    def total_used_mm(self) -> float:
        return self.hbm_edge_mm + self.ici_edge_mm

    @property
    def utilization(self) -> float:
        """Fraction of usable perimeter consumed (>1.0 means infeasible)."""
        return self.total_used_mm / self.usable_perimeter_mm

    @property
    def is_feasible(self) -> bool:
        return self.total_used_mm <= self.usable_perimeter_mm


def estimate_perimeter_usage(chip: ChipConfig, area: AreaBreakdownMM2) -> PerimeterBudget:
    """Check whether HBM + ICI bandwidth fits on the perimeter of *chip*'s die.

    Models the die as a 2:1 rectangle (width:height), which maximises perimeter
    for a given area while remaining within a single reticle field. Connections
    are treated as edge-limited:

    - HBM stacks sit adjacent to the die on a CoWoS interposer; their microbump
      arrays occupy a strip along the die edges facing each stack.
    - ICI SerDes lanes exit the die via C4 bumps arranged along the perimeter.

    The routing_mm2 component in AreaBreakdownMM2 already accounts for PDN,
    clocks, and pad ring, so total_mm2 is a reasonable estimate of true die area.
    """
    # Perimeter of a 2:1 rectangle with area A:
    #   W = sqrt(2A), H = sqrt(A/2)  ->  P = 2(W + H) = 2*sqrt(A)*(sqrt(2) + 1/sqrt(2))
    A = area.total_mm2
    die_perimeter = 2 * (math.sqrt(_MAX_ASPECT_RATIO * A) + math.sqrt(A / _MAX_ASPECT_RATIO))

    # HBM: bandwidth -> number of stacks -> edge length
    hbm_bw_per_stack = _HBM_PINS_PER_STACK * _HBM_GBPS_PER_PIN          # GB/s per stack
    hbm_edge_per_stack = (_HBM_PINS_PER_STACK / _HBM_ROWS_PER_STACK) * _HBM_BUMP_PITCH_MM  # mm per stack
    hbm_edge_mm = (chip.hbm_bw_GBps / hbm_bw_per_stack) * hbm_edge_per_stack

    # ICI: bandwidth -> lanes -> bumps -> edge length
    ici_lanes = chip.ici_bw_GBps / _ICI_GBPS_PER_LANE
    ici_edge_mm = ici_lanes * _ICI_BUMPS_PER_LANE * _ICI_BUMP_PITCH_MM

    return PerimeterBudget(
        die_perimeter_mm=die_perimeter,
        hbm_edge_mm=hbm_edge_mm,
        ici_edge_mm=ici_edge_mm,
    )

# ── Cost constants ──────────────────────────────────────────────────────────

# USD per mm2 of logic die area at TSMC N5.
# Basis: ~$17,000/wafer / ~70,686 mm2 = $0.24/mm2 gross silicon cost;
# packaging, test, and overhead bring effective cost to ~$0.35/mm2.
_K_DIE_USD_PER_MM2: float = 0.35

# USD per GB of HBM memory capacity.
# Basis: HBM3/HBM3e spot pricing ~$10/GB (Yole Developpement, 2024).
_K_HBM_USD_PER_GB: float = 10.0

# USD per GB/s of ICI bandwidth.
# Basis: short-reach 112G optical transceiver ~$50 per link at 14 GB/s
# -> $50 / 14 = $3.57/GB/s, rounded to $3.50.
_K_ICI_USD_PER_GBPS: float = 3.50


@dataclass(frozen=True)
class CostBreakdownUSD:
    """Per-component chip cost estimate in USD."""
    die_usd: float   # logic die fabrication cost
    hbm_usd: float   # HBM memory module cost
    ici_usd: float   # ICI link hardware cost (transceivers + cables)

    @property
    def total_usd(self) -> float:
        return self.die_usd + self.hbm_usd + self.ici_usd


def estimate_cost(chip: ChipConfig, area: AreaBreakdownMM2) -> CostBreakdownUSD:
    """Return a per-component cost estimate for *chip*.

    Three cost drivers:
    - Logic die: scales with total die area including routing overhead.
    - HBM modules: scales with total HBM capacity (hbm_size_GB).
    - ICI fabric: scales with ICI bandwidth (transceivers and cables).
    """
    die_usd = area.total_mm2 * _K_DIE_USD_PER_MM2
    hbm_usd = chip.hbm_size_GB * _K_HBM_USD_PER_GB
    ici_usd = chip.ici_bw_GBps * _K_ICI_USD_PER_GBPS

    return CostBreakdownUSD(
        die_usd=die_usd,
        hbm_usd=hbm_usd,
        ici_usd=ici_usd,
    )
