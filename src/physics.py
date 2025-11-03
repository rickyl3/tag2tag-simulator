from __future__ import annotations
from math import sqrt, log10
from typing import TYPE_CHECKING, Iterable

import scipy.spatial.distance as dist
from scipy.constants import c, pi
import random
import cmath


from tags.tag import TagMode

if TYPE_CHECKING:
    from tags.tag import Exciter, PhysicsObject, Tag


def mW_to_dBm(mw: float) -> float:
    if mw <= 0:
        return -999.0
    return 10.0 * log10(mw)


def dBm_to_mW(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def dbi_to_linear(dbi: float) -> float:
    """Convert gain in dBi to linear scale."""
    return 10 ** (dbi / 10.0)


class PhysicsEngine:
    def __init__(
        self,
        exciters: dict[str, Exciter],
        default_power_on_dbm: float = -100.0,  # TODO make this configurable per-tag and find a good default
        noise_std_volts: float = 0,  # 0.0001 is 0.1 mV noise,
        passive_ref_mag: float = 0,
    ):
        """
        Initialize the physics engine.

        Parameters:
            exciters (dict[str, Exciter]): The dictionary of exciter objects.
            default_power_on_dbm (float): The default power threshold (in dBm) for a tag to be considered "powered".
            noise_std_volts (float): Standard deviation of Gaussian noise (in volts) added to envelope-detector output.
                                   Default is 0 (no noise).
        """
        self.exciters = exciters
        self.default_power_on_dbm = default_power_on_dbm
        self.noise_std_volts = noise_std_volts
        self.passive_ref_mag = passive_ref_mag

    def attenuation(
        self, distance: float, wavelength: float, tx_gain_dbi=1.0, rx_gain_dbi=1.0
    ) -> float:
        """
        Helper function for calculating the attenuation between two antennas

        Parameters:
            distance (float): Distance between the two antennas in meters.
            wavelength (float): Wavelength in meters.
            tx_gain (float): Transmitting antenna gain in dBi.
            rx_gain (float): Receiving antenna gain in dBi.
        Returns:
            float: Power ratio (unitless) between transmitted and received power.

        # TODO Add Radiating near-field model (1/d^3) for distances < wavelength/(2*pi) (NOT HIGH PRIORITY)
        """
        if distance <= 0:
            return 0.0

        tx_gain = dbi_to_linear(tx_gain_dbi)
        rx_gain = dbi_to_linear(rx_gain_dbi)

        reactive_limit = wavelength / (2 * pi)

        if distance < reactive_limit:
            # Near-field region (reactive near-field, use approximate 1/d^3 model)
            return (
                (tx_gain * rx_gain * (wavelength**2))
                / ((4 * pi * reactive_limit) ** 2)
                * (reactive_limit / distance) ** 3
            )
        else:
            # Far-field region (Friis transmission equation, 1/d^2 model)
            num = tx_gain * rx_gain * (wavelength**2)
            den = (4 * pi * distance) ** 2
            return num / den

    def get_sig_tx_rx(self, tx: PhysicsObject, rx: Tag):
        """
        Gets the signal from a tag or an exciter to another tag

        Parameters:
            tx (PhysicsObject): The transmitting object.
            rx (Tag): The receiving tag.
        Returns:
            complex: A complex phasor representing the contribution from tx -> rx.
        """
        distance = dist.euclidean(tx.get_position(), rx.get_position())
        wavelen = c / tx.get_frequency()
        att = sqrt(self.attenuation(distance, wavelen, tx.get_gain(), rx.get_gain()))
        return att * (
            cmath.exp(1j * 2 * pi * distance / wavelen)
        )  # Use Cmath for e not e from scipy

    def power_from_exciters_at_tag_mw(self, tag: Tag) -> float:
        """
        Gets the power (in mW) delivered from the engine's exciters to the tag antenna input using Friis transmission formula

        Parameters:
            tag (Tag): The receiving tag.
        Returns:
            float: The power (in mW) delivered to the tag.

        Assumptions:
            - exciter.get_power() returns transmit power in mW
            - gains are linear directivities (not dBi). If gain is provided in dBi, convert before using.

        # TODO Check if gains are in linear directivities or dBi(if DBI, convert to linear)
        """
        exs = self.exciters
        power_rxs = 0.0
        for ex in exs.values():
            power_tx_mw = ex.get_power()
            if power_tx_mw <= 0:
                continue

            distance = dist.euclidean(ex.get_position(), tag.get_position())
            wavelength = c / ex.get_frequency()

            power_rx = power_tx_mw * self.attenuation(
                distance, wavelength, ex.get_gain(), tag.get_gain()
            )
            power_rxs += max(power_rx, 0.0)
        return power_rxs

    def is_tag_powered(self, tag: Tag) -> bool:
        """
        Determines whether a tag has sufficient harvested power to run logic ("listening" capability).

        Parameters:
            tag (Tag): The tag to check.
        Returns:
            bool: True if the tag is powered, False otherwise.

        Checks for:
            - per-tag attribute `power_on_threshold_dbm` (if present)
            - otherwise uses engine.default_power_on_dbm
        """
        power_tag_mw = self.power_from_exciters_at_tag_mw(tag)
        power_tag_dbm = mW_to_dBm(power_tag_mw)
        threshold_dbm = getattr(
            tag, "power_on_threshold_dbm", self.default_power_on_dbm
        )  # TODO Add power_on_threshold_dbm to Tag class
        return power_tag_dbm >= threshold_dbm

    def effective_reflection_coefficient(self, tag: Tag) -> complex:
        """
        Returns the complex reflection coefficient used when the tag is contributing to the channel.
        Rules implemented:
            - If tag is not powered -> return a very small passive reflection (near-zero complex) to represent the tag's metal scatter but not a powered, modulated reflection.
            - If tag.mode.is_listening() -> return a small unmodulated reflection (the envelope detector input typically presents an absorbing load; we model a low baseline reflection).
            - If tag is transmitting (mode != listening) -> return the reflection coefficient based on the antenna and current chip impedances.

        Parameters:
            tag (Tag): The tag to get the reflection coefficient for.
        Returns:
            complex: The effective reflection coefficient.
        """

        PASSIVE_REF = complex(self.passive_ref_mag, 0.0)

        if not self.is_tag_powered(tag) or tag.get_mode().is_listening():
            return PASSIVE_REF

        # Otherwise tag is actively reflecting (transmit index)
        Z_ant = tag.get_impedance()
        Z_chip = tag.get_chip_impedance()

        try:
            gamma = (Z_chip - Z_ant.conjugate()) / (Z_chip + Z_ant)
        except ZeroDivisionError:
            gamma = complex(0.0, 0.0)

        return gamma

    def voltage_at_tag(
        self, tags: dict[str, Tag], receiving_tag: Tag, include_helpers: bool = True
    ) -> float:
        """
        Get's the total voltage delivered to a given tag by the rest of the
        tags. This currently makes the assumption that there are no feedback
        loops in the backscatter for simplicity.
        # TODO Look into feedback loops
        # TODO Look into include_helpers parameter

        Parameters:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
            receiving_tag (Tag): The tag to get the voltage for.
            include_helpers (bool): Whether to include helper tags in the calculation.
        Returns:
            float: The voltage at the receiving tag's envelope detector input.
        """
        exs = self.exciters
        rx_impedance = receiving_tag.get_impedance()

        # This will be summed later
        sigs_to_rx = []
        for ex in exs.values():
            sigs_to_rx.append(self.get_sig_tx_rx(ex, receiving_tag))
            for tag in tags.values():
                if tag is receiving_tag:
                    continue

                reflection_coeff = self.effective_reflection_coefficient(tag)
                if abs(reflection_coeff) < 1e-6:
                    continue

                sig_ex_tx = self.get_sig_tx_rx(ex, tag)
                sig_tx_rx = self.get_sig_tx_rx(tag, receiving_tag)
                sigs_to_rx.append(sig_ex_tx * reflection_coeff * sig_tx_rx)

        pwr_received = abs(sum(sigs_to_rx))
        v_pk = sqrt(abs(rx_impedance * pwr_received) / 500.0)
        v_rms = v_pk / sqrt(2.0)

        # Add optional AWGN noise (applied to the RMS read-out)
        if self.noise_std_volts and self.noise_std_volts > 0.0:
            v_rms = max(0.0, random.gauss(v_rms, self.noise_std_volts))

        return v_rms

    def modulation_depth_for_tx_rx(
        self,
        tags: dict[str, Tag],
        tx: Tag,
        rx: Tag,
        tx_indices: Iterable[int] | None = None,
    ) -> float:
        """
        Compute modulation depth metric for a specific (tx, rx) pair.

        Parameters:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
            tx (Tag): The transmitting tag.
            rx (Tag): The receiving tag.
            tx_indices (Iterable[int] | None): Optional pair of indices to use for the tx tag. If None, will use [0, 1] if possible.
        Returns:
            float: The modulation depth (absolute voltage difference) at the rx tag when tx switches between the two specified modes.
        """
        # Choose indicies if not provided (avoid listening idx=0)
        if tx_indices is None:
            num_tx = len(tx.chip_impedances)
            if num_tx >= 3:
                idx0, idx1 = 1, 2
            elif num_tx >= 2:
                idx0, idx1 = 0, 1
            else:
                idx0, idx1 = 0, 0
        else:
            idx0, idx1 = tuple(tx_indices)

        original_mode = tx.get_mode()

        # Get voltage when tx is in state at index idx0
        tx.set_mode(TagMode(idx0))
        v0 = self.voltage_at_tag(tags, rx)

        # Get voltage when tx is in state at index idx1
        tx.set_mode(TagMode(idx1))
        v1 = self.voltage_at_tag(tags, rx)

        # Restore original mode
        tx.set_mode(original_mode)

        return abs(v1 - v0)

    def phase_ang_and_diff(self, tx: Tag, rx: Tag, pre_rx: Tag) -> []:
	    frequency = tx.get_frequency()
	    lambda_freq = c / frequency

	    dx = tx.pos[0] - rx.pos[0]
	    dy = tx.pos[1] - rx.pos[1]
	    dz = tx.pos[2] - rx.pos[2]
	    distance = ((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5
	    phase_angle = (2 * pi * distance) / lambda_freq

	    prev_dx = tx.pos[0] - pre_rx.pos[0]
	    prev_dy = tx.pos[1] - pre_rx.pos[1]
	    prev_dz = tx.pos[2] - pre_rx.pos[2]
	    prev_distance = ((prev_dx ** 2) + (prev_dy ** 2) + (prev_dz ** 2)) ** 0.5
	    phase_difference = (2 * pi * (distance - prev_distance)) / lambda_freq

	    # TODO: Do the calculations here and whatnot: SciPy may have you covered!



	    return phase_angle, phase_difference




