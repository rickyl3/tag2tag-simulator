from __future__ import annotations
from math import sqrt, log10
from typing import TYPE_CHECKING, Iterable
import numpy as np

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
        default_power_on_dbm: float = -100.0,
        noise_std_volts: float = 0,
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

        # Tag topology caching
        self._cached_state = {
            "tag_names": None,
            "H": None,
            "Gamma": None,
            "h_exciter": None,
            "S": None,
            "hash": None,
        }

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
        return att * cmath.exp(1j * 2 * pi * distance / wavelen)

    def power_from_exciters_at_tag_mw(self, tag: Tag) -> float:
        """
        Gets the power (in mW) delivered from the engine's exciters to the tag antenna input using Friis transmission formula

        Parameters:
            tag (Tag): The receiving tag.
        Returns:
            float: The power (in mW) delivered to the tag.
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
            break 
        return power_rxs

    def is_tag_powered(self, tag: Tag) -> bool:
        """
        Determines whether a tag has sufficient harvested power to run logic ("listening" capability).

        Parameters:
            tag (Tag): The tag to check.
        Returns:
            bool: True if the tag is powered, False otherwise.
        """
        power_tag_mw = self.power_from_exciters_at_tag_mw(tag)
        power_tag_dbm = mW_to_dBm(power_tag_mw)
        threshold_dbm = getattr(
            tag, "power_on_threshold_dbm", self.default_power_on_dbm
        )
        return power_tag_dbm >= threshold_dbm

    def effective_reflection_coefficient(self, tag: Tag) -> complex:
        """
        Returns the complex reflection coefficient used when the tag is contributing to the channel.
        Rules implemented:
            - If tag is not powered -> return a very small passive reflection
            - If tag.mode.is_listening() -> return a small unmodulated reflection
            - If tag is transmitting (mode != listening) -> return the reflection coefficient based on impedances

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

    def _compute_state_hash(self, tags):
        """
        Compute a hash representing the current state of the tags (positions and impedances).

        Parameters:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
        Returns:
            int: A hash value representing the current state.
        """
        data = []
        for tag in tags.values():
            pos = tag.get_position()
            z_chip = tag.get_chip_impedance()
            z_ant = tag.get_impedance()
            data.extend([*pos, complex(z_chip), complex(z_ant)])
        return hash(tuple(round(float(x.real if isinstance(x, complex) else x), 8) for x in data))


    def voltage_at_tag(self, tags: dict[str, Tag], receiving_tag: Tag) -> float:
        """
        Get's the total voltage delivered to a given tag by the rest of the tags.
        This includes feedback loops in the backscatter.

        Parameters:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
            receiving_tag (Tag): The tag to get the voltage for.
        Returns:
            float: The voltage at the receiving tag's envelope detector input.
        """
        rx_impedance = receiving_tag.get_impedance()

        # --- Collect all tag names ---
        tag_names = list(tags.keys())
        if receiving_tag.get_name() not in tag_names:
            tag_names.append(receiving_tag.get_name())
        n = len(tag_names)
        if n == 0:
            return 0.0

        # --- Compute current simulation state hash ---
        current_hash = self._compute_state_hash(tags)

        # --- Rebuild matrices only if topology or reflection coefficients changed ---
        if (
            self._cached_state["hash"] != current_hash
            or self._cached_state["tag_names"] != tag_names
        ):
            # --- Build channel matrix H ---
            H = np.zeros((n, n), dtype=np.complex128)
            for i, name_i in enumerate(tag_names):
                tag_i = tags[name_i] if name_i in tags else receiving_tag
                for j, name_j in enumerate(tag_names):
                    if i == j:
                        continue
                    tag_j = tags[name_j] if name_j in tags else receiving_tag
                    H[i, j] = self.get_sig_tx_rx(tag_j, tag_i)

            # --- Build reflection coefficients Γ ---
            gammas = np.zeros(n, dtype=np.complex128)
            for j, name_j in enumerate(tag_names):
                tag_j = tags[name_j] if name_j in tags else receiving_tag
                gammas[j] = self.effective_reflection_coefficient(tag_j)
            Gamma = np.diag(gammas)

            # --- Compute & cache inverse (I - HΓ)^(-1) ---
            I = np.eye(n, dtype=np.complex128)
            A = I - H @ Gamma
            try:
                S_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(A)

            # --- Update cache ---
            self._cached_state.update({
                "hash": current_hash,
                "tag_names": tag_names,
                "H": H,
                "Gamma": Gamma,
                "S_inv": S_inv,
            })
        else:
            # --- Reuse cached matrices ---
            H = self._cached_state["H"]
            Gamma = self._cached_state["Gamma"]
            S_inv = self._cached_state["S_inv"]

        # --- Build exciter contribution vector (sum over all exciters) ---
        h_exciter = np.zeros(n, dtype=np.complex128)
        for i, name_i in enumerate(tag_names):
            tag_i = tags[name_i] if name_i in tags else receiving_tag
            total_field = sum(self.get_sig_tx_rx(ex, tag_i) for ex in self.exciters.values())
            h_exciter[i] = total_field

        # --- Solve S = (I - HΓ)^(-1) * h_exciter using cached inverse ---
        S = S_inv @ h_exciter

        # --- Extract field at receiving tag ---
        rx_field = S[tag_names.index(receiving_tag.get_name())]
        pwr_received = abs(rx_field)

        # --- Convert to voltage ---
        v_pk = sqrt(abs(rx_impedance * pwr_received) / 500.0)
        v_rms = v_pk / sqrt(2.0)

        # --- Add optional Gaussian noise ---
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
            tx_indices (Iterable[int] | None): Optional pair of indices to use for the tx tag.
        Returns:
            float: The modulation depth (absolute voltage difference) at the rx tag.
        """
        # Choose indices if not provided (avoid listening idx=0)
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