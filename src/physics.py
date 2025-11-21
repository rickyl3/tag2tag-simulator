from __future__ import annotations
from math import sqrt, log10
from typing import TYPE_CHECKING, Iterable, Any

import scipy.spatial.distance as dist
from numpy.random import default_rng
from scipy.constants import c, pi
from scipy.optimize import least_squares
import random
import cmath

from numpy.random import default_rng
import numpy as np

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

    def doppler_effect(self, tags: dict[str, Tag], tx: Tag, rx: Tag) -> tuple[float, float, float]:
        """
        Calculates the phase angle, phase difference,
        and perceived frequency with the doppler effect between a sender and a tag.

        Args:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
            tx (Tag): The sender.
            rx (Tag): The receiver.

        Returns:
            phase_angle (float): The phase angle. Mainly for graphing.
            phase_difference (float): The phase difference.
            doppler_freq (float): The perceived frequency with the doppler effect.
        """
        frequency = tx.get_frequency()
        lambda_freq = c / frequency
        delta_t = 0.0333333333333333333333333333333
        # TODO: This is a placeholder for the difference of time.
        #  The line of receivers that simulate movement have no concept of time,
        #  so this "0.0333..." indicating 30 samples per second is a placeholder.

        phase_angle = self.phase_ang(tx, rx)
        phase_difference = self.phase_diff(tags, tx, rx)

        delta_distance = (lambda_freq * phase_difference) / ((2 * pi) * delta_t)
        # TODO: delta_distance seems rather roundabout to just saying something like "distance - prev_distance".
        #  Is there some way to measure phase difference directly without the positions being a given?

        # doppler_freq is the frequency measured at the receiver with the doppler effect in mind.
        # TODO: This is the non-relativistic Doppler effect. Did we mean the relativistic Doppler effect?
        try:
            doppler_freq = frequency * ((c - delta_distance) / (c + 0))
        except ZeroDivisionError:
            doppler_freq = float("inf")
        return phase_angle, phase_difference, doppler_freq

    def phase_ang(self, tx: Tag, rx: Tag) -> float:
        """
        Calculates the phase angle between a sender and a tag.

        Args:
            tx (Tag): The sender.
            rx (Tag): The receiver.

        Returns:
            phase_angle (float): The phase angle.
        """
        frequency = tx.get_frequency()
        lambda_freq = c / frequency
        distance = dist.euclidean(tx.get_position(), rx.get_position())
        phase_angle = (2 * pi * distance) / lambda_freq
        return phase_angle


    def phase_diff(self, tags: dict[str, Tag], tx: Tag, rx: Tag) -> float:
        """
        Calculates the phase angle between a sender and a tag.

        Args:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
                                   Used to reference the receiver's most recent location.
            tx (Tag): The sender.
            rx (Tag): The receiver.

        Returns:
            phase_difference (float): The phase angle.
        """
        pre_rx = rx
        tagz = list(tags.values())
        for x in range(1, len(tagz)):
            if tagz[x] is rx:
                if tagz[x - 1].tag_machine.input_machine.state.name == "EMPTY":
                    break  # No senders listed as a receiver, please!
                pre_rx = tagz[x - 1]
                break

        frequency = tx.get_frequency()
        lambda_freq = c / frequency
        delta_t = 0.0333333333333333333333 # TODO: This, too, is a placeholder. See above.
        distance = dist.euclidean(tx.get_position(), rx.get_position())
        prev_distance = dist.euclidean(tx.get_position(), pre_rx.get_position())
        phase_difference = ((2 * pi) * (distance - prev_distance)) / (lambda_freq * delta_t)
        return phase_difference

    def estimate_rx_velocity(self, tags: dict[str, Tag], rx: Tag) -> tuple[float, float, float]:
        """
        Calculates the estimated receiver velocity using a methods described in
        https://dl.acm.org/doi/pdf/10.1145/2639108.2639111 .
        Limited to 2-dimensions where positions along the Z-axis are ideally ignored,
        but it sounds possible to implement down the line, maybe?

        Args:
            tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
            tx (Tag): The sender.
            rx (Tag): The receiver.

        Returns:
            v_n_spd (float): The estimated receiver velocity speed.
            v_n_dir (float): The estimated receiver velocity angle along the x-axis.
        """
        tagz = list(tags.values())

        next_rx = rx
        tagz = list(tags.values())
        for x in range(0, len(tagz)):
            if x == len(tagz) - 1:
                return 0, 0, 0 # Skips processing the last receiver, as that is the destination point.
            if tagz[x] is rx:
                next_rx = tagz[x + 1]
                break

        senders = []
        v_mn_array = []
        dir_v_mn_array = []
        v_mn_xy = []
        tx = None
        for tag in tagz:
            if tag.tag_machine.input_machine.state.name == "EMPTY":
                senders.append(tag)
                # TODO: Keep note of the receiver's "starting location" for end-game trajectory fitting.
                tx = tag
                frequency = tx.get_frequency()
                lambda_freq = c / frequency

                theta_n = self.phase_ang(tx, rx)
                theta_next_n = self.phase_ang(tx, next_rx)
                theta_adjacent = theta_next_n - theta_n

                # As one can infer, these "sanity checks" are placeholders.
                # delta_d_sanitycheck1 = dist.euclidean(tx.get_position(), next_rx.get_position()) - dist.euclidean(tx.get_position(), rx.get_position())
                # delta_d_sanitycheck2 = (lambda_freq * self.phase_diff(tags, tx, next_rx)) / ((2 * pi) * 1)
                delta_d = 0
                if abs(theta_adjacent) < pi:
                    delta_d = theta_next_n - theta_n
                elif theta_adjacent >= pi:
                    delta_d = (2 * pi) - theta_n + theta_next_n
                else:
                    delta_d = theta_next_n - theta_n - (2 * pi)
                delta_d /= (4 * pi)
                delta_d *= lambda_freq
                delta_d *= 2 # For some reason, this ends up being only half of the actual distance displacement...

                delta_t = 0.0333333333333333333333  # TODO: Wow! A placeholder!
                v_mn = delta_d / delta_t
                dir_v_mn = self.angle_of_radical_instant_speed(tx, rx)
                if delta_d < 0:
                    delta_d = -delta_d
                    dir_v_mn = ((dir_v_mn + 180) % 360)
                    v_mn = -v_mn
                dir_v_mn = ((dir_v_mn + 180) % 360) - 180

                v_mn_array.append(v_mn)
                dir_v_mn_array.append(dir_v_mn)
                v_mn_xy.append([v_mn * cmath.cos(dir_v_mn / (180 / pi)).real, v_mn * cmath.sin(dir_v_mn / (180 / pi)).real])

        # super_v_mn_array = []
        # for x in dir_v_mn_array:
        #     super_v_mn_array.append([y * v_mn for y in x])

        # rng = default_rng()
        # a = 0.5
        # b = 2.0
        # cee = -1
        # t_min = 0
        # t_max = 10
        # n_points = 15
        # t_train = np.linspace(t_min, t_max, n_points)
        # y_train = self.gen_data(t_train, a, b, cee, noise=0.0, n_outliers=0)

        # x0 = np.array([1.0, 1.0, 0.0])
        # res_lsq = least_squares(self.fun, x0, args=(t_train, y_train))

        minbound = [min([x[0] for x in v_mn_xy]), min([y[1] for y in v_mn_xy])]
        maxbound = [max([x[0] for x in v_mn_xy]), max([y[1] for y in v_mn_xy])]
        avgbound = [sum([x[0] for x in v_mn_xy]) / len(v_mn_xy), sum([y[1] for y in v_mn_xy]) / len(v_mn_xy)]
        lsq = least_squares(self.f_wrap, avgbound, bounds=(minbound, maxbound), args=[v_mn_xy])

        v_n_xy = [lsq.x[0] * 2, lsq.x[1] * 2]
        v_n_spd = dist.euclidean([0, 0], v_n_xy) # For some reason, this ends up being only half of the ideally estimated distance...
        v_n_dir = self.angle_between_segments([v_n_xy[0] - 1, v_n_xy[1]], v_n_xy, [0, 0])

        return v_n_spd, v_n_dir, 0

    def f_wrap(self, x, v_mn):
        distances = []
        for v_Mn in v_mn:
            distances.append(dist.euclidean([0, 0], np.array([x[0], x[1]]) - v_Mn))
        super_distances = [d - (sum(distances) / len(distances)) for d in distances]

        return super_distances

    def angle_of_radical_instant_speed(self, tx: Tag, rx: Tag) -> list[float]:
        """
        Calculates the angle in degrees between a sender and a receiver
        in a 2d environment where the Z-axis is ignored.
        In the Tagoram paper, this would be ∠˜Vm,n.
        Helper function for estimating receiver velocity.

        Args:
            tx (Tag): The position of the sender.
            rx (Tag): The position of the receiver.

        Returns:
            dir_v_mn (float): The angle between in degrees.
        """

        tx_pos = tx.get_position()
        rx_pos = rx.get_position()

        a_pos = (rx_pos[0], rx_pos[1])
        b_pos = (rx_pos[0] - 1, rx_pos[1])
        c_pos = (tx_pos[0], tx_pos[1])

        return self.angle_between_segments(a_pos, b_pos, c_pos)

    def angle_between_segments(self, a: list[float], b: list[float], c: list[float]) -> list[float]:
        """
        Calculates the angle in degrees between two segments
        in a 2d environment (without a Z-axis).

        The arctans mainly model how they work in Desmos,
        which has a second parameter that cmath's arctan does not have.

        Args:
            a (list[float]): Point shared by both segments.
            b (list[float]): Point at the end of segment 1.
            c (list[float]): Point at the end of segment 2.

        Returns:
            ang (float): The angle between in degrees.
        """
        cax = c[0] - a[0]
        bax = b[0] - a[0]
        cay = c[1] - a[1]
        bay = b[1] - a[1]

        atan1 = 0
        if cax == 0:
            if cay > 0:
                atan1 = (pi / 2)
            elif cay < 0:
                atan1 = -(pi / 2)
            else:
                atan1 = 0
        else:
            atan1 = cmath.atan(cay / cax).real
            if cax < 0:
                atan1 = (atan1 % (2 * pi)) - pi


        atan2 = 0
        if bax == 0:
            if bay > 0:
                atan2 = (pi / 2)
            elif bay < 0:
                atan2 = -(pi / 2)
            else:
                atan2 = 0
        else:
            atan2 = cmath.atan(bay / bax).real
            if bax < 0:
                atan2 = (atan2 % (2 * pi)) - pi

        ang = (360 + (atan1 - atan2) * (180 / pi)) % 360
        if ang > 180:
            ang -= 360
        return ang