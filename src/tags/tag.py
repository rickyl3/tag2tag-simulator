from __future__ import annotations
from typing import Optional, Self, Any

import logging

from state import AppState
from tags.state_machine import TagMachine
from util.app_logger import init_tag_logger
from util.custom_types import Position

def _format_impedance(z) -> str:
    """
    Return a compact string for a complex impedance without parentheses,
    e.g. "20+0j" or "50-25j". If z is already a string, return it unchanged.
    """
    if isinstance(z, str):
        return z
    # z is expected to be a complex or numeric
    try:
        zr = float(z.real)
        zi = float(z.imag)
    except Exception:
        # fallback to str
        return str(z)
    sign = '+' if zi >= 0 else ''
    # use general format to avoid extra decimals
    return f"{zr:g}{sign}{zi:g}j"



class TagMode:
    """
    A mode which a tag's antenna can be in. This is a wrapper for an index
    into a tag's chip impedance table. It is assumed that 0
    refers to a connection to the envelope detector, or "listening mode".
    """

    _LISTENING_IDX = 0

    LISTENING: "TagMode"

    def __init__(self, index: int):
        """
        Initializes a TagMode

        Args:
            index (int): Antenna index.
        """
        self._index = index

    def is_listening(self) -> bool:
        """
        Returns:
            is_listening (bool): True if this tag mode refers to a listening configuration.
        """
        return self._index == TagMode._LISTENING_IDX

    def get_chip_index(self) -> int:
        """
        Returns:
            index (int): Returns the chip index associated with this mode.
        """
        return self._index

    def log_extra(self) -> dict:
        if self.is_listening():
            return {"is_listening": True}
        return {
            "is_listening": False,
            "chip_index": self.get_chip_index(),
        }

    def from_data(mode_str: str, chip_index: Optional[int]) -> Self:
        mode_str = mode_str.upper()
        match (mode_str):
            case "TRANSMIT":
                if chip_index is not None:
                    return TagMode(chip_index)
                raise ValueError("TRANSMIT mode requires a chip_index")
            case "LISTEN":
                return TagMode.LISTENING
            case _:
                raise ValueError(f"Unknown TagMode: {mode_str}")


TagMode.LISTENING = TagMode(TagMode._LISTENING_IDX)


class PhysicsObject:
    """
    An object which interacts with the physics engine. Used as a base class
    for Exciter and Tag.
    """

    def __init__(
        self,
        app_state: AppState,
        name: str,
        pos: Position,
        power: float,
        gain: float,
        impedance: float,
        frequency: float,
    ):
        """
        Creates a PhysicsObject.

        Args:
            app_state (AppState): The AppState.
            name (str): The name of this physics object.
            pos (Position): The position of this physics object.
            power (float): Power.
            gain (float): Gain.
            impedance (float): Antenna's Impedance.
            frequency (float): Frequency.
        """
        self.app_state = app_state
        self.name = name
        self.pos = tuple([float(p) for p in pos])
        self.power = power
        self.gain = gain
        self.impedance = impedance
        self.frequency = frequency

    def get_name(self) -> str:
        """
        Returns:
            name (str): The name of this physics object.
        """
        return self.name

    def get_position(self) -> Position:
        return self.pos

    def get_power(self):
        """
        Returns:
            power (float): Power.
        """
        return self.power

    def get_gain(self):
        """
        Returns:
            gain (float): Gain.
        """
        return self.gain

    def get_impedance(self):
        """
        Returns:
            impedance (float): Antenna's Impedance.
        """
        return self.impedance

    def get_frequency(self):
        """
        Returns:
            frequency (float): Frequency.
        """
        return self.frequency


class Exciter(PhysicsObject):
    """An exciter object, which transmits a signal backscattering tags can reflect"""

    def __init__(
        self,
        app_state: AppState,
        name: str,
        pos: Position,
        power: float,
        gain: float,
        impedance: float,
        frequency: float,
    ):
        """
        Creates a PhysicsObject.

        Args:
            app_state (AppState): The AppState.
            name (str): The name of this exciter.
            pos (Position): The position of this exciter.
            power (float): Power.
            gain (float): Gain.
            impedance (float): Antenna's Impedance.
            frequency (float): Frequency.
        """
        super().__init__(app_state, name, pos, power, gain, impedance, frequency)

    def to_dict(self) -> Any:
        """
        Converts an exciter object into a form that can be stored as JSON

        Returns:
            out (Any): Data storable as JSON.
        """
        return {
            "id": self.name,
            "x": self.pos[0],
            "y": self.pos[1],
            "z": self.pos[2],
            "power": self.power,
            "gain": self.gain,
            "impedance": self.impedance,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, app_state: AppState, data: Any) -> Exciter:
        """
        Converts data loaded from JSON into a new Exciter object.

        Args:
            app_state (AppState): The app state.
            data (Any): Data loaded from JSON.
        """
        return Exciter(
            app_state,
            data["id"],
            (data["x"], data["y"], data["z"]),
            data["power"],
            data["gain"],
            data["impedance"],
            data["frequency"],
        )


class Tag(PhysicsObject):
    """
    An object representing a backscattering tag.
    """

    def __init__(
        self,
        app_state: AppState,
        name: str,
        tag_machine: TagMachine,
        mode: TagMode,
        pos: Position,
        power: float,
        gain: float,
        impedance: float,
        chip_impedances: list[complex],
        frequency: float,
        power_on_threshold_dbm: float = -100.0,
    ):
        """
        Creates a Tag.

        Args:
            app_state (AppState): The app state.
            name (str): The name of this tag.
            tag_machine (TagMachine): The TagMachine associated with this tag.
            mode (TagMode): The initial mode this tag's antenna starts in.
            pos (Position): The position of this tag.
            power (float): Power.
            gain (float): Gain.
            impedance (float): Antenna's Impedance.
            chip_impedances (list[complex]): A list of chip impedances.
            power_on_threshold_dbm (float): Power threshold in dBm for tag to operate. Defaults to -100.0
            frequency (float): Frequency.
        """
        super().__init__(app_state, name, pos, power, gain, impedance, frequency)
        self.tag_machine = tag_machine
        self.mode = mode
        self.chip_impedances = chip_impedances
        self.power_on_threshold_dbm = power_on_threshold_dbm
        self.logger: logging.LoggerAdapter = init_tag_logger(self)

    def __str__(self):
        return f"Tag={{{self.name}}}"

    def run(self):
        """
        Run this tag with simpy
        """
        self.tag_machine.prepare()

    def set_mode(self, tag_mode: TagMode):
        self.mode = tag_mode

        msg: str
        if self.mode.is_listening():
            msg = "Set mode to LISTENING"
        else:
            msg = f"Set mode to REFLECT with index {self.mode.get_chip_index()}"
        self.logger.info(msg, extra={"mode": self.mode.log_extra()})

    def set_mode_listen(self):
        self.set_mode(TagMode.LISTENING)

    def set_mode_reflect(self, index: int):
        self.set_mode(TagMode(index))

    def get_mode(self):
        return self.mode

    def get_chip_impedance(self) -> complex:
        index = self.get_mode().get_chip_index()
        return self.chip_impedances[index]

    def read_voltage(self) -> float:
        tag_manager = self.app_state.tag_manager
        voltage = tag_manager.get_received_voltage(self)
        self.logger.info(
            f"Read voltage: {voltage}",
            extra={"voltage": voltage},
        )
        return voltage

    def to_dict(self):
        """For placing tags into dicts correctly on JSON"""

        # TODO what if self.power was set to default?
        return {
            "tag_machine": self.tag_machine.to_dict(),
            "x": self.pos[0],
            "y": self.pos[1],
            "z": self.pos[2],
            "power": self.power,
            "gain": self.gain,
            "impedance": self.impedance,
            "chip_impedances": [_format_impedance(x) for x in self.chip_impedances],
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(
        cls,
        app_state: AppState,
        name: str,
        data: dict,
        serializer,
        default: dict,
    ):
        """
        Creates a tag object from a JSON input

        Args:
            env (Environment): SimPy environment
            tag_manager (TagManager): Tag manager
            logger:
            name (str): Unique name for tag
            data (list): list of Coordinates

        Returns:
            tag: returns tag loaded from JSON
        """
        tag_machine = TagMachine.from_dict(app_state, data["tag_machine"], serializer)
        # prefer per-tag values; fall back to Default section
        chip_imp_list = data.get("chip_impedances", default.get("chip_impedances", []))
        freq = data.get("frequency", default.get("frequency"))
        pthr = data.get("power_on_threshold_dbm", default.get("power_on_threshold_dbm", -100.0))

        tag = cls(
            app_state,
            name,
            tag_machine,
            TagMode.LISTENING,
            (
                data["x"],
                data["y"],
                data["z"],
            ),
            0,
            default["gain"],
            default["impedance"],
            [complex(x) for x in chip_imp_list],
            freq,
            pthr,
        )
        tag_machine.set_tag(tag)
        return tag
