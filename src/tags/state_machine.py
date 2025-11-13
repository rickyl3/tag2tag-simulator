from __future__ import annotations

from typing import Optional, Self, Any, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import logging
from logging import Logger
import heapq

from simpy.core import SimTime
from simpy import Interrupt

from state import AppState
from util.app_logger import init_machine_logger
from util.custom_types import StateMethod

if TYPE_CHECKING:
    from tags.tag import Tag


class Timer:
    """
    Object which represents a delayed callback. Timer instances
    compare as less-than Timer instances which run before them.
    """

    def __init__(self, timer_acceptor: TimerAcceptor, next_run: SimTime):
        """
        Create a Timer object.

        Args:
            timer_acceptor (TimerAcceptor): The TimerAcceptor that scheduled this Timer.
            next_run (SimTime): The SimPy time at which this Timer should be run.
        """
        self._timer_acceptor = timer_acceptor
        self._next_run = next_run
        self._is_canceled = False

    def get_next_run(self) -> SimTime:
        """
        Returns:
            next_run (SimTime): The SimPy time at which this Timer should be run.
        """
        return self._next_run

    def cancel(self):
        """
        Cancels this timer, making future attempts to run this Timer's callback a no-op.
        """
        self._is_canceled = True

    def is_canceled(self) -> bool:
        """
        Returns:
            is_canceled (bool): True if this Timer has been canceled.
        """
        return self._is_canceled

    def run(self):
        """
        Runs this Timer and then cancels it, so that subsequent runs will be a no-op.
        If this Timer has already been canceled, this is a no-op.
        """
        if not self.is_canceled():
            self._timer_acceptor.on_timer()
            self.cancel()

    def __lt__(self, other: Self) -> bool:
        """
        Compares two Timer instances via their next_run times.

        Args:
            other (Self): Other Timer instance.

        Returns:
            is_lt (bool): True if this timer should be sorted before the other Timer.
        """
        return self._next_run < other._next_run


class TimerScheduler:
    """
    Used to schedule Timers and run their callbacks at the appropriate times.
    """

    def __init__(self, app_state: AppState):
        """
        Creates a new TimerScheduler.

        Args:
            app_state (AppState): The app state.
        """
        self.app_state = app_state
        self.timers: list[Timer] = []
        self.next_run: Optional[int] = None
        self.process = self.app_state.env.process(self.run())

    def run(self):
        """
        Fullfils Timers asynchronously, in such a way that it can be run as a SimPy process.
        """
        while True:
            while (
                len(self.timers) != 0
                and self.timers[0].get_next_run() <= self.app_state.now()
            ):
                self.timers[0].run()
                heapq.heappop(self.timers)
            delay: SimTime
            if len(self.timers) == 0:
                self.next_run = None
                delay = float("inf")
            else:
                self.next_run = self.timers[0].get_next_run()
                delay = self.next_run - self.app_state.now()
            try:
                yield self.app_state.env.timeout(delay)
            except Interrupt:
                pass

    def set_timer(self, timer_acceptor: TimerAcceptor, delay: int) -> Timer:
        """
        Schedules a timer event.

        Args:
            timer_acceptor (TimerAcceptor): The TimerAcceptor which is requesting a future callback.
            delay (int): The delay in SimPy simulation ticks before the callback should occur.
        """
        assert delay >= 0
        timer = Timer(timer_acceptor, self.app_state.now_plus(delay))
        heapq.heappush(self.timers, timer)
        if self.next_run is None or self.timers[0].get_next_run() < self.next_run:
            self.process.interrupt()
        return timer


# Maybe rename to TimerAccessor
class TimerAcceptor(ABC):
    """
    Abstract class representing something that can receive delayed callbacks.
    """

    def __init__(self, timer: TimerScheduler):
        """
        Create a TimerAcceptor.

        Args:
            timer (TimerScheduler): The TimerScheduler which should be used to schedule delayed callbacks.
        """
        self._scheduler = timer
        self._last_timer: Optional[Timer] = None

    # TODO remove this method?
    def set_timer(self, delay: int):
        """
        Schedules a delayed callback. If the delay is 0, cancel the last callback instead.
        Cancels any pending callbacks.

        Args:
            delay (int): The delay in SimPy simulation ticks.
        """
        if self._last_timer is not None:
            self._last_timer.cancel()
            self._last_timer = None
        if delay != 0:
            self._last_timer = self._scheduler.set_timer(self, delay)
        # TODO: What about if delay is zero?

    @abstractmethod
    def on_timer(self):
        """
        Called when a timer event goes off.
        """
        pass


class State:
    """
    A State in a StateMachine.
    """

    def __init__(self, name: str):
        """
        Creates a State.

        Args:
            name (str): The state's unique name.
        """

        self.transitions: dict[str, tuple[StateMethod, State]] = {}
        self.name = name

    def add_transition(self, expect_symbol: str, method: StateMethod, state: "State"):
        """
        Adds a transition to another state.

        Args:
            expect_symbol (str): The symbol which should cause this transition.
            method (StateMethod): The command which should be run after this transition.
            state (State): The state to enter after this transition.
        """
        self.transitions[expect_symbol] = (method, state)

    def follow_symbol(self, symbol: str):
        """
        Returns the state this state should transition to, or None if no transition should happen.

        Args:
            symbol (str): The received symbol.
        """
        return self.transitions.get(symbol)

    def does_accept_symbol(self, symbol: str):
        """
        Returns True if this state has a transition upon seeing the given symbol.

        Args:
            symbol (str): The given symbol.
        """
        return symbol in self.transitions

    def get_name(self):
        """
        Returns the name of this state.

        Returns:
            name (str): The state name.
        """
        return self.name

    @classmethod
    def _method_from_dict(cls, d: Any) -> StateMethod:
        """
        Construct a StateMethod from data parsed from JSON.

        Args:
            method: The StateMethod in a JSON-storable format.
        Returns:
            method (StateMethod): The StateMethod.
        """
        if isinstance(d, list):
            return tuple([cls._method_from_dict(x) for x in d])
        else:
            return d

    @classmethod
    def _method_to_dict(cls, method: StateMethod) -> Any:
        """
        Converts a StateMethod into a format which can be stored as JSON.

        Args:
            method (StateMethod): The StateMethod.
        Returns:
            method: The StateMethod in a JSON-storable format.
        """
        if isinstance(method, tuple):
            return [cls._method_to_dict(x) for x in method]
        else:
            return method

    def to_dict(self):
        """
        Converts a State into a format which can be stored as JSON.

        Returns:
            state: The State in a JSON-storable format.
        """
        transitions_serialized = {}
        for expect_input, (method, state) in self.transitions.items():
            transitions_serialized[expect_input] = [
                self._method_to_dict(method),
                state.name,
            ]
        return {"id": self.name, "transitions": transitions_serialized}

    @classmethod
    def from_dict(cls, data, serializer: StateSerializer, id=None):
        """
        Construct a State from data parsed from JSON.

        Args:
            data: The State in a JSON-storable format.
            serializer (StateSerializer): The serializer to use to construct neighboring states.
        Returns:
            method (StateMethod): The StateMethod.
        """
        if isinstance(data, str):
            return serializer.get_state(data)
        id = data.get("id")
        state = serializer.get_state(id)
        for expect_input, (method, output_id) in data["transitions"].items():
            output_state = serializer.get_state(output_id)
            state.add_transition(
                expect_input, cls._method_from_dict(method), output_state
            )
        return state


class StateSerializer:
    """
    Used to deserialize States with cyclic relationships
    """

    states: Dict[str, State]

    def __init__(self):
        """
        Initialize a StateSerializer.
        """
        self.states = {}

    def get_state(self, name: str) -> State:
        """
        Retrieve a State by name, creating one if missing.

        Args:
            name (str): The state name.
        Returns:
            state (State): The state.
        """
        if name not in self.states:
            self.states[name] = State(name)
        return self.states[name]

    def get_state_map(self) -> Dict[str, State]:
        """
        Gets the current mapping between state names and states

        Returns:
            data (Dict[str, State]): The dictionary of currently loaded states.
        """
        return self.states

    def to_dict(self):
        """
        Converts a State into a format which can be stored as JSON.

        Returns:
            state: The State in a JSON-storable format.
        """
        return [{"id": state.name, **state.to_dict()} for state in self.states.values()]


class StateMachine:
    """
    A simple state machine.
    """

    def __init__(self, init_state: State):
        """
        Create a StateMachine.

        Args:
            init_state (State): The state machine's initial state.
        """
        self.state = init_state
        self.init_state = init_state

    def get_state(self):
        """
        Returns:
            state (State): The state machine's current state.
        """
        return self.state

    def get_init_state(self):
        """
        Returns:
            initial_state (State): The state machine's initial state.
        """
        return self.init_state

    def transition(self, symbol) -> Optional[StateMethod]:
        """
        Causes the state machine to attempt to follow a transition symbol.

        Returns:
            method (Optional[StateMethod]): None if the state machine didn't perform
            a transition, the StateMethod this machine expects to be executed otherwise.
        """
        out = self.state.follow_symbol(symbol)
        if out is None:
            return None
        self.state = out[1]
        return out[0]


class MachineLogger:
    """
    Logging interface used to buffer logs from state machines
    """

    def __init__(self):
        """
        Create a MachineLogger.
        """
        self.store = ""
        self.logger: logging.LoggerAdapter

    def log(self, s: str):
        """
        Log a message into the MachineLogger's buffer.
        The buffer's contents up to the last newline character will be flushed.

        Args:
            s (str): Message to store in buffer.
        """
        newline_index = s.find("\n")
        while newline_index != -1:
            msg = self.store + s[:newline_index]
            self.logger.info(msg, extra={"action": "write_output"})
            self.store = ""
            s = s[newline_index + 1 :]
            newline_index = s.find("\n")
        self.store += s

    def set_logger(self, logger: logging.LoggerAdapter):
        """
        Sets the logger to forward buffered logging to.

        Args:
            logger (Logger): Logger to forward buffered logging to.
        """
        self.logger = init_machine_logger(logger)


class ExecuteMachine(StateMachine, TimerAcceptor):
    """
    A state machine that can execute StateMethod instances during transitions.
    Methods starting with "_cmd_" directly map to StateMethod commands.
    """

    def __init__(self, tag_machine: TagMachine, init_state: State):
        """
        Creates an ExecuteMachine.

        Args:
            tag_machine (TagMachine): The TagMachine this ExecutionMachine belongs to.
            init_state (State): The state machine's initial state.
        """
        super().__init__(init_state)
        self.tag_machine = tag_machine
        self.transition_queue: Optional[list[str]] = None
        self.registers: list[int | float] = [0 for _ in range(8)]

    def logger(self) -> logging.LoggerAdapter:
        return self.tag_machine.tag.logger

    def _cmd(self, cmd_first: str, cmd_rest: list[StateMethod]):
        """
        Calls a _cmd_* method.

        Args:
            cmd_first (str): Command name.
            cmd_rest (list[StateMethod]): Command arguments.
        """
        method_name = "_cmd_" + cmd_first
        getattr(self, method_name)(*cmd_rest)

    def _cmd_mov(self, dst: int, src: int):
        """
        Command that performs dst := src.

        Args:
            dst (int): Destination register.
            src (int): Source register.
        """
        value = self.registers[src]
        self.registers[dst] = value
        self.logger().debug(
            "cmd_mov(%s,%(src)s): reg[%s] = reg[%(src)s]: %s",
            dst,
            src,
            dst,
            src,
            value,
            extra={"dst": dst, "src": src, "value": value},
        )

    def _cmd_load_imm(self, dst: int, val: Union[int, float]):
        """
        Command that performs dst := $val.

        Args:
            dst (int): Destination register.
            src (Union[int, float]): Immediate value.
        """
        self.registers[dst] = val
        self.logger().debug(
            "cmd_load_imm(%s,%s): reg[%s] = %s",
            dst,
            val,
            dst,
            val,
            extra={"dst": dst, "value": val},
        )

    def _cmd_sub(self, dst: int, a: int, b: int):
        """
        Command that performs dst := a - b.

        Args:
            dst (int): Destination register.
            a (int): First operand register.
            b (int): Second operand register.
        """
        value = self.registers[dst] = self.registers[a] - self.registers[b]
        self.logger().debug(
            "cmd_sub(%s,%s,%s): reg[%s] = reg[%s] - reg[%s]: %s",
            dst,
            a,
            b,
            dst,
            a,
            b,
            value,
            extra={"dst": dst, "a": a, "b": b, "value": value},
        )

    def _cmd_add(self, dst: int, a: int, b: int):
        """
        Command that performs dst := a + b.

        Args:
            dst (int): Destination register.
            a (int): First operand register.
            b (int): Second operand register.
        """
        value = self.registers[dst] = self.registers[a] + self.registers[b]
        self.logger().debug(
            "cmd_add(%s,%s,%s): reg[%s] = reg[%s] + reg[%s]: %s",
            dst,
            a,
            b,
            dst,
            a,
            b,
            value,
            extra={"dst": dst, "a": a, "b": b, "value": value},
        )

    def _cmd_floor(self, a: int):
        """
        Command that performs a := int(a). In other words, it performs floor(n).

        Args:
            a (int): Register used for both input and output.
        """
        value = self.registers[a] = int(self.registers[a])
        self.logger().debug(
            "cmd_floor(%s): floor(reg[%s]): %s",
            a,
            a,
            value,
            extra={"a": a, "value": value},
        )

    def _cmd_abs(self, a: int):
        """
        Command that performs a := abs(a).

        Args:
            a (int): Register used for both input and output.
        """
        value = self.registers[a] = abs(self.registers[a])
        self.logger().debug(
            "cmd_abs(%s): abs(reg[%s]): %s",
            a,
            a,
            value,
            extra={"a": a, "value": value},
        )

    def _cmd_compare(self, a, b):
        """
        Sends symbol "lt", "eq", or "gt" to self,
        depending on whether a < b, a = b, or a > b respectively.

        Args:
            a (int): First operand register.
            b (int): Second operand register.
        """
        a_val = self.registers[a]
        b_val = self.registers[b]
        sym = None
        if a_val < b_val:
            sym = "lt"
        elif a_val == b_val:
            sym = "eq"
        else:
            sym = "gt"
        self._accept_symbol(sym)
        self.logger().debug(
            "cmd_compare(%s,%s): comp(reg[%s], reg[%s]): %s",
            a,
            b,
            a,
            b,
            sym,
            extra={"a": a, "b": b, "value": sym},
        )

    def _cmd__comment(self, *comment_lines: Any):
        """
        Command that performs no operation, intended for use in documenting state machines.

        Args:
            *comment_lines (Any): Ignored parameters.
        """
        pass

    def _cmd_sequence(self, *cmd_list: StateMethod):
        """
        Command that executes a group of commands in order.

        Args:
            *cmd_list (StateMethod): Commands to execute.
        """
        for cmd in cmd_list:
            (cmd_first, *cmd_rest) = cmd
            self._cmd(cmd_first, cmd_rest)

    def _cmd_self_trigger(self, symbol: str):
        """
        Command that sends a symbol to this state machine.

        Args:
            symbol (str): Symbol to send.
        """
        self._accept_symbol(symbol)
        self.logger().debug(
            "cmd_self_trigger(%s)",
            symbol,
            extra={"symbol": symbol},
        )

    def _cmd_set_timer(self, timer_reg: int):
        """
        Command that sets a timer. Setting a timer of 0 delay cancels the current timer.

        Args:
            timer_reg (int): Input register for the timer delay in SimPy ticks.
        """
        delay = self.registers[timer_reg]
        self.tag_machine.timer.set_timer(self, delay)
        self.logger().debug(
            "cmd_set_timer(%s): set timer to %s",
            timer_reg,
            delay,
            extra={"timer_reg": timer_reg, "delay": delay},
        )

    def prepare(self):
        """
        Sends an initialization symbol to the state machine, which the state machine
        can use to execute initialization commands (like setting a timer).
        """
        self._accept_symbol("init")

    def on_timer(self):
        """
        Run when a timer set by this state machine is triggered.
        """
        self._accept_symbol("on_timer")

    def _accept_symbol(self, symbol: str):
        """
        Dispatches symbol reception events to _cmd_* methods.

        Args:
            symbol (str): Symbol received.
        """
        if self.transition_queue is None:
            self.transition_queue = [symbol]
        else:
            self.transition_queue.append(symbol)
            return
        while len(self.transition_queue) != 0:
            symbol = self.transition_queue[0]
            self.transition_queue = self.transition_queue[1:]
            cmd = self.transition(symbol)
            if cmd is not None:
                (cmd_first, *cmd_rest) = cmd
                self._cmd(cmd_first, cmd_rest)
        self.transition_queue = None


class InputMachine(ExecuteMachine):
    """
    An execute machine used as the first stage of a TagMachine.
    """

    def __init__(self, tag_machine: TagMachine, init_state: State):
        """
        Creates an InputMachine.

        Args:
            tag_machine (TagMachine): The TagMachine this ExecutionMachine belongs to.
            init_state (State): The state machine's initial state.
        """
        super().__init__(tag_machine, init_state)

    def _cmd_save_voltage(self, out_reg):
        """
        Command that saves the input voltage from the envelope detector to a register.

        Args:
            out_reg (int): Output register.
        """
        voltage = self.registers[out_reg] = self.tag_machine.tag.read_voltage()
        self.logger().debug(
            "cmd_save_voltage(%s): reg[%s] = %s",
            out_reg,
            out_reg,
            voltage,
            extra={"out_reg": out_reg, "voltage": voltage},
        )

    def _cmd_send_bit(self, reg: int):
        """
        Command that sends a bit of data to the processing machine.

        Args:
            reg (int): The register where the bit is stored.
        """
        self.tag_machine.processing_machine.on_recv_bit(self.registers[reg] != 0)

    def _cmd_forward_voltage(self):
        """
        Command that sends a voltage reading to the processing machine.
        """
        self.tag_machine.processing_machine.on_recv_voltage(
            self.tag_machine.tag.read_voltage()
        )


class ProcessingMachine(ExecuteMachine):
    """
    An execute machine used as the second stage of a TagMachine, intended for processing data.
    """

    def __init__(self, tag_machine: TagMachine, init_state: State):
        """
        Creates a ProcessingMachine.

        Args:
            tag_machine (TagMachine): The TagMachine this ExecutionMachine belongs to.
            init_state (State): The state machine's initial state.
        """
        super().__init__(tag_machine, init_state)
        self.tag_machine = tag_machine
        self.mem = [0 for _ in range(64)]

    def on_recv_bit(self, val: bool):
        """
        Called when a processing machine receives a bit from its associated input machine.

        Args:
            val (bool): The received bit.
        """
        self.registers[7] = val and 1 or 0
        self._accept_symbol("on_recv_bit")

    def on_recv_voltage(self, val: float):
        """
        Called when a processing machine receives a voltage from its associated input machine.

        Args:
            val (float): The received voltage.
        """
        self.registers[7] = val
        self._accept_symbol("on_recv_voltage")

    def on_queue_up(self):
        """
        Called when a processing machine receives a voltage from its associated input machine.

        Args:
            val (float): The received voltage.
        """
        self._accept_symbol("on_queue_up")

    def _cmd_send_int_out(self, reg: int):
        """
        Command that sends an integer to an associated output machine.

        Args:
            reg (int): Input register.
        """
        self.tag_machine.output_machine.on_recv_int(self.registers[reg])

    def _cmd_send_int_log(self, reg):
        """
        Command that sends an integer to an associated logger.

        Args:
            reg (int): Input register.
        """
        self.tag_machine.machine_logger.log(str(self.registers[reg]))

    def _cmd_send_str_log(self, s: str):
        """
        Command that sends a string to an associated logger.

        Args:
            reg (int): Input register.
        """
        self.tag_machine.machine_logger.log(s)

    def _cmd_store_mem_imm(self, reg_addr: int, imm: Union[tuple[int,], int]):
        """
        Command that stores immediate value(s) in memory.

        Args:
            reg_addr (int): Input register containing the memory address to start saving values at.
            imm: (Union[tuple[int,], int]): Values to save to memory.
        """
        if isinstance(imm, tuple):
            imm = list(imm)
        else:
            imm = [imm]
        base_idx = self.registers[reg_addr]
        for idx in range(len(imm)):
            self.mem[base_idx + idx] = imm[idx]

    def _cmd_load_mem(self, dst: int, addr_reg: int):
        """
        Command that loads a value into a register from memory.

        Args:
            dst (int): Output register.
            addr_reg (int): Input register containing the memory address to load from.
        """
        self.registers[dst] = self.mem[self.registers[addr_reg]]


class OutputMachine(ExecuteMachine, TimerAcceptor):
    """
    An execute machine used as the third and final stage of a TagMachine, intended for sending data over the tag network.
    """

    def __init__(self, tag_machine: TagMachine, init_state: State):
        """
        Creates an OutputMachine.

        Args:
            tag_machine (TagMachine): The TagMachine this ExecutionMachine belongs to.
            init_state (State): The state machine's initial state.
        """
        super().__init__(tag_machine, init_state)

    def _cmd_set_antenna(self, reg: int):
        """
        Command that sets the associated tag's antenna to an associated mode.

        Args:
            reg (int): Register containing the antenna index.
        """
        reflection_index = self.registers[reg]
        self.tag_machine.tag.set_mode_reflect(reflection_index)

    def _cmd_set_listen(self):
        """
        Command that sets the associated tag's antenna to the envelope detector (index 0).
        """
        self.tag_machine.tag.set_mode_listen()

    def _cmd_queue_processing(self):
        """
        Command that sends a wake-up command to an associated processing machine.
        """
        self.tag_machine.processing_machine.on_queue_up()

    def on_recv_int(self, n: int):
        """
        Called when an output machine receives an integer from its associated processing machine.

        Args:
            n (int): The received integer.
        """
        self.registers[7] = n
        self._accept_symbol("on_recv_int")


class TagMachine:
    """
    A combination of an input machine, processing machine, output machine, and logger.
    """

    def __init__(
        self,
        app_state: AppState,
        init_states: tuple[State, State, State],
    ):
        """
        Creates a TagMachine.

        Args:
            app_state (AppState): The app state.
            init_states (tuple[State, State, State]): The initial states for each state machine.
        """
        self.timer = TimerScheduler(app_state)
        self.machine_logger = MachineLogger()
        self.input_machine = InputMachine(self, init_states[0])
        self.processing_machine = ProcessingMachine(self, init_states[1])
        self.output_machine = OutputMachine(self, init_states[2])
        self.tag: Tag

    def set_tag(self, tag: Tag):
        """
        Sets the tag associated with this TagMachine. Should be called right after initializing a TagMachine.

        Args:
            tag (Tag): The associated Tag.
        """
        self.tag = tag
        self.machine_logger.set_logger(tag.logger)

    def prepare(self):
        """
        Sends an initialization symbol to each contained state machine.
        """
        # can set antenna settings before everything else
        self.output_machine.prepare()
        self.input_machine.prepare()
        self.processing_machine.prepare()

    def to_dict(self) -> Any:
        """
        Converts this tag machine into a form which can be stored in JSON.
        """
        return {
            "input_machine": self.input_machine.init_state.name,
            "processing_machine": self.processing_machine.init_state.name,
            "output_machine": self.output_machine.init_state.name,
        }

    @classmethod
    def from_dict(
        cls, app_state: AppState, data, serializer: StateSerializer
    ) -> TagMachine:
        """
        Creates a tag machine from a JSON input

        Args:
            app_state (AppState): The app state.
            logger (Logger): A logger.
            data (Any): Data loaded from JSON.
            serializer (StateSerializer): The state serializer used to load states.

        Returns:
            tag_machine: A new tag machine.
        """
        return cls(
            app_state,
            (
                State.from_dict(data["input_machine"], serializer),
                State.from_dict(data["processing_machine"], serializer),
                State.from_dict(data["output_machine"], serializer),
            ),
        )
