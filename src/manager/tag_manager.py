from physics import PhysicsEngine
from tags.tag import Exciter, Tag


class TagManager:
    """
    A store of Tag instances, along with Exciters.
    """

    def __init__(
        self,
        exciters: dict[str, Exciter],
        tags: dict[str, Tag] = dict(),
        passive_ref_mag: float = 0,
    ):
        """
        Creates a TagManager.

        Args:
            exciters (dict[str, Exciter]): A dictionary mapping exciter names to Exciter instances.
            tags (dict[str, Tag]): A dictionary mapping tag names to Tag instances.
        """
        self.tags: dict[str, Tag] = tags
        self.physics_engine = PhysicsEngine(exciters, passive_ref_mag=passive_ref_mag)

    def add_tags(self, *tags: Tag) -> None:
        """
        Register one or more tags with the TagManager for any future references

        Args:
            tag (Tag): The tag to register
        """
        for tag in tags:
            self.tags[tag.name] = tag

    def remove_by_name(self, *names: str) -> None:
        """
        Remove one or more tags by name from the TagManager

        Args:
            names (str): The names to remove from the tags dictionary
        """
        for name in names:
            self.tags.pop(name)

    def get_by_name(self, name: str) -> Tag:
        """
        Retreive a Tag by name. Raises a ValueError if the tag doesn't exist.

        Args:
            name (str): The tag name.
        """
        tag = self.tags[name]
        if tag is None:
            raise ValueError(f"{name}: Tag by this name does not exist!")
        return tag

    def get_received_voltage(self, asking_tag: Tag) -> float:
        """
        Retrieve the voltage received by a tag.

        Args:
            name (str): The tag name.

        Returns:
            voltate (float): The received voltage.
        """
        return self.physics_engine.voltage_at_tag(self.tags, asking_tag)

    def get_doppler_effect(self, asking_receiver: Tag) -> tuple[float, float, float]:
	    """
        Calculates the phase angle, phase difference,
        and perceived frequency with the doppler effect between sender and a tag.

	    Assumes 1 sender and that all receivers are ordered alphabetically.
	    # TODO: Is there a way to make this not an assumption?

	    Args:
	        asking_receiver (str): The receiver.

	    Returns:
	        phase_ang (float): The phase angle.
	        phase_diff (float): The phase difference.
	        doppler_freq (float): The perceived frequency with the doppler effect.
	    """
	    tagz = list(self.tags.values())
	    asking_sender = None
	    for tag in tagz:
		    if tag.tag_machine.input_machine.state.name == "EMPTY":
			    asking_sender = tag
			    break  # TODO: The first sender alphabetically is picked by default. We need multiple to estimate receiver velocity.

	    return self.physics_engine.doppler_effect(self.tags, asking_sender, asking_receiver)

    def get_estimated_rx_velocity(self, asking_receiver: Tag) -> tuple[float, float, float]:
	    """
	    Calculates the estimated receiver velocity using a methods described in
	    https://dl.acm.org/doi/pdf/10.1145/2639108.2639111 .
	    Limited to 2-dimensions where positions along the Z-axis are ideally ignored,
	    but it sounds possible to implement down the line, maybe?

	    Assumes all receivers are ordered alphabetically.
	    # TODO: Is there a way to make this not an assumption?

	    Args:
	        asking_receiver (str): The receiver.

        Returns:
            v_n_spd (float): The estimated receiver velocity speed.
            v_n_dir (float): The estimated receiver velocity angle along the x-axis.
	    """
	    return self.physics_engine.estimate_rx_velocity(self.tags, asking_receiver)
