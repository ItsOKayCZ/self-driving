import uuid

from mlagents_envs.side_channel.side_channel import IncomingMessage, OutgoingMessage, SideChannel


class DataChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("9bc23f51-e0e8-450c-b3c5-e4d2032151ec"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        print(msg.read_string())

    def set_int_parameter(self, key: str, val: int) -> None:
        msg = OutgoingMessage()
        msg.write_string(f"int|{key}|{val}")
        super().queue_message_to_send(msg)

    def set_color_parameter(self, key: str, val: tuple[int, ...]) -> None:
        msg = OutgoingMessage()
        msg.write_string(f"color|{key}|{val[0]},{val[1]},{val[2]}")
        super().queue_message_to_send(msg)

    def set_float_parameter(self, key: str, val: float) -> None:
        msg = OutgoingMessage()
        msg.write_string(f"float|{key}|{val}")
        super().queue_message_to_send(msg)

    def set_bool_parameter(self, key: str, val: bool) -> None:
        msg = OutgoingMessage()
        msg.write_string(f"int|{key}|{1 if val else 0}")
        super().queue_message_to_send(msg)
