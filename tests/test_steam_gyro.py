from ngc import protocol as P
from ngc.steam_gyro import (
    _IMU_CAL,
    _REPORT_DESCRIPTOR,
    _REPORT_SIZE,
    _STICK_CAL,
    _pack_input,
    _rumble_frame,
    _switch_reply,
)


def test_switch_pro_report_contains_buttons_sticks_and_three_imu_samples():
    buttons = P.SWITCH_BUTTONS["A"] | P.SWITCH_BUTTONS["UP"] | P.SWITCH_BUTTONS["ZL"]
    report = _pack_input(buttons, (0.0, 0.0), (1.0, -1.0), (0, 0, 4096), (164, 0, 0), 7)

    assert len(report) == _REPORT_SIZE == 64
    assert report[0] == 0x30
    assert report[1] == 7
    assert report[3] & 0x08  # A
    assert report[5] & 0x02  # D-pad up
    assert report[5] & 0x80  # ZL
    assert report[6:9] == bytes.fromhex("000880")  # centered left stick
    assert report[9:12] == bytes.fromhex("ff0f00")  # right stick max/min
    assert report[13:25] == report[25:37] == report[37:49]


def test_switch_pro_handshake_and_device_info_replies():
    handshake = _switch_reply(bytes.fromhex("8001"), 1)
    assert handshake is not None
    assert handshake[0:2] == bytes.fromhex("8101")

    info = _switch_reply(bytes.fromhex("0100000000000000000002"), 2)
    assert info is not None
    assert info[0] == 0x21
    assert info[13:15] == bytes.fromhex("8202")
    assert info[15:19] == bytes.fromhex("04330302")


def test_factory_calibration_payload_sizes_and_spi_read():
    assert len(_REPORT_DESCRIPTOR) > 200
    assert len(_STICK_CAL) == 18
    assert len(_IMU_CAL) == 24

    request = bytearray(16)
    request[0] = 0x01
    request[10] = 0x10
    request[11:15] = (0x603D).to_bytes(4, "little")
    request[15] = 18
    reply = _switch_reply(bytes(request), 3)
    assert reply is not None
    assert reply[13:15] == bytes.fromhex("9010")
    assert reply[20:38] == _STICK_CAL


def test_rumble_frame_decoder():
    assert _rumble_frame(bytes.fromhex("00004040"), 0) == (0, 0)
    assert _rumble_frame(bytes.fromhex("00000000"), 0) is None
