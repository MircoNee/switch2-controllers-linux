"""Nintendo Switch Pro HID emulation for Steam Input gyro support.

The Switch 2 BLE bridge already receives motion samples.  This module exposes
those samples through a native Switch Pro HID interface over Linux UHID, which
lets Steam's HIDAPI path discover a controller with native IMU reports.
"""

from __future__ import annotations

import errno
import logging
import os
import select
import struct
import threading
from typing import Callable, Optional

from . import protocol as P

logger = logging.getLogger(__name__)

# Linux include/uapi/linux/uhid.h event identifiers.
_UHID_DESTROY = 1
_UHID_START = 2
_UHID_STOP = 3
_UHID_OUTPUT = 6
_UHID_GET_REPORT = 9
_UHID_GET_REPORT_REPLY = 10
_UHID_CREATE2 = 11
_UHID_INPUT2 = 12
_UHID_SET_REPORT = 13
_UHID_SET_REPORT_REPLY = 14

_UHID_DATA_MAX = 4096
_UHID_EVENT_SIZE = 4 + 128 + 64 + 64 + 2 + 2 + 16 + _UHID_DATA_MAX
_BUS_BLUETOOTH = 0x05
_VENDOR_NINTENDO = 0x057E
_PRODUCT_SWITCH_PRO = 0x2009
_REPORT_ID_INPUT = 0x30
_REPORT_SIZE = 64

# Nintendo's Switch Pro descriptor, including native input, output and feature
# report IDs. Kept byte-for-byte compatible with the standard HIDAPI profile.
_REPORT_DESCRIPTOR = bytes.fromhex(
    "050115000904a1018530050105091901290a150025017501950a550065008102"
    "0509190b290e150025017501950481027501950281030b01000100a100"
    "0b300001000b310001000b320001000b35000100150027ffff0000751095048102c0"
    "0b39000100150025073500463b0165147504950181020509190f291215002501"
    "7501950481027508953481030600ff852109017508953f8103858109027508953f"
    "8103850109037508953f9183851009047508953f9183858009057508953f9183"
    "8209067508953f9183c0"
)

# Standard factory calibration blocks returned to HIDAPI during initialization.
_STICK_CAL = bytes.fromhex("fff77f000880fff77f000880fff77ffff77f")
_IMU_CAL = bytes.fromhex(
    "0000000000000040004000400000000000003b343b343b34"
)

# Switch rumble encodes a 0..100 amplitude index in each motor frame.
def _rumble_frame(report: bytes, offset: int) -> Optional[tuple[int, int]]:
    if len(report) < offset + 4:
        return None
    high_index = (report[offset + 1] & 0xFE) // 2
    low_code = report[offset + 3]
    if high_index > 100 or not 0x40 <= low_code <= 0x72:
        return None
    low_index = (low_code - 0x40) * 2 + (1 if report[offset + 2] & 0x80 else 0)
    if low_index > 100:
        return None
    return low_index, high_index


def _switch_reply(output: bytes, timer: int) -> Optional[bytes]:
    """Answer Switch Pro handshake and subcommand reports sent by HIDAPI."""
    if not output:
        return None
    if output[0] == 0x80:
        if len(output) < 2 or output[1] == 0x04:
            return None
        reply = bytearray(_REPORT_SIZE)
        reply[0] = 0x81
        reply[1] = output[1]
        if output[1] == 0x01:
            reply[3] = 0x03
            reply[4:10] = bytes.fromhex("010000000002")
        return bytes(reply)
    if output[0] != 0x01 or len(output) < 11:
        return None

    subcommand = output[10]
    if subcommand == 0x10 and len(output) < 16:
        return None

    reply = bytearray(_REPORT_SIZE)
    reply[0] = 0x21
    reply[1] = timer & 0xFF
    reply[2] = 0x81
    reply[6:12] = bytes.fromhex("000880000880")
    reply[13] = 0x90 if subcommand == 0x10 else (0x82 if subcommand == 0x02 else 0x80)
    reply[14] = subcommand

    if subcommand == 0x02:
        reply[15:27] = bytes.fromhex("043303020200000000010101")
    elif subcommand == 0x10:
        address = int.from_bytes(output[11:15], "little")
        size = output[15]
        reply[15:20] = output[11:16]
        blocks = ((0x603D, _STICK_CAL), (0x6020, _IMU_CAL))
        for base, data in blocks:
            begin = max(address, base)
            end = min(address + size, base + len(data))
            if begin < end:
                src = begin - base
                dst = 20 + begin - address
                reply[dst:dst + end - begin] = data[src:src + end - begin]
    return bytes(reply)


def _pack_stick(report: bytearray, offset: int, x: float, y: float) -> None:
    sx = max(0, min(4095, round((max(-1.0, min(1.0, x)) + 1.0) * 2047.5)))
    sy = max(0, min(4095, round((max(-1.0, min(1.0, y)) + 1.0) * 2047.5)))
    report[offset] = sx & 0xFF
    report[offset + 1] = ((sx >> 8) & 0x0F) | ((sy & 0x0F) << 4)
    report[offset + 2] = (sy >> 4) & 0xFF


def _pack_input(
    buttons: int,
    left_stick: tuple[float, float],
    right_stick: tuple[float, float],
    accel: tuple[int, int, int],
    gyro: tuple[int, int, int],
    timer: int,
) -> bytes:
    """Pack one full 64-byte Switch Pro 0x30 report with three IMU samples."""
    out = bytearray(_REPORT_SIZE)
    out[0] = _REPORT_ID_INPUT
    out[1] = timer & 0xFF
    out[2] = 0x81  # Full battery; wireless connection.

    def pressed(name: str) -> bool:
        return bool(buttons & P.SWITCH_BUTTONS[name])

    # Face buttons, right shoulder/trigger.
    for name, mask in (("Y", 0x01), ("X", 0x02), ("B", 0x04), ("A", 0x08),
                       ("SR_R", 0x10), ("SL_R", 0x20), ("R", 0x40), ("ZR", 0x80)):
        if pressed(name):
            out[3] |= mask
    # Minus/plus, stick clicks, Home and Capture.
    for name, mask in (("MINUS", 0x01), ("PLUS", 0x02), ("R_STK", 0x04),
                       ("L_STK", 0x08), ("HOME", 0x10), ("CAPTURE", 0x20)):
        if pressed(name):
            out[4] |= mask
    # D-pad, left shoulder/trigger.
    for name, mask in (("DOWN", 0x01), ("UP", 0x02), ("RIGHT", 0x04),
                       ("LEFT", 0x08), ("SR_L", 0x10), ("SL_L", 0x20),
                       ("L", 0x40), ("ZL", 0x80)):
        if pressed(name):
            out[5] |= mask

    _pack_stick(out, 6, *left_stick)
    _pack_stick(out, 9, *right_stick)

    ax, ay, az = accel
    gx, gy, gz = gyro

    # Accelerometer orientation, derived from the DSU-vs-SDL direction test:
    #   DSU right   -> SDL up
    #   DSU up      -> SDL forward
    #   DSU forward -> SDL left
    #   HID X = -Switch2 Z, HID Y = +Switch2 X, HID Z = +Switch2 Y.
    ax, ay, az = -az, ax, ay

    # Gyroscope needs its own permutation; its axes do not share the
    # accelerometer's orientation in the observed DSU/SDL mappings:
    #   DSU pitch up  -> SDL yaw right
    #   DSU roll left -> SDL pitch up
    #   DSU yaw left  -> SDL roll right
    # Therefore HID pitch/roll/yaw = DSU roll/-yaw/pitch.
    gx, gy, gz = gy, -gz, gx

    # The bridge's DSU path interprets gyro as 16.4 LSB/(deg/s);
    # native Pro reports use 14.2842 LSB/(deg/s). Accel is already 4096 LSB/g.
    gyro_scale = 14.2842 / 16.4
    sample = struct.pack(
        "<6h",
        max(-32768, min(32767, ax)),
        max(-32768, min(32767, ay)),
        max(-32768, min(32767, az)),
        max(-32768, min(32767, round(gx * gyro_scale))),
        max(-32768, min(32767, round(gy * gyro_scale))),
        max(-32768, min(32767, round(gz * gyro_scale))),
    )
    for offset in (13, 25, 37):
        out[offset:offset + 12] = sample
    return bytes(out)


class SteamProHID:
    """Expose a Switch 2 Pro controller as a native Nintendo Pro HID device."""

    def __init__(
        self,
        mac: str,
        rumble_callback: Optional[Callable[[float, float], None]] = None,
    ):
        self.mac = mac.upper()
        self.rumble_callback = rumble_callback
        self._fd: Optional[int] = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._started = threading.Event()
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._reporter: Optional[threading.Thread] = None
        self._timer = 0
        self._state = (0, (0.0, 0.0), (0.0, 0.0), (0, 0, 4096), (0, 0, 0))

    def _write(self, data: bytes) -> None:
        fd = self._fd
        if fd is None:
            return
        with self._write_lock:
            written = os.write(fd, data)
            if written != len(data):
                raise OSError("short write to /dev/uhid")

    def _create_event(self) -> bytes:
        buf = bytearray(_UHID_EVENT_SIZE)
        struct.pack_into("<I", buf, 0, _UHID_CREATE2)
        buf[4:4 + 128] = b"Nintendo Switch Pro Controller".ljust(128, b"\0")
        buf[132:132 + 64] = f"ngc/{self.mac}".encode()[:63].ljust(64, b"\0")
        buf[196:196 + 64] = self.mac.encode()[:63].ljust(64, b"\0")
        struct.pack_into("<HHIIII", buf, 260, len(_REPORT_DESCRIPTOR), _BUS_BLUETOOTH,
                         _VENDOR_NINTENDO, _PRODUCT_SWITCH_PRO, 0x0210, 0)
        buf[280:280 + len(_REPORT_DESCRIPTOR)] = _REPORT_DESCRIPTOR
        return bytes(buf)

    def start(self) -> None:
        if not os.path.exists("/dev/uhid"):
            raise RuntimeError("/dev/uhid is missing; load the uhid kernel module")
        self._fd = os.open("/dev/uhid", os.O_RDWR | os.O_CLOEXEC)
        self._reader = threading.Thread(target=self._read_loop, name="ngc-steam-hid", daemon=True)
        self._reader.start()
        try:
            self._write(self._create_event())
            if not self._started.wait(3.0):
                raise RuntimeError("timed out waiting for UHID_START")
            self._reporter = threading.Thread(
                target=self._report_loop, name="ngc-steam-hid-reports", daemon=True
            )
            self._reporter.start()
            logger.info("Steam native Pro Controller HID ready for %s", self.mac)
        except Exception:
            self.close()
            raise

    def update(
        self,
        report: P.InputReport,
        left_stick: tuple[float, float],
        right_stick: tuple[float, float],
    ) -> None:
        with self._state_lock:
            self._state = (report.buttons, left_stick, right_stick, report.accel, report.gyro)

    def _next_report(self) -> bytes:
        with self._state_lock:
            state = self._state
        self._timer = (self._timer + 1) & 0xFF
        return _pack_input(*state, self._timer)

    def _send_input(self, report: bytes) -> None:
        if len(report) > _UHID_DATA_MAX:
            return
        event = struct.pack("<IH", _UHID_INPUT2, len(report)) + report
        self._write(event)

    def _report_loop(self) -> None:
        while not self._stop.wait(0.015):
            try:
                self._send_input(self._next_report())
            except OSError as exc:
                logger.warning("Steam HID input report failed: %s", exc)
                break

    def _read_loop(self) -> None:
        assert self._fd is not None
        poller = select.poll()
        poller.register(self._fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        while not self._stop.is_set():
            try:
                events = poller.poll(100)
                if not events:
                    continue
                if events[0][1] & (select.POLLERR | select.POLLHUP):
                    break
                event = os.read(self._fd, _UHID_EVENT_SIZE)
                if len(event) < 4:
                    continue
                kind = struct.unpack_from("<I", event)[0]
                if kind == _UHID_START:
                    self._started.set()
                elif kind == _UHID_STOP:
                    logger.debug("kernel stopped Steam HID device")
                elif kind == _UHID_OUTPUT:
                    size = struct.unpack_from("<H", event, 4 + _UHID_DATA_MAX)[0]
                    self._handle_output(event[4:4 + min(size, _UHID_DATA_MAX)])
                elif kind == _UHID_SET_REPORT:
                    request_id = struct.unpack_from("<I", event, 4)[0]
                    report_number = event[8]
                    size = struct.unpack_from("<H", event, 10)[0]
                    data = event[12:12 + min(size, _UHID_DATA_MAX)]
                    if report_number and (not data or data[0] != report_number):
                        data = bytes([report_number]) + data
                    self._handle_output(data)
                    self._write(struct.pack("<IIH", _UHID_SET_REPORT_REPLY, request_id, 0))
                elif kind == _UHID_GET_REPORT:
                    request_id = struct.unpack_from("<I", event, 4)[0]
                    self._write(struct.pack("<IIHH", _UHID_GET_REPORT_REPLY, request_id, errno.EIO, 0))
            except OSError as exc:
                if not self._stop.is_set():
                    logger.debug("Steam HID reader stopped: %s", exc)
                break

    def _handle_output(self, output: bytes) -> None:
        if not output:
            return
        reply = _switch_reply(output, self._timer)
        if reply is not None:
            try:
                self._send_input(reply)
            except OSError as exc:
                logger.debug("Switch Pro initialization reply failed: %s", exc)

        if output[0] not in (0x01, 0x10):
            return
        left = _rumble_frame(output, 2)
        right = _rumble_frame(output, 6)
        if left is None or right is None or self.rumble_callback is None:
            return
        low = max(left[0], right[0]) / 100.0
        high = max(left[1], right[1]) / 100.0
        self.rumble_callback(low, high)

    def close(self) -> None:
        self._stop.set()
        fd = self._fd
        if fd is not None:
            try:
                self._write(struct.pack("<I", _UHID_DESTROY))
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None
        for thread in (self._reporter, self._reader):
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=0.5)
        self._reporter = None
        self._reader = None
