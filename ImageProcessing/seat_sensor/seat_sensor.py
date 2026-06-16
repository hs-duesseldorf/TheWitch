from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from shared.events import ErrorEvent, PersonEvent, PersonTrigger, WitchEvent


logger = logging.getLogger(__name__)

LD2410S_BAUD = 115200
LD2410S_SERIAL_TIMEOUT_SECONDS = 0.005
LD2410S_POLL_SECONDS = 0.005
LD2410S_DISTANCE_SCALE = 1.0
LD2410S_DISTANCE_OFFSET_MM = 0
LD2410S_DEBUG_RAW = False
LD2410S_CONFIGURE = False
LD2410S_FAR_GATE = 5
LD2410S_NEAR_GATE = 0
LD2410S_HOLD_TIME_SECONDS = 10
LD2410S_REPORT_FREQUENCY_HZ = 80
LD2410S_RESPONSE_SPEED = 10
LD2410S_WARMUP_SECONDS = 10


@dataclass(frozen=True, slots=True)
class PresenceReading:
    present: bool
    distance_mm: int | None = None
    target_state: int = 0
    raw_frame: bytes | None = None


class PersonZone(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"
    SEATED = "seated"


class LD2410SSerialSensor:
    REPORT_HEADER = bytes([0xF4, 0xF3, 0xF2, 0xF1])
    REPORT_FOOTER = bytes([0xF8, 0xF7, 0xF6, 0xF5])
    COMMAND_HEADER = bytes([0xFD, 0xFC, 0xFB, 0xFA])
    COMMAND_FOOTER = bytes([0x04, 0x03, 0x02, 0x01])
    COMPACT_HEADER = 0x6E
    COMPACT_FOOTER = 0x62
    COMPACT_FRAME_LEN = 5

    def __init__(self, *, port: str, baud: int, timeout_s: float = 0.02):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "LD2410S support is missing. Install pyserial on the Jetson."
            ) from exc

        self._buffer = bytearray()
        self._serial = serial.Serial(port=port, baudrate=baud, timeout=timeout_s)

    def close(self) -> None:
        self._serial.close()

    def read_presence(self) -> PresenceReading | None:
        first = self._serial.read(1)
        if first:
            self._buffer.extend(first)
            while True:
                waiting = getattr(self._serial, "in_waiting", 0) or 0
                if waiting <= 0:
                    break
                self._buffer.extend(self._serial.read(min(256, waiting)))
        return self._pop_latest_reading()

    def configure_common(
        self,
        *,
        far_gate: int,
        near_gate: int,
        hold_time_s: int,
        status_frequency_hz: int,
        distance_frequency_hz: int,
        response_speed: int,
    ) -> bool:
        far_gate = min(max(far_gate, 1), 16)
        near_gate = min(max(near_gate, 0), 16)
        hold_time_s = min(max(hold_time_s, 10), 120)
        status_frequency_hz = min(max(status_frequency_hz, 5), 80)
        distance_frequency_hz = min(max(distance_frequency_hz, 5), 80)
        response_speed = 10 if response_speed >= 10 else 5

        self._drain_serial()
        if not self._send_command(
            0x00FF, b"\x01\x00", expected_command=0x01FF, timeout_s=0.5
        ):
            return False
        try:
            payload = b"".join(
                (
                    self._parameter(0x0005, far_gate),
                    self._parameter(0x000A, near_gate),
                    self._parameter(0x0006, hold_time_s),
                    self._parameter(0x0002, status_frequency_hz),
                    self._parameter(0x000C, distance_frequency_hz),
                    self._parameter(0x000B, response_speed),
                )
            )
            return self._send_command(
                0x0070, payload, expected_command=0x0170, timeout_s=1.0
            )
        finally:
            self._send_command(0x00FE, b"", expected_command=0x01FE, timeout_s=0.5)

    def _send_command(
        self, command: int, payload: bytes, *, expected_command: int, timeout_s: float
    ) -> bool:
        body = command.to_bytes(2, "little") + payload
        frame = (
            self.COMMAND_HEADER
            + len(body).to_bytes(2, "little")
            + body
            + self.COMMAND_FOOTER
        )
        self._serial.write(frame)
        self._serial.flush()

        deadline = time.monotonic() + timeout_s
        response = bytearray()
        while time.monotonic() < deadline:
            waiting = getattr(self._serial, "in_waiting", 0) or 0
            if waiting:
                response.extend(self._serial.read(waiting))
                start = response.find(self.COMMAND_HEADER)
                if start >= 0 and len(response) >= start + 10:
                    length = int.from_bytes(response[start + 4 : start + 6], "little")
                    frame_len = 4 + 2 + length + 4
                    if len(response) >= start + frame_len:
                        ack = bytes(response[start : start + frame_len])
                        if ack.endswith(self.COMMAND_FOOTER):
                            ack_command = int.from_bytes(ack[6:8], "little")
                            return ack_command == expected_command
            else:
                time.sleep(0.01)
        return False

    def _drain_serial(self) -> None:
        waiting = getattr(self._serial, "in_waiting", 0) or 0
        if waiting:
            self._serial.read(waiting)
        self._buffer.clear()

    def _parameter(self, word: int, value: int) -> bytes:
        return word.to_bytes(2, "little") + int(value).to_bytes(4, "little")

    def _pop_latest_reading(self) -> PresenceReading | None:
        latest = None
        while True:
            report_start = self._buffer.find(self.REPORT_HEADER)
            compact_start = self._buffer.find(bytes([self.COMPACT_HEADER]))
            starts = [start for start in (report_start, compact_start) if start >= 0]
            if not starts:
                del self._buffer[:-3]
                return latest
            start = min(starts)
            if start:
                del self._buffer[:start]
            if self._buffer[0] == self.COMPACT_HEADER:
                if len(self._buffer) < self.COMPACT_FRAME_LEN:
                    return latest
                frame = bytes(self._buffer[: self.COMPACT_FRAME_LEN])
                del self._buffer[: self.COMPACT_FRAME_LEN]
                parsed = self._parse_compact_payload(frame)
                if parsed is not None:
                    latest = parsed
                continue
            if len(self._buffer) < 8:
                return latest

            length = int.from_bytes(self._buffer[4:6], "little")
            frame_len = 4 + 2 + length + 4
            if len(self._buffer) < frame_len:
                return latest
            frame = bytes(self._buffer[:frame_len])
            del self._buffer[:frame_len]
            if not frame.endswith(self.REPORT_FOOTER):
                continue

            parsed = self._parse_report_payload(frame[6:-4])
            if parsed is not None:
                latest = parsed

    def _parse_compact_payload(self, frame: bytes) -> PresenceReading | None:
        if (
            len(frame) != self.COMPACT_FRAME_LEN
            or frame[0] != self.COMPACT_HEADER
            or frame[-1] != self.COMPACT_FOOTER
        ):
            return None
        target_state = frame[1]
        distance_cm = int.from_bytes(frame[2:4], "little")
        return PresenceReading(
            present=target_state != 0,
            distance_mm=distance_cm * 10 if distance_cm > 0 else None,
            target_state=target_state,
            raw_frame=frame,
        )

    def _parse_report_payload(self, payload: bytes) -> PresenceReading | None:
        if (
            len(payload) < 13
            or payload[0] != 0x02
            or payload[1] != 0xAA
            or payload[-2:] != b"\x55\x00"
        ):
            return None
        target_state = payload[2]
        moving_distance_cm = int.from_bytes(payload[3:5], "little")
        stationary_distance_cm = int.from_bytes(payload[6:8], "little")
        detection_distance_cm = int.from_bytes(payload[9:11], "little")

        present = target_state != 0
        distance_cm = (
            detection_distance_cm or stationary_distance_cm or moving_distance_cm
        )
        distance_mm = int(distance_cm * 10) if distance_cm > 0 else None
        return PresenceReading(
            present=present,
            distance_mm=distance_mm,
            target_state=target_state,
            raw_frame=payload,
        )


class SeatSensor:
    def __init__(self):
        self._callback: Callable[[WitchEvent], None] | None = None
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self, callback: Callable[[WitchEvent], None]) -> None:
        if self.worker is not None:
            return
        self._callback = callback
        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self._run, name="ld2410s-seat-monitor", daemon=True
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if (
            self.worker is not None
            and self.worker.is_alive()
            and self.worker is not threading.current_thread()
        ):
            self.worker.join(timeout=2.0)
        self.worker = None

    def _run(self) -> None:
        try:
            port = self._required_env("WITCH_LD2410S_PORT")
            seated_max_mm = int(self._required_env("WITCH_LD2410S_SEATED_MAX_MM"))
            present_max_mm = int(self._required_env("WITCH_LD2410S_ROOM_MAX_MM"))
        except ValueError as exc:
            self._publish_error(str(exc))
            return

        try:
            logger.info("Opening seat sensor on %s at %s baud", port, LD2410S_BAUD)
            sensor = LD2410SSerialSensor(
                port=port,
                baud=LD2410S_BAUD,
                timeout_s=LD2410S_SERIAL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Seat sensor open failed: %s", exc)
            return

        if LD2410S_CONFIGURE:
            configured = sensor.configure_common(
                far_gate=LD2410S_FAR_GATE,
                near_gate=LD2410S_NEAR_GATE,
                hold_time_s=LD2410S_HOLD_TIME_SECONDS,
                status_frequency_hz=LD2410S_REPORT_FREQUENCY_HZ,
                distance_frequency_hz=LD2410S_REPORT_FREQUENCY_HZ,
                response_speed=LD2410S_RESPONSE_SPEED,
            )
            if configured:
                logger.info(
                    "Configured LD2410S: far_gate=%s near_gate=%s hold_time_s=%s report_frequency_hz=%s response_speed=%s",
                    LD2410S_FAR_GATE,
                    LD2410S_NEAR_GATE,
                    LD2410S_HOLD_TIME_SECONDS,
                    LD2410S_REPORT_FREQUENCY_HZ,
                    LD2410S_RESPONSE_SPEED,
                )
            else:
                logger.warning(
                    "LD2410S configuration failed; continuing with existing sensor settings"
                )

        warmup_until = time.monotonic() + LD2410S_WARMUP_SECONDS
        if LD2410S_WARMUP_SECONDS > 0:
            logger.info(
                "Ignoring seat sensor readings for %.1f seconds during warmup",
                LD2410S_WARMUP_SECONDS,
            )

        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    reading = sensor.read_presence()
                except Exception as exc:
                    self._publish_error(f"LD2410S read failed: {exc}")
                    if self.stop_event.wait(LD2410S_POLL_SECONDS):
                        break
                    continue

                if reading is not None:
                    if time.monotonic() < warmup_until:
                        logger.debug(
                            "Seat sensor warmup ignored: present=%s distance_mm=%s target_state=%s",
                            reading.present,
                            reading.distance_mm,
                            reading.target_state,
                        )
                        reading = None
                    else:
                        distance_mm = self._calibrated_distance_mm(
                            reading.distance_mm,
                            scale=LD2410S_DISTANCE_SCALE,
                            offset_mm=LD2410S_DISTANCE_OFFSET_MM,
                        )
                        if LD2410S_DEBUG_RAW:
                            logger.info(
                                "LD2410S raw=%s target_state=%s present=%s distance_mm=%s",
                                reading.raw_frame.hex(" ")
                                if reading.raw_frame
                                else "-",
                                reading.target_state,
                                reading.present,
                                reading.distance_mm,
                        )
                        if not reading.present:
                            self._publish_zone(PersonZone.ABSENT, None)
                        elif distance_mm is None:
                            logger.debug(
                                "Ignoring seat sensor reading without distance: present=%s target_state=%s",
                                reading.present,
                                reading.target_state,
                            )
                            reading = None
                        else:
                            zone = self._ld2410s_zone(
                                PresenceReading(
                                    present=reading.present,
                                    distance_mm=distance_mm,
                                ),
                                seated_max_mm=seated_max_mm,
                                present_max_mm=present_max_mm,
                            )
                            self._publish_zone(zone, distance_mm)

                if reading is None:
                    elapsed = time.monotonic() - started
                    if self.stop_event.wait(max(0.0, LD2410S_POLL_SECONDS - elapsed)):
                        break
        finally:
            sensor.close()

    def _ld2410s_zone(
        self,
        reading: PresenceReading,
        *,
        seated_max_mm: int,
        present_max_mm: int,
    ) -> PersonZone:
        if not reading.present:
            return PersonZone.ABSENT
        if reading.distance_mm is None:
            return PersonZone.ABSENT
        if reading.distance_mm <= seated_max_mm:
            return PersonZone.SEATED
        if reading.distance_mm <= present_max_mm:
            return PersonZone.PRESENT
        return PersonZone.ABSENT

    def _publish_person_present(self, distance_mm: int | None) -> None:
        self._emit(
            PersonEvent(
                trigger=PersonTrigger.DETECTED,
            )
        )

    def _publish_person_seated(self, distance_mm: int | None) -> None:
        self._emit(
            PersonEvent(
                trigger=PersonTrigger.SEATED,
            )
        )

    def _publish_person_absent(self) -> None:
        self._emit(
            PersonEvent(
                trigger=PersonTrigger.ABSENT,
            )
        )

    def _publish_error(self, message: str) -> None:
        self._emit(
            ErrorEvent(
                message=f"Seat sensor error: {message}",
            )
        )

    def _publish_zone(self, zone: PersonZone, distance_mm: int | None) -> None:
        if zone is PersonZone.PRESENT:
            logger.debug("Seat sensor person present: distance_mm=%s", distance_mm)
            self._publish_person_present(distance_mm)
        elif zone is PersonZone.SEATED:
            logger.debug("Seat sensor person seated: distance_mm=%s", distance_mm)
            self._publish_person_seated(distance_mm)
        else:
            logger.debug("Seat sensor person absent")
            self._publish_person_absent()

    def _calibrated_distance_mm(
        self, distance_mm: int | None, *, scale: float, offset_mm: int
    ) -> int | None:
        if distance_mm is None:
            return None
        return max(0, int(round(distance_mm * scale + offset_mm)))

    def _required_env(self, name: str) -> str:
        value = os.environ[name]
        if not value.strip():
            raise ValueError(f"Missing required seat sensor env var: {name}")
        return value.strip()

    def _emit(self, event: WitchEvent) -> None:
        if self._callback is not None:
            self._callback(event)


SeatPresenceMonitor = SeatSensor
