from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
import logging

from .transport import WebSocketClient
from shared.events import ErrorEvent, PersonEvent, PersonTrigger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SeatSensorConfig:
    ld2410s_baud: int = 115200
    serial_timeout_ms: int = 5
    poll_ms: int = 5
    required_hits: int = 2
    stale_ms: int = 500
    far_gate: int = 5
    near_gate: int = 0
    hold_time_s: int = 10
    report_frequency_hz: int = 80
    response_speed: int = 10


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
            raise RuntimeError("LD2410S support is missing. Install pyserial on the Jetson.") from exc

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
        if not self._send_command(0x00FF, b"\x01\x00", expected_command=0x01FF, timeout_s=0.5):
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
            return self._send_command(0x0070, payload, expected_command=0x0170, timeout_s=1.0)
        finally:
            self._send_command(0x00FE, b"", expected_command=0x01FE, timeout_s=0.5)

    def _send_command(self, command: int, payload: bytes, *, expected_command: int, timeout_s: float) -> bool:
        body = command.to_bytes(2, "little") + payload
        frame = self.COMMAND_HEADER + len(body).to_bytes(2, "little") + body + self.COMMAND_FOOTER
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
                frame = bytes(self._buffer[:self.COMPACT_FRAME_LEN])
                del self._buffer[:self.COMPACT_FRAME_LEN]
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
        if len(payload) < 13 or payload[0] != 0x02 or payload[1] != 0xAA or payload[-2:] != b"\x55\x00":
            return None
        target_state = payload[2]
        moving_distance_cm = int.from_bytes(payload[3:5], "little")
        stationary_distance_cm = int.from_bytes(payload[6:8], "little")
        detection_distance_cm = int.from_bytes(payload[9:11], "little")

        present = target_state != 0
        distance_cm = detection_distance_cm or stationary_distance_cm or moving_distance_cm
        distance_mm = int(distance_cm * 10) if distance_cm > 0 else None
        return PresenceReading(
            present=present,
            distance_mm=distance_mm,
            target_state=target_state,
            raw_frame=payload,
        )


class SeatPresenceMonitor:
    def __init__(
        self,
        *,
        event_client: WebSocketClient,
        config: SeatSensorConfig | None = None,
    ):
        self.config = config or SeatSensorConfig()
        self.event_client = event_client
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self) -> None:
        if self.worker is not None:
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, name="ld2410s-seat-monitor", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=2.0)
        self.worker = None

    def _run(self) -> None:
        if os.getenv("WITCH_SEAT_SENSOR_OVERRIDE", "false") == "true":
            logger.info("Seat sensor override enabled; publishing detected and seated person")
            self._publish_person_present(None)
            self._publish_person_seated(None)
            return

        try:
            port = self._required_env("WITCH_LD2410S_PORT")
            seated_max_mm = int(self._required_env("WITCH_LD2410S_SEATED_MAX_MM"))
            present_max_mm = int(self._required_env("WITCH_LD2410S_ROOM_MAX_MM"))
        except ValueError as exc:
            self._publish_error(str(exc))
            return

        baud = int(os.getenv("WITCH_LD2410S_BAUD", str(self.config.ld2410s_baud)))
        serial_timeout_s = max(
            float(os.getenv("WITCH_LD2410S_SERIAL_TIMEOUT_MS", str(self.config.serial_timeout_ms))) / 1000.0,
            0.001,
        )
        poll_s = max(self.config.poll_ms / 1000.0, 0.001)
        stale_s = max(float(os.getenv("WITCH_LD2410S_STALE_MS", str(self.config.stale_ms))) / 1000.0, poll_s)
        required_hits = max(1, int(os.getenv("WITCH_LD2410S_REQUIRED_HITS", str(self.config.required_hits))))
        distance_scale = float(os.getenv("WITCH_LD2410S_DISTANCE_SCALE", "1.0"))
        distance_offset_mm = int(os.getenv("WITCH_LD2410S_DISTANCE_OFFSET_MM", "0"))
        debug_raw = os.getenv("WITCH_LD2410S_DEBUG_RAW", "false") == "true"
        configure_sensor = os.getenv("WITCH_LD2410S_CONFIGURE", "false") == "true"
        far_gate = int(os.getenv("WITCH_LD2410S_FAR_GATE", str(self.config.far_gate)))
        near_gate = int(os.getenv("WITCH_LD2410S_NEAR_GATE", str(self.config.near_gate)))
        hold_time_s = int(os.getenv("WITCH_LD2410S_HOLD_TIME_S", str(self.config.hold_time_s)))
        report_frequency_hz = int(os.getenv("WITCH_LD2410S_REPORT_FREQUENCY_HZ", str(self.config.report_frequency_hz)))
        response_speed = int(os.getenv("WITCH_LD2410S_RESPONSE_SPEED", str(self.config.response_speed)))

        try:
            logger.info("Opening seat sensor on %s at %s baud", port, baud)
            sensor = LD2410SSerialSensor(port=port, baud=baud, timeout_s=serial_timeout_s)
        except Exception as exc:
            logger.warning("Seat sensor open failed: %s", exc)
            self._publish_error(str(exc))
            return

        if configure_sensor:
            configured = sensor.configure_common(
                far_gate=far_gate,
                near_gate=near_gate,
                hold_time_s=hold_time_s,
                status_frequency_hz=report_frequency_hz,
                distance_frequency_hz=report_frequency_hz,
                response_speed=response_speed,
            )
            if configured:
                logger.info(
                    "Configured LD2410S: far_gate=%s near_gate=%s hold_time_s=%s report_frequency_hz=%s response_speed=%s",
                    far_gate,
                    near_gate,
                    max(hold_time_s, 10),
                    report_frequency_hz,
                    response_speed,
                )
            else:
                logger.warning("LD2410S configuration failed; continuing with existing sensor settings")

        zone = PersonZone.ABSENT
        candidate_zone = PersonZone.ABSENT
        candidate_hits = 0
        last_reading_at = 0.0
        last_distance_mm: int | None = None

        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    reading = sensor.read_presence()
                except Exception as exc:
                    self._publish_error(f"LD2410S read failed: {exc}")
                    if self.stop_event.wait(poll_s):
                        break
                    continue

                if reading is not None:
                    now = time.monotonic()
                    last_reading_at = now
                    distance_mm = self._calibrated_distance_mm(
                        reading.distance_mm,
                        scale=distance_scale,
                        offset_mm=distance_offset_mm,
                    )
                    reading = PresenceReading(
                        present=reading.present,
                        distance_mm=distance_mm,
                        target_state=reading.target_state,
                        raw_frame=reading.raw_frame,
                    )
                    if debug_raw:
                        logger.info(
                            "LD2410S raw=%s target_state=%s present=%s distance_mm=%s",
                            reading.raw_frame.hex(" ") if reading.raw_frame else "-",
                            reading.target_state,
                            reading.present,
                            reading.distance_mm,
                        )
                    last_distance_mm = reading.distance_mm

                    next_zone = self._ld2410s_zone(
                        reading,
                        seated_max_mm=seated_max_mm,
                        present_max_mm=present_max_mm,
                    )

                    if next_zone == candidate_zone:
                        candidate_hits += 1
                    else:
                        candidate_zone = next_zone
                        candidate_hits = 1

                    if candidate_zone != zone and candidate_hits >= required_hits:
                        previous_zone = zone
                        zone = candidate_zone
                        if previous_zone is PersonZone.ABSENT and zone is PersonZone.SEATED:
                            self._publish_zone(PersonZone.PRESENT, last_distance_mm)
                        self._publish_zone(zone, last_distance_mm if zone is not PersonZone.ABSENT else None)

                now = time.monotonic()
                if zone is not PersonZone.ABSENT and last_reading_at > 0 and now - last_reading_at >= stale_s:
                    zone = PersonZone.ABSENT
                    candidate_zone = PersonZone.ABSENT
                    candidate_hits = 0
                    last_distance_mm = None
                    self._publish_zone(zone, None)

                if reading is None:
                    elapsed = time.monotonic() - started
                    if self.stop_event.wait(max(0.0, poll_s - elapsed)):
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
            return PersonZone.PRESENT
        if reading.distance_mm <= seated_max_mm:
            return PersonZone.SEATED
        if reading.distance_mm <= present_max_mm:
            return PersonZone.PRESENT
        return PersonZone.ABSENT

    def _publish_person_present(self, distance_mm: int | None) -> None:
        self.event_client.send_message(
            PersonEvent(
                trigger=PersonTrigger.DETECTED,
            )
        )

    def _publish_person_seated(self, distance_mm: int | None) -> None:
        self.event_client.send_message(
            PersonEvent(
                trigger=PersonTrigger.SEATED,
            )
        )

    def _publish_person_absent(self) -> None:
        self.event_client.send_message(
            PersonEvent(
                trigger=PersonTrigger.ABSENT,
            )
        )

    def _publish_error(self, message: str) -> None:
        self.event_client.send_message(
            ErrorEvent(
                message=f"Seat sensor error: {message}",
            )
        )

    def _publish_zone(self, zone: PersonZone, distance_mm: int | None) -> None:
        if zone is PersonZone.PRESENT:
            logger.info("Seat sensor person present: distance_mm=%s", distance_mm)
            self._publish_person_present(distance_mm)
        elif zone is PersonZone.SEATED:
            logger.info("Seat sensor person seated: distance_mm=%s", distance_mm)
            self._publish_person_seated(distance_mm)
        else:
            logger.info("Seat sensor person absent")
            self._publish_person_absent()

    def _calibrated_distance_mm(self, distance_mm: int | None, *, scale: float, offset_mm: int) -> int | None:
        if distance_mm is None:
            return None
        return max(0, int(round(distance_mm * scale + offset_mm)))

    def _required_env(self, name: str) -> str:
        value = os.getenv(name)
        if value is None or not value.strip():
            raise ValueError(f"Missing required seat sensor env var: {name}")
        return value.strip()
