"""Incremental serial reception; each valid block is delivered immediately."""

from dataclasses import dataclass
import logging
from threading import Event
from time import monotonic
from typing import Callable

from .models import Block, Frame
from .serial_protocol import SerialBlockParser
from .binary_protocol import PROTOCOL_VERSION
from .transport import BAUDRATES, DEFAULT_BAUDRATE, open_port
from .serial_diagnostics import SerialDiagnostics


@dataclass(frozen=True)
class Scan:
    frame: Frame
    expected_ids: tuple[int, ...]
    elapsed: float
    protocol_errors: int = 0

    @property
    def complete(self) -> bool:
        return {block.block_id for block in self.frame.blocks} == set(self.expected_ids)


class SerialAcquisition:
    def __init__(self, port: str, stop: Event, on_scan: Callable[[Scan], None],
                 on_status: Callable[[str], None], factory=None,
                 response_timeout: float = .8, quiet_time: float = .003,
                 clock=monotonic, on_block: Callable[[Block], None] | None = None,
                 mode: str = "poll", detect_timeout: float = 2.3,
                 protocol: str = "auto", baudrate: int = DEFAULT_BAUDRATE):
        if mode not in ("auto", "poll", "stream"):
            raise ValueError("采集模式必须为 auto / poll / stream")
        if type(baudrate) is not int or baudrate not in BAUDRATES:
            raise ValueError("波特率必须为 " + "、".join(map(str, BAUDRATES)))
        self.baudrate = baudrate
        self.port, self.stop = port, stop
        self.on_scan, self.on_status = on_scan, on_status
        self.on_block = on_block or (lambda block: None)
        self.factory = factory if factory is not None else lambda name: open_port(name, baudrate=self.baudrate)
        self.clock = clock
        self.response_timeout, self.quiet_time = response_timeout, quiet_time
        self.mode, self.detect_timeout = mode, detect_timeout
        self.active_mode = None
        self.parser = SerialBlockParser(protocol, clock=clock)
        self.detected_ids = set()
        self.diagnostics = SerialDiagnostics(port, clock)
        self.request_failures = {}

    def _read(self, serial_port):
        # read(256) waits for 256 bytes or timeout even if a short reply is ready.
        # Block only for the first byte, then consume bytes already available.
        data = serial_port.read(max(1, min(serial_port.in_waiting, 4096)))
        self.diagnostics.received(data)
        return data

    def _feed(self, data):
        blocks = self.parser.feed(data)
        self.diagnostics.parsed(blocks, self.parser)
        if not self.stop.is_set():
            self.diagnostics.tick(self.parser, self.active_mode or "detect")
        return blocks

    def _settle(self, serial_port, timeout=False, end_before=None):
        """Consume optional END; retain a longer quiet guard only after failure."""
        quiet = max(self.quiet_time, .06) if timeout else self.quiet_time
        now = self.clock()
        last_data, deadline = now, now + (max(.3, quiet * 4) if timeout else .05)
        while not self.stop.is_set() and self.clock() < deadline:
            if end_before is not None and self.parser.ends > end_before:
                # Documented device direction change is 150 us after END.
                self.stop.wait(.001)
                return
            data = self._read(serial_port)
            if data:
                last_data = self.clock()
                discarded = self._feed(data)
                if discarded:
                    self.diagnostics.limited_log("settle_data", "等待总线安静时收到额外数据，未送入绘图",
                                                 block_ids=[b.block_id for b in discarded])
            elif self.clock() - last_data >= quiet:
                self.parser.reset()
                return
        self.parser.reset()
        if timeout and not self.stop.is_set():
            self.diagnostics.log(logging.ERROR, "等待总线安静失败，总线持续发送", guard_seconds=quiet)
            raise RuntimeError("总线持续发送；请使用连续接收模式")

    def _request(self, serial_port, block_id):
        if self.stop.is_set():
            return None
        self.parser.reset()
        end_before = self.parser.ends
        command = f"hello num {block_id}\r\n".encode("ascii")
        written = serial_port.write(command)
        self.diagnostics.tx_bytes += written or 0
        if written != len(command):
            self.diagnostics.log(logging.ERROR, "采集命令未完整写入", block_id=block_id, expected=len(command), written=written)
            raise OSError("串口指令未完整写入")
        request_bytes = self.diagnostics.byte_count
        request_errors = self.parser.errors
        deadline = self.clock() + self.response_timeout
        while not self.stop.is_set() and self.clock() < deadline:
            for block in self._feed(self._read(serial_port)):
                if block.block_id == block_id:
                    if self.request_failures.pop(block_id, 0):
                        self.diagnostics.log(logging.INFO, "轮询应答恢复", block_id=block_id)
                    self.detected_ids.add(block_id)
                    self.on_block(block)
                    self._settle(serial_port, end_before=end_before)
                    return block
                self.diagnostics.limited_log(("wrong_id", block_id), "轮询收到非目标块应答",
                                             expected_id=block_id, received_id=block.block_id)
            if block_id in self.parser.not_ready_ids:
                self.diagnostics.limited_log(("not_ready", block_id), "皮肤块返回未就绪", block_id=block_id)
                self.detected_ids.add(block_id)
                self._settle(serial_port, end_before=end_before)
                return None
        if self.stop.is_set():
            return None
        self.request_failures[block_id] = self.request_failures.get(block_id, 0) + 1
        self.diagnostics.limited_log(("timeout", block_id), "轮询应答超时",
                                     block_id=block_id, known_device=block_id in self.detected_ids,
                                     consecutive_timeouts=self.request_failures[block_id],
                                     timeout_seconds=self.response_timeout,
                                     rx_bytes_during_request=self.diagnostics.byte_count - request_bytes,
                                     parse_errors_during_request=self.parser.errors - request_errors,
                                     pending_binary_bytes=len(self.parser.buffer),
                                     pending_text_bytes=len(self.parser.text.buffer),
                                     incomplete_text_block=self.parser.text.current,
                                     command=command.decode("ascii").strip(),
                                     recent_rx_hex=self.diagnostics.recent.hex(" "))
        self._settle(serial_port, timeout=True)
        return None

    def _poll(self, serial_port):
        sequence, active_ids = 0, ()
        while not self.stop.is_set():
            discovery = not active_ids
            ids = tuple(range(1, 17)) if discovery else active_ids
            blocks = []
            started = self.clock()
            for block_id in ids:
                if self.stop.is_set():
                    break
                if discovery:
                    self.on_status(f"检测皮肤块 {block_id}/16 · 已发现 {len(blocks)} 块；已收数据立即送往显示处理")
                block = self._request(serial_port, block_id)
                if block is not None:
                    blocks.append(block)
            if self.stop.is_set():
                break
            if discovery:
                active_ids = tuple(sorted(self.detected_ids))
                self.diagnostics.log(logging.INFO, "轮询检测完成", detected_ids=active_ids)
                if not active_ids:
                    self.on_status("没有收到有效 BLOCK，应检查接线、串口和固件；正在重试")
                    self.stop.wait(.5)
                    continue
                self.on_status(f"指令轮询 · 已识别 {len(active_ids)} 块，各块收到即更新")
            sequence += 1
            self.on_scan(Scan(Frame(tuple(blocks), sequence), active_ids,
                              self.clock() - started, self.parser.errors))

    def _stream(self, serial_port, initial=()):
        """Grouping is diagnostic only; on_block precedes any frame-end wait."""
        pending, known = {}, set()
        sequence = 0
        started = last_data = self.clock()

        def publish():
            nonlocal sequence, started
            sequence += 1
            self.on_scan(Scan(Frame(tuple(pending.values()), sequence), tuple(sorted(known)),
                              self.clock() - started, self.parser.errors))
            pending.clear()
            started = self.clock()

        while not self.stop.is_set():
            blocks = initial
            initial = ()
            if not blocks:
                data = self._read(serial_port)
                if data:
                    last_data = self.clock()
                blocks = self._feed(data)
            for block in blocks:
                if block.block_id in pending:
                    publish()
                known.add(block.block_id)
                pending[block.block_id] = block
                self.on_block(block)
            if pending and self.clock() - last_data >= .1:
                publish()

    def run(self):
        serial_port = None
        failed = False
        self.diagnostics.log(logging.INFO, "开始串口连接", baud=self.baudrate, serial_format="8N1", mode=self.mode,
                             protocol=self.parser.mode, binary_version=PROTOCOL_VERSION, binary_tail="0x0A",
                             response_timeout=self.response_timeout, frame_timeout=self.parser.frame_timeout)
        try:
            serial_port = self.factory(self.port)
            self.diagnostics.log(logging.INFO, "串口打开成功")
            self.on_status(f"串口已打开 · {self.baudrate} bps，请保持无按压，约3秒自动标定")
            mode = self.mode
            initial = ()
            if mode == "auto":
                self.on_status("自动识别发送方式：先监听连续数据（最多 2.3 秒）")
                deadline = self.clock() + self.detect_timeout
                while not self.stop.is_set() and self.clock() < deadline:
                    initial = self._feed(self._read(serial_port))
                    if initial or self.parser.not_ready_ids:
                        break
                mode = "stream" if initial or self.parser.not_ready_ids else "poll"
            if self.stop.is_set():
                return
            self.active_mode = mode
            self.diagnostics.log(logging.INFO, "确定发送方式", requested=self.mode, active=mode,
                                 formats=self.parser.counts)
            if mode == "stream":
                self.on_status("连续接收 · 收到哪块就更新哪块")
                self._stream(serial_port, initial)
            else:
                serial_port.reset_input_buffer()
                self.diagnostics.log(logging.INFO, "进入轮询前清空接收缓冲")
                self.parser.reset()
                self._poll(serial_port)
        except Exception:
            failed = True
            logging.getLogger(__name__).exception("串口采集异常 session=%s port=%s active_mode=%s",
                                                  self.diagnostics.session, self.port, self.active_mode)
            raise
        finally:
            self.diagnostics.summary(self.parser, self.active_mode)
            if serial_port is not None:
                try:
                    serial_port.close()
                    self.diagnostics.log(logging.INFO, "串口已关闭")
                except Exception:
                    logging.getLogger(__name__).exception("关闭串口失败 session=%s port=%s",
                                                          self.diagnostics.session, self.port)
                    if not failed:
                        raise
            self.diagnostics.log(logging.INFO, "采集会话结束", reason="exception" if failed else
                                 "stop_requested" if self.stop.is_set() else "returned")
