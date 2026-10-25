"""Recording bar above the heatmap: identity fields, boundary buttons, live counts.

Widgets only; MainWindow owns the policy (what may be recorded, what gets locked).
The bar shows the fields the CSV plan requires: subject, session, action, trial,
duration, per-block frames and the save state.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QMenu,
                               QPushButton, QSizePolicy, QToolButton, QVBoxLayout)

from .control_panel import label
from ..collection_quality import QUALITY_NAMES


class CollectionBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 7, 16, 7)
        box.setSpacing(5)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("训练采集", "section"))
        row.addSpacing(6)
        row.addWidget(label("受试者"))
        self.subject = QComboBox()
        self.subject.setEditable(True)
        self.subject.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.subject.lineEdit().setPlaceholderText("student01")
        self.subject.lineEdit().setMaxLength(64)
        self.subject.setFixedWidth(115)
        self.subject.setToolTip("受试者编号；字母、数字、下划线或连字符。操作人员与受试者不是同一概念。")
        row.addWidget(self.subject)
        self.add_subject, self.delete_subject = self._identity_buttons(row, "受试者")
        row.addWidget(label("场次"))
        self.session = QComboBox()
        self.session.setEditable(True)
        self.session.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.session.lineEdit().setPlaceholderText("session_YYYYMMDD_01")
        self.session.lineEdit().setMaxLength(64)
        self.session.setFixedWidth(180)
        self.session.setToolTip("同一场次内重复采集同一动作时次数自动递增；更换皮肤或条件明显变化时新建场次。")
        row.addWidget(self.session)
        self.add_session, self.delete_session = self._identity_buttons(row, "场次")
        # 身份管理单独成行，窗口缩到最小宽度时按钮也不会覆盖编号。
        row.addStretch(1)
        box.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("动作"))
        self.action = QComboBox()
        self.action.setMinimumWidth(125)
        self.action.setToolTip("动作编号全数据集固定，界面排序变化不会改变编号。")
        row.addWidget(self.action)
        row.addWidget(label("次数"))
        self.trial = label("—", "section")
        self.trial.setMinimumWidth(62)
        self.trial.setToolTip("本次样本的编号；已存在的样本绝不覆盖，作废重采使用新编号。")
        row.addWidget(self.trial)
        row.addStretch(1)
        self.start = QPushButton("开始采集")
        self.start.setObjectName("primary")
        self.start.setToolTip("录制原始 ADC 到数据集；开始前需连接硬件并固定采集布局。")
        self.stop = QPushButton("结束采集")
        self.stop.setEnabled(False)
        self.stop.setToolTip("关闭本次样本的录制边界，不必断开串口；结束后先完成写盘。")
        row.addWidget(self.start)
        row.addWidget(self.stop)
        box.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.duration = label("时长 0.0 秒")
        self.duration.setMinimumWidth(96)
        row.addWidget(self.duration)
        self.records = label("记录 0")
        self.records.setMinimumWidth(74)
        self.records.setToolTip("写入 raw.csv 的数据行数；一行是某一块皮肤新收到的一帧，不是全身同步帧。")
        row.addWidget(self.records)
        self.frame_button = QToolButton()
        self.frame_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.frame_button.setToolTip("展开查看本次录制各块已收到的帧数；预期块缺失时整段样本标为未完成。")
        self.frame_menu = QMenu(self.frame_button)
        self.frame_menu.setToolTipsVisible(True)
        self.frame_button.setMenu(self.frame_menu)
        row.addWidget(self.frame_button)
        # 状态信息允许收缩；操作按钮始终完整可见。
        self.status = label("未开始")
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.status, 1)
        self.accepted = label("未判定")
        self.accepted.setMinimumWidth(58)
        row.addWidget(self.accepted)
        self.mark_valid = QPushButton("标记有效")
        self.mark_valid.setEnabled(False)
        self.mark_valid.setToolTip("把刚结束的样本标为 accepted=true，可进入训练集合。")
        self.mark_void = QPushButton("作废")
        self.mark_void.setEnabled(False)
        self.mark_void.setToolTip("将刚结束的样本移出正式数据集和索引，放入本地回收目录，可通过“恢复删除”找回。")
        row.addWidget(self.mark_valid)
        row.addWidget(self.mark_void)
        self.dataset_button = QPushButton("数据集目录")
        self.dataset_button.setToolTip("打开当前数据集根目录；index.csv 与各样本 meta.json 可在此查看。")
        row.addWidget(self.dataset_button)
        self.restore_identity = QPushButton("恢复删除")
        self.restore_identity.setToolTip("从本地回收目录选择受试者、场次或作废样本，恢复到原路径；不会覆盖同名目录。")
        row.addWidget(self.restore_identity)
        box.addLayout(row)

        self._frame_keys = ()
        self._frame_actions = {}
        self._set_frame_counts({})

    @staticmethod
    def _identity_buttons(row, title):
        """紧凑按钮配合下拉框，删除含义在提示中明确为可恢复回收。"""
        add, delete = QToolButton(), QToolButton()
        add.setText("＋")
        delete.setText("−")
        add.setAccessibleName(f"新增{title}")
        delete.setAccessibleName(f"删除{title}")
        add.setToolTip(f"新增{title}")
        delete.setToolTip(f"删除当前{title}：关联数据移入本地回收目录，已有编号不变")
        row.addWidget(add)
        row.addWidget(delete)
        return add, delete

    def set_actions(self, classes):
        """Populate the fixed action list; item data stays the numeric action id."""
        self.action.clear()
        for item in classes:
            self.action.addItem(f"{item.display_name}（{item.name}）", item.id)
            # 下拉项展示采集标准，帮助操作者区分短暂和持续接触。
            description = "\n".join(text for text in (item.definition, item.contact_features) if text)
            self.action.setItemData(self.action.count() - 1, description, Qt.ItemDataRole.ToolTipRole)

    def action_id(self):
        return self.action.currentData()

    def set_trial(self, trial_id, hint=""):
        self.trial.setText(trial_id or "—")
        self.trial.setToolTip(hint or "本次样本的编号；已存在的样本绝不覆盖，作废重采使用新编号。")

    def set_recording(self, active):
        """Identity fields and the boundary buttons follow the recording state."""
        for widget in (self.subject, self.session, self.action, self.add_subject,
                       self.delete_subject, self.add_session, self.delete_session):
            widget.setEnabled(not active)
        self.restore_identity.setEnabled(not active)
        self.start.setEnabled(not active)
        self.stop.setEnabled(active)

    def set_status(self, text, level="idle"):
        self.status.setText(text)
        self.status.setToolTip(text)
        colors = {"idle": "", "ok": "color: #147d83;", "warn": "color: #b45309;",
                  "error": "color: #b91c1c;"}
        self.status.setStyleSheet(colors.get(level, ""))

    def set_stats(self, duration, records, frames, missing=()):
        """Refresh counters; the menu is rebuilt only when the block set changes."""
        self.duration.setText(f"时长 {duration:.1f} 秒")
        self.records.setText(f"记录 {records}")
        self._set_frame_counts(frames)
        missing = tuple(missing)
        if not frames:
            self.frame_button.setText("各块帧数")
        elif missing:
            self.frame_button.setText(f"各块 {min(frames.values())}–{max(frames.values())} 帧 · 缺 {len(missing)} 块")
        else:
            low, high = min(frames.values()), max(frames.values())
            self.frame_button.setText(f"各块 {low} 帧" if low == high else f"各块 {low}–{high} 帧")
        self.frame_button.setEnabled(bool(frames))

    def _set_frame_counts(self, frames):
        frames = dict(sorted(frames.items()))
        keys = tuple(frames)
        if keys != self._frame_keys:
            self.frame_menu.clear()
            self._frame_actions = {key: self.frame_menu.addAction("") for key in keys}
            self._frame_keys = keys
        for key, count in frames.items():
            self._frame_actions[key].setText(f"BLOCK {key:02d}    {count} 帧")

    def set_sample_state(self, text, level="idle", *, can_accept=False, can_void=False):
        """Show the quality judgement; only finished samples may be marked."""
        self.accepted.setText(text)
        self.accepted.setStyleSheet({"warn": "color: #b45309;", "error": "color: #b91c1c;",
                                     "ok": "color: #147d83;"}.get(level, ""))
        self.mark_valid.setEnabled(can_accept)
        self.mark_void.setEnabled(can_void)

    def set_integrity(self, report):
        """逐块展示实时空缺；包含零帧块，且将历史异常保留到结束。"""
        blocks = report.get("blocks", {})
        kinds = {"head_gap": "首帧迟到", "interval_gap": "中间空缺", "tail_gap": "结尾缺失",
                 "ongoing_gap": "持续静默", "no_data": "未收到数据"}
        details = {int(key): [] for key in blocks}
        for event in report.get("events", []):
            details[event["block_id"]].append(
                f"{kinds[event['kind']]}：{event['start_s']:.3f}～{event['end_s']:.3f} 秒"
                f"（持续 {event['duration_s']:.3f} 秒）")
        self._set_frame_counts({int(key): value["frames"] for key, value in blocks.items()})
        for key, value in blocks.items():
            self._frame_actions[int(key)].setText(
                f"BLOCK {int(key):02d} · {value['frames']} 帧 · {value['message']}")
            # 悬停该块可定位异常区间；明细截断时元数据仍保留总次数和最严重状态。
            self._frame_actions[int(key)].setToolTip("\n".join(details[int(key)]) or value["message"])
        state = report.get("status", "pending")
        text = QUALITY_NAMES[state] if report.get("final") else ("接收正常" if state == "pass" else QUALITY_NAMES[state])
        if not report.get("final") and state == "pass" and any(not item["frames"] for item in blocks.values()):
            text = "等待首帧"
        self.frame_button.setText("各块 · " + text)
        self.frame_button.setEnabled(bool(blocks))
        self.frame_button.setToolTip(("\n".join(report.get("reasons", [])) + "\n展开后悬停各块查看异常时间区间").strip())
        self.frame_button.setStyleSheet({"fail": "color: #b91c1c", "warn": "color: #b45309"}.get(state, ""))
