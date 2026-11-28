"""Review who actually appeared in recorded authentication attempts."""

from datetime import datetime
import sqlite3
import tkinter as tk
from tkinter import ttk

from auth_calibration import MIN_CONFIRMED


OUTCOMES = {'accepted': '승인', 'rejected': '거절', 'interrupted': '중단', 'error': '오류', None: '진행 중'}


def policy_text(policy):
    counts = f"본인 {policy['genuine']}/{MIN_CONFIRMED} · 타인 {policy['impostor']}/{MIN_CONFIRMED}"
    if policy['threshold'] is None:
        return f'{counts} · 기본 기준 사용'
    point = f"교차점 {policy['threshold']:.3f} · 오승인 {policy['far']:.1%} · 오거부 {policy['frr']:.1%}"
    return f"{point} · {'다음 인증부터 적용' if policy['enabled'] else '자동 적용 꺼짐'}"


class AuthenticationReview:
    def __init__(self, root, store, on_change):
        self.store, self.calibration, self.on_change = store, store.calibration, on_change
        self.window = tk.Toplevel(root)
        self.window.title('인증 시도 확인 · 유사도 기준')
        self.window.geometry('820x530')
        content = ttk.Frame(self.window, padding=14)
        content.pack(fill='both', expand=True)
        ttk.Label(content, text='기록을 선택하고, 승인·거절과 별도로 확인한 실제 인물을 지정하세요.').pack(anchor='w')
        self.policy = tk.StringVar()
        ttk.Label(content, textvariable=self.policy).pack(anchor='w', pady=(4, 8))
        actions = ttk.Frame(content)
        actions.pack(fill='x', pady=(0, 6))
        self.enabled = tk.BooleanVar()
        ttk.Checkbutton(actions, text='교차점 자동 적용', variable=self.enabled, command=self._toggle).pack(side='left')
        ttk.Button(actions, text='새로고침', command=self.refresh).pack(side='right')
        listing = ttk.Frame(content)
        listing.pack(fill='both', expand=True)
        columns = ('id', 'time', 'outcome', 'best', 'threshold', 'verified')
        self.table = ttk.Treeview(listing, columns=columns, show='headings', selectmode='browse')
        for key, title, width in zip(columns, ('번호', '시각', '결과', '최고 정면 점수', '정면 기준', '실제 인물 확인'),
                                     (55, 130, 95, 160, 105, 165)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=45, stretch=key in ('best', 'verified'))
        scroll = ttk.Scrollbar(listing, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.table.pack(fill='both', expand=True)
        self.table.bind('<<TreeviewSelect>>', self._select)
        self.detail = tk.StringVar(value='인증 시도를 선택하세요.')
        ttk.Label(content, textvariable=self.detail, wraplength=785).pack(anchor='w', pady=6)
        review = ttk.Frame(content)
        review.pack(fill='x')
        self.person = ttk.Combobox(review, state='readonly', width=19)
        self.person.pack(side='left', padx=(0, 4))
        ttk.Button(review, text='선택한 등록자로 확인', command=lambda: self._save('registered')).pack(side='left')
        ttk.Button(review, text='미등록 타인으로 확인', command=lambda: self._save('visitor')).pack(side='left', padx=4)
        ttk.Button(review, text='확인 취소', command=lambda: self._save(None)).pack(side='left')
        self.refresh()

    def refresh(self):
        selection = self.table.selection()
        self.attempts = {str(row['id']): row for row in self.calibration.recent_attempts()}
        self.table.delete(*self.table.get_children())
        for key, row in self.attempts.items():
            scores = row['score_candidates']
            best = '점수 없음'
            if scores:
                name = max(scores, key=scores.get)
                best = f'{name} · {scores[name]:.3f}'
            verified = row['verified_name'] or ('미등록 타인' if row['verified_kind'] == 'visitor' else '확인 대기')
            when = datetime.fromisoformat(row['started_at']).astimezone().strftime('%m-%d %H:%M:%S')
            threshold = f"{row['threshold']:.3f} · {'교차점' if row['threshold_source'] == 'eer' else '기본'}"
            outcome = OUTCOMES[row['outcome']] + (' · 진단' if row['diagnostic'] else '')
            self.table.insert('', 'end', iid=key, values=(key, when, outcome, best, threshold, verified))
        policy = self.calibration.status()
        self.policy.set(policy_text(policy))
        self.enabled.set(policy['enabled'])
        if selection and selection[0] in self.attempts:
            self.table.selection_set(selection[0])
        self._select()

    def _select(self, event=None):
        selection = self.table.selection()
        row = self.attempts.get(selection[0]) if selection else None
        if row is None:
            self.detail.set('인증 시도를 선택하세요.')
            self.person.configure(values=())
            self.person.set('')
            return
        scores = row['score_candidates']
        self.person.configure(values=sorted(scores or self.store.people))
        self.person.set(row['verified_name'] or '')
        score_text = ', '.join(f'{name} {score:.3f}' for name, score in sorted(scores.items()))
        self.detail.set(f"#{row['id']} · {row['reason'] or '진행 중'}\n{score_text or '정면 유사도 점수 없음'}")

    def _save(self, kind):
        selection = self.table.selection()
        if not selection:
            self.detail.set('확인할 인증 시도를 선택하세요.')
            return
        try:
            self.calibration.review(int(selection[0]), kind, self.person.get() if kind == 'registered' else None)
            self.refresh()
            self.on_change()
        except (sqlite3.Error, ValueError) as exc:
            self.detail.set(str(exc))

    def _toggle(self):
        try:
            self.calibration.set_enabled(self.enabled.get())
            self.refresh()
            self.on_change()
        except sqlite3.Error as exc:
            self.detail.set(str(exc))
