from pytorch_lightning.callbacks import RichProgressBar
from rich.progress import Task, Progress
from abc import abstractmethod

class CustomRichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # training update is done by core framework

        self.dreaming_progress = None
        self.repeat_progress = None
        self.iteration_progress = None
        self.other_progress = dict()

    def _find_task(self, tid) -> Task:
        for t in self.progress.tasks:
            if(t.id == tid):
                return t

    # required fix for out of range.
    def _update(self, progress_bar_id: int, current: int, visible: bool = True) -> None:
        if self.progress is not None and self.is_enabled:
            total = self._find_task(progress_bar_id).total
            if not self._should_update(current, total):
                return

            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(progress_bar_id, advance=advance, visible=visible)
            self.refresh()

    def clear(self, key):
        if(self.other_progress.get(key) is not None):
            self.progress.remove_task(self.other_progress[key])
            self.other_progress[key] = None

    def setup_progress_bar(self, key, text:str, iterations):
        self.clear(key)
        self.other_progress[key] = self.progress.add_task(
            f"{text}\n", total=iterations
        )

    def update(self, key, advance_by=1):
        if(self.other_progress.get(key) is not None):
            self.progress.update(self.other_progress[key], advance=advance_by)

    @property
    def val_progress_bar(self) -> Task:
        return self._find_task(self.val_progress_bar_id)

    @property
    def val_sanity_check_bar(self) -> Task:
        return self._find_task(self.val_sanity_progress_bar_id)

    @property
    def main_progress_bar(self) -> Task:
        return self._find_task(self.main_progress_bar_id)

    @property
    def test_progress_bar(self) -> Task:
        return self._find_task(self.test_progress_bar_id)