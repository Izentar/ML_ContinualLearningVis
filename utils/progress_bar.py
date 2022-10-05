from pytorch_lightning.callbacks import RichProgressBar
from rich.progress import Task, Progress
from abc import abstractmethod

class BaseProgress():

    @abstractmethod
    def setup_dreaming(self, dream_targets, iterations):
        pass

    @abstractmethod
    def next_dream(self, target, iterations):
        pass

    @abstractmethod
    def update_dreaming(self):
        pass

    @abstractmethod
    def clear_dreaming(self):
        pass

class CustomRichProgressBar(RichProgressBar, BaseProgress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # training update is done by core framework

        self.dreaming_progress = None
        self.target_progress = None

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

    def setup_dreaming(self, dream_targets, iterations):
        self.clear_dreaming()
        self.dreaming_progress = self.progress.add_task(
            "[bright_blue]Dreaming...", total=(len(dream_targets) * iterations)
        )

    def next_dream(self, target, iterations):
        self._clear_target()
        self.target_progress = self.progress.add_task(
            f"[bright_red]Class: {target}\n", total=iterations
        )

    def update_dreaming(self):
        if(self.target_progress is not None):
            self.progress.update(self.target_progress, advance=1)
        if(self.dreaming_progress is not None):
            self.progress.update(self.dreaming_progress, advance=1)

    def _clear_target(self):
        if(self.target_progress is not None):
            self.progress.remove_task(self.target_progress)

    def clear_dreaming(self):
        if(self.dreaming_progress is not None):
            self.progress.remove_task(self.dreaming_progress)
        self._clear_target()

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

class DullProgress(Progress, BaseProgress):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def setup_dreaming(self, dream_targets, iterations):
        pass

    def next_dream(self, target, iterations):
        pass

    def update_dreaming(self):
        pass

    def clear_dreaming(self):
        pass