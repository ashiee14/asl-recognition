"""
sentence_builder.py

Turns noisy per-frame letter predictions into clean letters -> words -> sentences.

Logic:
  - A letter is only accepted once the SAME letter has been predicted for
    `stability_frames` frames in a row (filters out flicker/misreads).
  - If no hand is detected for `pause_frames` frames, the current word is
    considered finished and a space is added (this is how words are
    separated -- pull your hand out of frame briefly between words).

This file has no camera or model code in it -- it's pure logic, so it's
easy to test on its own and reuse in infer.py.
"""

from collections import deque


class SentenceBuilder:
    def __init__(self, stability_frames=15, pause_frames=20):
        self.buffer = deque(maxlen=stability_frames)
        self.stability_frames = stability_frames
        self.pause_frames = pause_frames
        self.no_hand_count = 0
        self.current_word = ""
        self.sentence = ""

    def update(self, letter):
        """Call this once per frame with the predicted letter, or None
        if no hand was detected in that frame."""
        if letter is None:
            self.no_hand_count += 1
            if self.no_hand_count == self.pause_frames and self.current_word:
                self.sentence += self.current_word + " "
                self.current_word = ""
            return

        self.no_hand_count = 0
        self.buffer.append(letter)
        if len(self.buffer) == self.stability_frames and len(set(self.buffer)) == 1:
            if not self.current_word or self.current_word[-1] != letter:
                self.current_word += letter
            self.buffer.clear()

    def get_display_text(self):
        return (self.sentence + self.current_word).strip()

    

    def reset(self):
        self.buffer.clear()
        self.no_hand_count = 0
        self.current_word = ""
        self.sentence = ""