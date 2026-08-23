"""CPU-only end-to-end smoke test for the method boundary."""

from __future__ import annotations

from nashrs import NashRSConfig, NashRSRewardConstructor


class CyclingSampler:
    def __init__(self, values: list[str]) -> None:
        self.values = values
        self.index = 0

    def sample(self, prompt: str, n: int) -> list[str]:
        del prompt
        result = [self.values[(self.index + i) % len(self.values)] for i in range(n)]
        self.index += n
        return result


class LengthPreference:
    def compare(self, prompts, left, right):
        del prompts
        return [0.75 if len(a) > len(b) else 0.25 if len(a) < len(b) else 0.5 for a, b in zip(left, right)]


constructor = NashRSRewardConstructor(
    NashRSConfig(tau=1.0, b1=4, b2=3, seed=7),
    policy_sampler=CyclingSampler(["short", "a longer current answer"]),
    reference_sampler=CyclingSampler(["reference", "a long reference response"]),
    preference_oracle=LengthPreference(),
)
batch = constructor.construct(["Explain Nash equilibrium."], ["A useful, sufficiently long answer."])
print({"reward": batch.rewards[0], **batch.accounting.to_logs()})

