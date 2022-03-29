import random
from math import sqrt

from BitVector import BitVector

GENETIC_CODE_SIZE = 16


class CovidVariant:
    genetic_code: BitVector
    base_infection_prob: float
    base_death_prob: float

    def __init__(
        self,
        infection_prob: float,
        death_prob: float,
        genetic_code: BitVector = None,
    ):
        self.genetic_code = genetic_code or BitVector(size=GENETIC_CODE_SIZE)
        self.base_infection_prob = infection_prob
        self.base_death_prob = death_prob

    def child_variant(self, mutation_chance: float) -> "CovidVariant":
        new_genetic_code = self.genetic_code.deep_copy()
        num_mutations = 0
        for i, bit in enumerate(new_genetic_code):
            if random.random() < mutation_chance:
                new_genetic_code[i] = not bit
                num_mutations += 1
        new_infection_prob = self.base_infection_prob + (
            num_mutations / new_genetic_code.size
        ) * self.base_infection_prob * random.choice((1, -1))
        new_death_prob = self.base_death_prob + (
            num_mutations / new_genetic_code.size
        ) * self.base_death_prob * random.choice((1, -1))
        return CovidVariant(new_infection_prob, new_death_prob, new_genetic_code)

    def similarity(self, other: "CovidVariant") -> float:
        """
        Returns the similarity in genetic code between the variant and a given other variant

        :param other: The variant to compare against
        :return: A float representing the proportion of similarity between the two variants
        """
        return (
            1
            - (self.genetic_code ^ other.genetic_code).count_bits()
            / self.genetic_code.size
        )

    @property
    def variant_code(self) -> str:
        return self.genetic_code.get_bitvector_in_hex()
