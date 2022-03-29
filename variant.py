import random
from math import log
from typing import TYPE_CHECKING

from BitVector import BitVector

if TYPE_CHECKING:
    from model import CovidModel

GENETIC_CODE_SIZE = 4
MAX_HEX_NUMERALS = int(log(2**GENETIC_CODE_SIZE, 16))


class CovidVariant:
    model: "CovidModel"
    genetic_code: BitVector
    base_infection_prob: float
    base_death_prob: float

    def __init__(
        self,
        model: "CovidModel",
        infection_prob: float,
        death_prob: float,
        genetic_code: BitVector = None,
    ):
        self.model = model
        self.genetic_code = genetic_code or BitVector(size=GENETIC_CODE_SIZE)
        self.base_infection_prob = infection_prob
        self.base_death_prob = death_prob

    def child_variant(self) -> "CovidVariant":
        new_genetic_code = self.genetic_code.deep_copy()
        num_mutations = 0
        for i, bit in enumerate(new_genetic_code):
            if random.random() < self.model.mutation_prob:
                new_genetic_code[i] = not bit
                num_mutations += 1

        variant_hex_code = new_genetic_code.get_bitvector_in_hex()
        if variant := self.model.variant_code_map.get(variant_hex_code):
            return variant

        new_infection_prob = self.base_infection_prob + (
            num_mutations / new_genetic_code.size
        ) * self.base_infection_prob * random.choice((1, -1))
        new_death_prob = self.base_death_prob + (
            num_mutations / new_genetic_code.size
        ) * self.base_death_prob * random.choice((1, -1))

        new_variant = CovidVariant(
            self.model, new_infection_prob, new_death_prob, new_genetic_code
        )
        self.model.variant_code_map[variant_hex_code] = new_variant
        return new_variant

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
    def name(self) -> str:
        hex_code = self.genetic_code.get_bitvector_in_hex().upper()
        if padding := MAX_HEX_NUMERALS - len(hex_code):
            hex_code = "0" * padding + hex_code
        return hex_code

    def __eq__(self, other: "CovidVariant") -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
