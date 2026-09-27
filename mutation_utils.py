import re
from dataclasses import dataclass


VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MUTATION_PATTERN = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class Mutation:
    wildtype: str
    position: int
    mutant: str

    @property
    def token(self):
        return f"{self.wildtype}{self.position}{self.mutant}"


def normalize_sequence(sequence):
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise ValueError("Protein sequence is required.")
    invalid = sorted(set(normalized) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(
            "Invalid character(s) in sequence: " + ", ".join(repr(c) for c in invalid) + "."
        )
    return normalized


def parse_mutation_set(value, max_mutations=None):
    mutation_set = str(value or "").strip().upper()
    if not mutation_set:
        raise ValueError("At least one mutation is required.")

    tokens = mutation_set.split("_")
    if any(not token for token in tokens):
        raise ValueError(
            "Invalid mutation format. Use tokens such as E94H or F88Y_L91A."
        )
    if max_mutations is not None and len(tokens) > max_mutations:
        raise ValueError(f"At most {max_mutations} simultaneous mutations are allowed.")

    mutations = []
    seen_positions = set()
    for token in tokens:
        match = _MUTATION_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(
                f"Invalid mutation token '{token}'. Use wildtype-position-mutant notation, for example E94H."
            )
        wildtype, position_text, mutant = match.groups()
        position = int(position_text)
        if wildtype == mutant:
            raise ValueError(f"Mutation '{token}' does not change the residue.")
        if position in seen_positions:
            raise ValueError(f"Position {position} is specified more than once.")
        seen_positions.add(position)
        mutations.append(Mutation(wildtype, position, mutant))

    return mutations


def mutation_set_label(mutations):
    return "_".join(mutation.token for mutation in mutations)


def create_multi_mutant(wildtype_sequence, mutations):
    sequence = normalize_sequence(wildtype_sequence)

    # Validate every mutation against the same reference before changing the sequence.
    for mutation in mutations:
        index = mutation.position - 1
        if index < 0 or index >= len(sequence):
            raise ValueError(
                f"Position {mutation.position} is out of range (1-{len(sequence)})."
            )
        observed = sequence[index]
        if observed != mutation.wildtype:
            raise ValueError(
                f"Wildtype mismatch: position {mutation.position} is {observed}, not {mutation.wildtype}."
            )

    mutant_sequence = list(sequence)
    for mutation in mutations:
        mutant_sequence[mutation.position - 1] = mutation.mutant
    return "".join(mutant_sequence)
