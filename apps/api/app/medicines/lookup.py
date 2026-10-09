"""From a medicine name a caller said, to the medicine it names in the catalogue.

    "dolo 650"       ->  Dolo 650 Tablet, paracetamol 650 mg   (in the catalogue, just so)
    "glycomet"       ->  Glycomet, and "which one? 250, 500, 850..."
    "glycomate 500"  ->  Glycomet 500, to be confirmed          (misheard, most likely)
    "shelcal 500"    ->  not found, and said so                 (not in the catalogue)

The rule that matters: nothing is said that the catalogue does not settle. A strength or
what a medicine contains is given only when every product the name could mean has the
same one. Otherwise it is left for the caller to say. A caller told about the wrong
medicine is worse off than one told "I could not find that".

The catalogue is large and still not complete: well-known brands are missing from it. So
a name is only ever changed to one it could have been misheard from: a letter out, or the
same consonants. A name that is merely like another ("Shelcal", "Selca") is left as it
was heard.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from collections.abc import Awaitable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.medicines.catalog import SOURCE_NAMES, Medicine
from app.medicines.encoders import NameEncoders, build_encoders
from app.medicines.store import MedicineStore
from app.medicines.text import (
    FORM_WORDS,
    alike,
    content_tokens,
    form_tokens,
    is_noise,
    is_number,
    near,
)
from app.observability.tracing import tracer
from app.rag.retrieval.qdrant_store import build_client

log = get_logger(__name__)

# not_indexed: Qdrant answers, and holds no catalogue. Nothing is asked of it until the
# server is started again with one (make medicines).
Status = Literal["not_started", "ready", "not_indexed", "unreachable"]

_WORD = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+")
# Past this a strength is the amounts of several ingredients run together: too much to say
# aloud, and the composition says it better.
_STRENGTH_MAX = 40
_SOURCE_ORDER = {"nlem": 0, "jan_aushadhi": 1, "az": 2}


class Match(BaseModel):
    # exact:   the catalogue has this name, spelt as it was said. `name` is the product.
    # several: the name is in the catalogue, and more than one product carries it, or the
    #          product has a word the caller did not say. `choices` lists them.
    # close:   not in the catalogue as said, but a name it could have been misheard from
    #          is: a letter out, or the same consonants. `name` is that spelling, to be
    #          confirmed, and `choices` any others. Nothing else follows from a guess.
    # unknown: not in the catalogue. `choices` may hold names a little like it.
    status: Literal["exact", "several", "close", "unknown"]
    heard: str
    name: str | None = None
    strength: str | None = None
    unit: str | None = None
    composition: str | None = None
    source: str | None = None
    choices: list[str] = Field(default_factory=list)


class _Fit(BaseModel):
    """How one catalogue name answers what was heard."""

    medicine: Medicine
    quality: float  # 0 to 1: how alike the matched words are
    spelt: list[str]  # the catalogue's spelling of each word that was heard, in the order heard
    extra: int  # words in the catalogue name that were not said
    as_said: bool  # every word is spelt in the catalogue just as it was heard
    starts: bool  # the catalogue name begins with the first word heard
    rank: int  # where the search (or the cross-encoder, where there is one) put it


def _words(name: str) -> list[tuple[str, str]]:
    """(lower-case, as written) for each word that says which medicine it is."""
    written = _WORD.findall(name)
    lower = [word.lower() for word in written]
    return [
        (lower[index], word)
        for index, word in enumerate(written)
        if lower[index] not in FORM_WORDS and not is_noise(lower, index)
    ]


def _fit(wanted: list[str], medicine: Medicine, rank: int = 0) -> _Fit | None:
    """The fit of a catalogue name to the words heard, or None if a word is missing.

    A word is there if the name has it, or has one it could have been misheard from.
    """
    have = _words(medicine.name)
    if not have:
        return None
    spelt: list[str] = []
    scores: list[float] = []
    used: list[int] = []
    for word in wanted:
        best, at = 0.0, -1
        for index, (lower, _) in enumerate(have):
            if index not in used and (score := alike(word, lower)) > best and near(word, lower):
                best, at = score, index
        if at < 0:
            break
        used.append(at)
        spelt.append(have[at][1])
        scores.append(best)
    else:
        return _Fit(
            medicine=medicine,
            quality=sum(scores) / len(scores),
            spelt=spelt,
            extra=len(have) - len(used),
            as_said=min(scores) == 1.0,
            starts=used[0] == 0,
            rank=rank,
        )

    # Word for word it does not fit. Speech recognition also splits and joins words
    # ("eco sprin", "pandy"), so the start of the name is tried as one run of letters.
    # Only when the first word heard is nowhere in the name: "Ascoril LS" is not "Ascoril C".
    said = [word for word in wanted if not is_number(word)]
    if any(near(said[0], lower) for lower, _ in have):
        return None
    letters = "".join(said)
    leading: list[tuple[str, str]] = []
    for lower, written in have:
        if is_number(lower) or len(leading) == 3:
            break
        leading.append((lower, written))
    best, count = 0.0, 0
    for size in range(1, len(leading) + 1):
        run = "".join(lower for lower, _ in leading[:size])
        # A letter out at most. Two words run together share too much to go by sound.
        if (score := alike(letters, run)) > best and near(letters, run, by_sound=False):
            best, count = score, size
    if len(letters) < 4 or not count:
        return None
    numbers: list[str] = []
    for number in (word for word in wanted if is_number(word)):
        figure = next((w for lower, w in have if is_number(lower) and alike(number, lower)), None)
        if figure is None:
            return None
        numbers.append(figure)
    return _Fit(
        medicine=medicine,
        quality=best * 0.98,
        spelt=[written for _, written in leading[:count]] + numbers,
        extra=len(have) - count - len(numbers),
        as_said=best == 1.0,
        starts=True,
        rank=rank,
    )


def _agreed(values: list[str | None]) -> str | None:
    """The one value they all share, or None if any differs or is missing."""
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def _order(fit: _Fit) -> tuple[float, bool, int, bool, int, int, str]:
    medicine = fit.medicine
    return (
        -fit.quality,
        not fit.starts,  # "Pan 40" before "A Pan 40"
        fit.extra,
        medicine.discontinued,
        _SOURCE_ORDER[medicine.source],
        # Only now the order the candidates came in: the search's, or the cross-encoder's.
        # It settles which of a brand's products are listed first, and nothing else.
        fit.rank,
        medicine.name,
    )


def _near_misses(wanted: list[str], candidates: Sequence[Medicine]) -> list[str]:
    """Up to three catalogue names that are close, for a name that is not in it."""
    first = next(index for index, word in enumerate(wanted) if not is_number(word))
    scored: list[tuple[float, str]] = []
    for medicine in candidates:
        have = [lower for lower, _ in _words(medicine.name)]
        if not have:
            continue
        scores = [max(alike(word, lower) for lower in have) for word in wanted]
        # The name itself has to be close. A shared "forte" or "500" is not enough.
        if scores[first] >= 0.8 and sum(scores) / len(scores) >= 0.6:
            scored.append((sum(scores) / len(scores), medicine.name))
    return list(dict.fromkeys(name for _, name in sorted(scored, reverse=True)))[:3]


def _spelling(fits: list[_Fit], unmatched: list[str]) -> str:
    """The catalogue's spelling of the words that were heard: "glycomate 500" -> "Glycomet 500"."""
    best = fits[0]
    shaped = [fit for fit in fits if len(fit.spelt) == len(best.spelt)]
    words = (
        Counter(fit.spelt[index] for fit in shaped).most_common(1)[0][0]
        for index in range(len(best.spelt))
    )
    return " ".join([*words, *unmatched])


def _answer(
    heard: str, words: list[str], fits: list[_Fit], *, close: bool, unmatched: list[str]
) -> Match:
    """The match that a set of fitting catalogue names adds up to.

    ``unmatched`` are numbers the person said that no product of this name carries
    ("Atorva 10", where the catalogue has Atorva, Atorva 20 and Atorva 40). They are
    kept as said, and nothing is claimed about the strength.
    """
    fits = sorted(fits, key=_order)
    # "Ascoril syrup": where the person said the form, keep to the products of that form.
    if (said := form_tokens(heard)) and (
        same := [f for f in fits if said & form_tokens(f.medicine.name)]
    ):
        fits = same

    # "Glycomate" could be Glucomate or Glycomet, misheard. The nearest is named and the
    # other is offered: which one it is, is the person's to say.
    others: list[str] = []
    if close:
        spellings: dict[tuple[str, ...], list[_Fit]] = {}
        for fit in fits:
            spellings.setdefault(tuple(word.lower() for word in fit.spelt), []).append(fit)
        fits, *rest = spellings.values()
        others = [_spelling(group, unmatched) for group in rest][:5]

    # One product is meant when a catalogue name has nothing more to it than was said, and
    # either the person gave a number ("Dolo 650") or no longer name carries those words.
    # A bare brand ("Dolo") stands for every product under it, and none is picked. Nor is
    # one picked that has a word the person did not say: "Glycomet 500" is not yet
    # "Glycomet 500 SR".
    whole = [fit for fit in fits if fit.extra == 0]
    numbered = any(is_number(word) for word in words)
    settled = whole if whole and not unmatched and (numbered or len(whole) == len(fits)) else []
    chosen = settled or fits
    names = list(dict.fromkeys(fit.medicine.name for fit in chosen))
    # "Ascoril LS" is a syrup and drops: two products, both just as said. Still a choice.
    one = bool(settled) and len(names) == 1
    name = names[0] if one else _spelling(fits, unmatched)
    source = SOURCE_NAMES[fits[0].medicine.source]
    if close:
        # The name is a guess, so nothing that follows from it is said yet.
        return Match(status="close", heard=heard, name=name, source=source, choices=others)
    strength = None if unmatched else _agreed([fit.medicine.strength for fit in chosen])
    return Match(
        status="exact" if one else "several",
        heard=heard,
        name=name,
        strength=strength if strength and len(strength) <= _STRENGTH_MAX else None,
        unit=_agreed([fit.medicine.unit for fit in chosen]),
        composition=_agreed([fit.medicine.composition for fit in chosen]),
        source=source,
        choices=[] if one else names[:6],
    )


def decide(heard: str, candidates: Sequence[Medicine]) -> Match:
    """Which medicine a name that was heard means, given the nearest catalogue names.

    ``candidates`` come most relevant first. Their order only breaks ties. In order: the
    name as it was said, numbers and all. Then the name as said, with any number the
    catalogue has no product for set aside. Only then a name it could have
    been misheard from, which is never taken as certain. Anything further off is left as
    it was heard.
    """
    wanted = content_tokens(heard)
    words = [word for word in wanted if not is_number(word)]
    if not words:
        return Match(status="unknown", heard=heard)
    numbers = [word for word in wanted if is_number(word)]

    misheard: tuple[list[str], list[_Fit], list[str]] | None = None
    for tried, unmatched in ((wanted, []), (words, numbers)) if numbers else ((wanted, []),):
        fits = [
            fit
            for rank, medicine in enumerate(candidates)
            if (fit := _fit(tried, medicine, rank)) is not None
        ]
        if as_said := [fit for fit in fits if fit.as_said]:
            return _answer(heard, tried, as_said, close=False, unmatched=unmatched)
        if fits and misheard is None:
            misheard = (tried, fits, unmatched)
    if misheard is not None:
        return _answer(heard, misheard[0], misheard[1], close=True, unmatched=misheard[2])
    return Match(status="unknown", heard=heard, choices=_near_misses(wanted, candidates))


class MedicineLookup:
    """Looks a name up in the catalogue, without ever holding a call up.

    A caller is waiting while this runs, so it gives up quickly. After two failures in a
    row it stops asking for a minute: a catalogue that is down must not add a pause to
    every medicine.

    With ``encoders`` it also asks the bi-encoder for names the spelling-and-sound search
    missed, and has the cross-encoder order what was found. Both are extras: until they
    are loaded, or if one fails or is slow, the lookup goes on without it.
    """

    def __init__(
        self,
        store: MedicineStore,
        *,
        encoders: NameEncoders | None = None,
        timeout_s: float = 1.5,
        candidates: int = 64,
        dense_candidates: int = 16,
        pause_s: float = 60.0,
    ) -> None:
        self._store = store
        self._encoders = encoders
        self._timeout_s = timeout_s
        self._candidates = candidates
        self._dense_candidates = dense_candidates
        self._pause_s = pause_s
        self._failures = 0
        self._resume_at = 0.0
        self._warming: asyncio.Task[None] | None = None
        self._checking: asyncio.Task[None] | None = None
        self.status: Status = "not_started"

    @property
    def encoders(self) -> NameEncoders | None:
        return self._encoders

    def start(self) -> None:
        """Check the catalogue is there, and load the models, in the background.

        No caller waits on either: the first lookup finds both already done.
        """
        if self._checking is None:
            self._checking = asyncio.create_task(self._check())
        if self._encoders is not None and not self._encoders.ready and self._warming is None:
            self._warming = asyncio.create_task(self._warm(self._encoders))

    async def _check(self) -> None:
        try:
            names = await self._store.count() if await self._store.exists() else None
        except Exception as exc:
            # It may only be down for now. Lookups go on being tried, and give up quickly.
            self.status = "unreachable"
            log.error("medicines.unreachable", error=type(exc).__name__)
            return
        if not names:
            self.status = "not_indexed"
            log.error(
                "medicines.not_indexed",
                collection=self._store.collection,
                hint="run `make medicines`",
            )
            return
        self.status = "ready"
        log.info("medicines.ready", collection=self._store.collection, names=names)

    async def _warm(self, encoders: NameEncoders) -> None:
        try:
            await asyncio.to_thread(encoders.warm)
        except Exception as exc:
            self._encoders = None
            log.warning("medicines.encoders_failed", stage="load", error=type(exc).__name__)

    async def _extra[T](self, stage: str, work: Awaitable[T], deadline: float) -> T | None:
        """An encoder's answer, or None if it failed or there is no time left for it."""
        try:
            left = deadline - asyncio.get_running_loop().time()
            return await asyncio.wait_for(work, max(left, 0.001))
        except Exception as exc:
            log.warning("medicines.encoders_failed", stage=stage, error=type(exc).__name__)
            return None

    async def _search(self, heard: str) -> list[Medicine]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_s
        encoders = self._encoders if self._encoders is not None and self._encoders.ready else None

        dense: list[float] | None = None
        if encoders and self._dense_candidates and await self._store.has_dense():
            # Half the time at most: the search itself must still fit.
            dense = await self._extra(
                "embed",
                asyncio.to_thread(encoders.embed_heard, heard),
                loop.time() + self._timeout_s / 2,
            )
        found = await asyncio.wait_for(
            self._store.search(
                heard, limit=self._candidates, dense=dense, dense_limit=self._dense_candidates
            ),
            max(deadline - loop.time(), 0.001),
        )
        if encoders and encoders.cross_encoder and len(found) > 1:
            ordered = await self._extra(
                "rerank", asyncio.to_thread(encoders.order, heard, found), deadline
            )
            found = ordered or found
        return found

    async def find(self, heard: str) -> Match | None:
        """The match for a name, or None if the catalogue could not be asked."""
        # The span says how the lookup went. The name is health information and is not in it.
        with tracer.start_as_current_span("medicines lookup") as span:
            match = await self._find(heard)
            span.set_attribute("status", match.status if match else "unavailable")
            return match

    async def _find(self, heard: str) -> Match | None:
        heard = " ".join(heard.split())[:120]
        if not heard or self.status == "not_indexed" or time.monotonic() < self._resume_at:
            return None
        try:
            found = await self._search(heard)
        except Exception as exc:
            self._failures += 1
            if self._failures >= 2:
                self._resume_at = time.monotonic() + self._pause_s
            # The name itself is health information: only the kind of failure is logged.
            log.warning(
                "medicines.lookup_failed", error=type(exc).__name__, failures=self._failures
            )
            return None
        self._failures = 0
        match = decide(heard, found)
        log.info("medicines.lookup", status=match.status, candidates=len(found))
        return match

    async def aclose(self) -> None:
        for task in (self._warming, self._checking):
            if task is not None and not task.done():
                task.cancel()
        await self._store.client.close()


def build_medicine_lookup(settings: Settings) -> MedicineLookup | None:
    """The catalogue for this server, or None where there is none to ask.

    Needs a Qdrant server: the embedded local index reads every one of a quarter of a
    million names for each lookup, which is too slow for a call.
    """
    if not settings.medicines_lookup:
        return None
    if not settings.qdrant_url:
        log.warning("medicines.not_configured", hint="set QDRANT_URL and QDRANT_API_KEY")
        return None
    return MedicineLookup(
        MedicineStore(build_client(settings), settings.medicines_collection),
        encoders=build_encoders(settings),
        timeout_s=settings.medicines_lookup_timeout_s,
        dense_candidates=settings.medicines_dense_candidates,
    )
