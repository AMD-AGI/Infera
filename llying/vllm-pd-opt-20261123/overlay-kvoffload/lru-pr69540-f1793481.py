# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import heapq
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)

_Rank = tuple[int, int]
_IndexEntry = tuple[_Rank, OffloadKey]

# Target sublist length of _SortedEntries; a sublist is split at twice this.
_SUBLIST_LOAD = 512


class _SortedEntries:
    """Sorted list of distinct index entries with O(log n + load) updates.

    Entries live in sorted sublists of bounded length, located by bisecting
    the per-sublist maxima, so an insert or remove never moves more than one
    sublist's worth of elements.
    """

    __slots__ = ("_len", "_lists", "_maxes")

    def __init__(self) -> None:
        self._lists: list[list[_IndexEntry]] = []
        self._maxes: list[_IndexEntry] = []
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[_IndexEntry]:
        for sublist in self._lists:
            yield from sublist

    def clear(self) -> None:
        self._lists.clear()
        self._maxes.clear()
        self._len = 0

    def add(self, entry: _IndexEntry) -> None:
        lists, maxes = self._lists, self._maxes
        if not maxes:
            lists.append([entry])
            maxes.append(entry)
            self._len += 1
            return
        i = bisect.bisect_left(maxes, entry)
        if i == len(maxes):
            i -= 1
            lists[i].append(entry)
            maxes[i] = entry
        else:
            bisect.insort(lists[i], entry)
        sublist = lists[i]
        if len(sublist) > 2 * _SUBLIST_LOAD:
            upper = sublist[_SUBLIST_LOAD:]
            del sublist[_SUBLIST_LOAD:]
            maxes[i] = sublist[-1]
            lists.insert(i + 1, upper)
            maxes.insert(i + 1, upper[-1])
        self._len += 1

    def remove(self, entry: _IndexEntry) -> None:
        lists, maxes = self._lists, self._maxes
        i = bisect.bisect_left(maxes, entry)
        if i == len(maxes):
            raise ValueError(f"{entry!r} not in index")
        sublist = lists[i]
        j = bisect.bisect_left(sublist, entry)
        if j == len(sublist) or sublist[j] != entry:
            raise ValueError(f"{entry!r} not in index")
        del sublist[j]
        self._len -= 1
        if not sublist:
            del lists[i]
            del maxes[i]
        elif j == len(sublist):
            maxes[i] = sublist[-1]


@dataclass(slots=True)
class _RequestRank:
    owner: object
    cache_generation: int
    rank: int


class LRUCachePolicy(CachePolicy):
    """LRU caching policy with logical recency independent of transfer pinning.

    Each chunk has a rank; eviction takes evictable chunks in increasing
    (rank, key) order. A chunk's recency can be updated while it is pinned;
    when it later becomes evictable, it is ordered by the rank assigned by the
    request rather than by its transfer-completion time.

    Active requests share a provisional rank across batches, ordered by token
    position so tails precede heads. Finalization commits the request's recency.

    A use is indicated by,
     - First time the key is added (store).
     - A request-scoped access.

    The evictable chunks are kept in an exact order index with two parts, so
    no operation ever rebuilds over all evictable chunks:
     - ``_tail``: an insertion-ordered dict whose iteration order is increasing
       (rank, key). A chunk is appended when it sorts after the current last
       entry. Finalized recency always takes a fresh maximum rank, so the bulk
       of the updates (``on_request_finished``) land here in O(1).
     - ``_sorted``: a sorted list holding every other evictable chunk, e.g.
       chunks ranked by a still-running request.
    ``evict`` merges both parts. Invariant: every evictable chunk is indexed
    exactly once, under its current rank; no other chunk is indexed.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, _Rank] = {}
        self._evictable: set[OffloadKey] = set()
        self._tail: OrderedDict[OffloadKey, _Rank] = OrderedDict()
        self._sorted = _SortedEntries()
        # key -> rank under which the key is stored in _sorted.
        self._sorted_ranks: dict[OffloadKey, _Rank] = {}
        self._next_rank = 0
        self._cache_generation = 0

    def _get_request_rank(self, req_context: ReqContext) -> int:
        state = req_context.get_state(_RequestRank)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            self._next_rank += 1
            state = _RequestRank(self, self._cache_generation, self._next_rank)
            req_context.set_state(state)
        return state.rank

    def _assign_new_rank(self, key: OffloadKey) -> bool:
        if key not in self.chunks:
            return False
        self._next_rank += 1
        self._ranks[key] = (self._next_rank, 0)
        return key in self._evictable

    def _unindex(self, key: OffloadKey) -> None:
        if self._tail.pop(key, None) is None:
            rank = self._sorted_ranks.pop(key, None)
            if rank is not None:
                self._sorted.remove((rank, key))

    def _push_evictable(self, key: OffloadKey) -> None:
        """(Re)index an evictable key under its current rank."""
        self._unindex(key)
        rank = self._ranks[key]
        tail = self._tail
        if tail:
            last_key = next(reversed(tail))
            if (rank, key) > (tail[last_key], last_key):
                tail[key] = rank
                return
        elif not self._sorted_ranks:
            tail[key] = rank
            return
        self._sorted.add((rank, key))
        self._sorted_ranks[key] = rank

    def _update_recency(self, keys: Iterable[OffloadKey]) -> None:
        tail = self._tail
        for key in keys:
            if self._assign_new_rank(key):
                # A fresh rank exceeds every assigned rank, so the key sorts
                # after everything indexed and can be appended directly.
                self._unindex(key)
                tail[key] = self._ranks[key]

    def _iter_evictable(self) -> Iterable[_IndexEntry]:
        """Evictable entries in increasing (rank, key) order."""
        tail_entries = ((rank, key) for key, rank in self._tail.items())
        if not self._sorted_ranks:
            return tail_entries
        return heapq.merge(iter(self._sorted), tail_entries)

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(
        self, key: OffloadKey, chunk: ChunkStatus, req_context: ReqContext
    ) -> None:
        self.chunks[key] = chunk
        position = req_context.get_offload_key_position(key)
        if position is None:
            self._next_rank += 1
            self._ranks[key] = (self._next_rank, 0)
        else:
            self._ranks[key] = (self._get_request_rank(req_context), -position)
        if chunk.ref_cnt == 0:
            self._evictable.add(key)
            self._push_evictable(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._ranks.pop(key, None)
        self._evictable.discard(key)
        self._unindex(key)

    @override
    def on_request_access(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        request_rank = self._get_request_rank(req_context)
        updated_evictable: list[OffloadKey] = []
        for key in keys:
            position = req_context.get_offload_key_position(key)
            if position is None or key not in self.chunks:
                continue
            # All batches share one request rank; later positions precede
            # earlier ones without revisiting the already observed prefix.
            rank = (request_rank, -position)
            if self._ranks[key] >= rank:
                continue
            self._ranks[key] = rank
            if key in self._evictable:
                updated_evictable.append(key)
        for key in updated_evictable:
            self._push_evictable(key)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        self._update_recency(reversed(order_request_keys(key_groups, req_context)))

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._evictable.clear()
        self._tail.clear()
        self._sorted.clear()
        self._sorted_ranks.clear()
        self._next_rank = 0
        self._cache_generation += 1

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []

        selected: list[OffloadKey] = []
        for _, key in self._iter_evictable():
            if key in protected:
                continue
            assert self.chunks[key].ref_cnt == 0
            selected.append(key)
            if len(selected) == n:
                break
        if len(selected) < n:
            return None

        candidates: list[tuple[OffloadKey, ChunkStatus]] = []
        for key in selected:
            chunk = self.chunks[key]
            self._evictable.remove(key)
            self._unindex(key)
            del self.chunks[key]
            del self._ranks[key]
            candidates.append((key, chunk))
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # chunks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self._evictable.add(key)
        self._push_evictable(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been evictable before it was pinned.
        self._evictable.remove(key)
        self._unindex(key)
