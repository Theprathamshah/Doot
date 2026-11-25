"""Free public remote & startup JSON feed collectors."""

from doot.collectors.feeds.himalayas import HimalayasCollector
from doot.collectors.feeds.hn_hiring import HNHiringCollector
from doot.collectors.feeds.jobicy import JobicyCollector
from doot.collectors.feeds.reddit import RedditHiringCollector
from doot.collectors.feeds.remotive import RemotiveCollector
from doot.collectors.feeds.wwr import WeWorkRemotelyCollector

__all__ = [
    "HNHiringCollector",
    "HimalayasCollector",
    "JobicyCollector",
    "RedditHiringCollector",
    "RemotiveCollector",
    "WeWorkRemotelyCollector",
]
