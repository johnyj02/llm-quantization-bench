#!/usr/bin/env python3
"""Count pages in a PDF with no third-party dependency.

Exists because gate G6 is a page-count gate checked repeatedly from Sep 20
onward, and `mdls` is unreliable here (Spotlight does not index the build
directory). tectonic emits PDF 1.5 with compressed object streams, so a plain
byte-scan for /Type /Page finds nothing -- the object streams must be inflated
first.
"""

from __future__ import annotations

import re
import sys
import zlib


def count_pages(path: str) -> int:
    with open(path, "rb") as handle:
        data = handle.read()

    # Fast path: uncompressed page objects.
    direct = len(re.findall(rb"/Type\s*/Page[^sC]", data))

    # Object streams (PDF >= 1.5): inflate each and count inside.
    inflated = 0
    for match in re.finditer(rb"stream", data):
        start = data.find(b"\n", match.end())
        end = data.find(b"endstream", match.end())
        if start < 0 or end < 0 or start >= end:
            continue
        try:
            raw = zlib.decompressobj().decompress(data[start + 1 : end])
        except zlib.error:
            continue
        inflated += len(re.findall(rb"/Type\s*/Page[^sC]", raw))

    pages = max(direct, inflated)
    if pages:
        return pages

    # Last resort: the page-tree /Count. Take the largest, which is the root.
    counts = [int(n) for n in re.findall(rb"/Count\s+(\d+)", data)]
    return max(counts) if counts else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: pdf_pages.py <file.pdf>")
    n = count_pages(sys.argv[1])
    if n == 0:
        sys.exit(f"could not determine page count for {sys.argv[1]}")
    print(n)
