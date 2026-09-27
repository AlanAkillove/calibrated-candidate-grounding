"""Quick reachability probe for data download sources (run: python tools/probe_sources.py).

Not part of the research pipeline; diagnostic only.
"""

import urllib.request

URLS = {
    "unc_refcoco+": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    "coco_annos": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "coco_val_img": "http://images.cocodataset.org/val2014/COCO_val2014_000000397133.jpg",
    "coco_train_img": "http://images.cocodataset.org/train2014/COCO_train2014_000000520309.jpg",
}


def head(url: str) -> None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=25) as resp:
            print(url.split("/")[2], "->", resp.status, "len=", resp.headers.get("Content-Length"))
    except Exception as exc:  # noqa: BLE001
        print(url, "FAIL", type(exc).__name__, str(exc)[:150])


if __name__ == "__main__":
    for name, u in URLS.items():
        print(f"[{name}]", end=" ")
        head(u)
