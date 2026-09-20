"""Injects the offline-computed JSON exports into the static template so
the page opens with a plain double-click, no server, no fetch/CORS issue."""
import json

DATA_DIR = "data"


def load(name):
    with open(f"{DATA_DIR}/{name}") as f:
        return json.load(f)


def main():
    metrics = load("metrics.json")
    graph_metrics = load("graph_metrics.json")
    alerts = load("alerts.json")
    rings = load("rings.json")

    with open("site/template.html") as f:
        html = f.read()

    html = html.replace("/*__METRICS__*/", json.dumps(metrics))
    html = html.replace("/*__GRAPH_METRICS__*/", json.dumps(graph_metrics))
    html = html.replace("/*__ALERTS__*/", json.dumps(alerts))
    html = html.replace("/*__RINGS__*/", json.dumps(rings))

    with open("site/index.html", "w") as f:
        f.write(html)
    print("Wrote site/index.html")


if __name__ == "__main__":
    main()
