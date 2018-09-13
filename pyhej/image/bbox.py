def be_merge(box1, box2, maxGap=1):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x, y = min(x1, x2), min(y1, y2)
    w, h = max(x1+w1, x2+w2) - x, max(y1+h1, y2+h2) - y

    if w - w1 - w2 > maxGap:
        return False

    if h - h1 - h2 > maxGap:
        return False

    return True


def to_merge(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1+w1, x2+w2) - x
    h = max(y1+h1, y2+h2) - y

    return x, y, w, h


def todo_logs(logs, resx=[]):
    items = logs.copy()
    times = range(len(logs))
    for i in 2*times:
        box1 = items.pop()
        for box2 in items:
            if be_merge(box1, box2, 3):
                break
            else:
                box2 = None
        if box2 is None:
            items.append(box1)
        else:
            items.remove(box2)
            items.append(to_merge(box1, box2))
    return items