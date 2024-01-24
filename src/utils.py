import numpy as np
from math import cos, sqrt, pi
from dv import AedatFile
from src.io.psee_loader import PSEELoader

class Events(object):
    def __init__(self, raw):
        num_events = len(raw['x'])
        self.data = np.rec.array(None, dtype=[('x', np.int16), ('y', np.int16), ('p', np.bool), ('ts', np.int64)], shape=(num_events))
        self.data.x = raw['x'].reshape(-1)
        self.data.y = raw['y'].reshape(-1)
        self.data.p = raw['p'].reshape(-1)
        self.data.ts = raw['t'].reshape(-1)

def event2img(frame_data, img_size):
    h, w = img_size[0], img_size[1]
    img = np.ones((h, w), dtype=np.uint8)
    if frame_data.size > 0:
        img.fill(128)
        for datum in np.nditer(frame_data):
            img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)
        img = np.piecewise(img, [img == 0, img == 1, img == 128],[0, 255, 128])
    else:
        print('NO EVENTS!')
    return np.asarray(img, dtype=np.uint8).reshape(h, w, 1)

def getDistanceAndTimeDiff(a, b):
    norm = 24 * 60 * 60
    distance = round(sqrt(((b[2] - a[2]) ** 2) + ((b[1] - a[1]) ** 2)), 2)
    time_diff = round(np.cos(2 * np.pi * b[-1] / norm) - np.cos(2 * np.pi * a[-1] / norm), 4)
    return distance, time_diff

def makeNodeAndEdge(evs):
    nodes = [[ev[1], ev[2], ev[3]] for ev in evs]
    edges = []
    edges_attr = []
    for i in range(len(evs)-1): 
        edges.append([evs[i][0], evs[i+1][0]])
        edges_attr.append(getDistanceAndTimeDiff(evs[i], evs[i+1]))
    for i in range(len(evs)-1):
        for j in range(i+1, len(evs)):
            if (evs[i][1] == evs[j][1]) and (evs[i][2] == evs[j][2]):
                edges.append([evs[i][0], evs[j][0]])
                edges_attr.append([0., getDistanceAndTimeDiff(evs[i], evs[j])[1]])
                break
    return nodes, edges, edges_attr

def readEvent(root):
    ev = None
    if isinstance(root, str):
        with AedatFile(root) as f:
            events = np.hstack([packet for packet in f['events'].numpy()])
            ev = events
            timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    else:
        video = PSEELoader(root + str(data_list[j]))
        event = video.load_n_events(video.event_count())
        ev = event

    return ev
