from utils import graphics
import argparse
from json import load
import numpy as np
from utils.environment import Environment, Route, targets
from math import radians, cos, sin, degrees, atan2


def positions(pos_angle, pos_dist):
    """
    Converts target position from course and distance to
    relative coordinates X and Y.
    :param pos_angle: course angle for target, degrees
    :param pos_dist: distance to target, nautical miles
    :return: relative coords X and Y
    """
    return round(pos_dist * cos(radians(pos_angle)), 2), \
           round(pos_dist * sin(radians(pos_angle)), 2)


if __name__ == "__main__":
    import time

    start_time = time.time()

    parser = argparse.ArgumentParser()
    # Result file
    parser.add_argument("result", type=str, help="--")
    # Data files
    # parser.add_argument("--nav-data", type=str, default='figures', help="navigational parameters data file")
    # parser.add_argument("--targets", type=str, default='figures', help="targets data file")
    # parser.add_argument("--constraints", type=str, default='figures', help="navigational constraints data file")
    # parser.add_argument("--hydrometeo", type=str, default='figures', help="hydrometeo data file")
    # parser.add_argument("--route", type=str, default='figures', help="route description file")
    # parser.add_argument("--ongoing", type=str, default='figures', help="ongoing maneuver data file")
    # parser.add_argument("--offered", type=str, default='figures', help="offered maneuvers set file")
    # parser.add_argument("--settings", type=str, default='figures', help="settings file")

    parser.add_argument("-c", "--circles", action="store_false", help="Disable time circles")
    parser.add_argument("--show", action="store_false", help="Do not show plot")
    # parser.add_argument("--savekt", action="store_true", help="Save result for KT")
    parser.add_argument("--savefig", action="store_true", help="Save figure")
    parser.add_argument("--figdir", type=str, default='figures', help="Directory to place figures")
    parser.add_argument("-e", "--epsilon", type=float, default=0.25, help="Epsilon value for RDP algorithm")
    parser.add_argument("-n", "--solution", type=int, default=0, help="Number of solution")
    parser.add_argument("--video", action="store_true", help="Epsilon value for RDP algorithm")
    # parser.add_argument("-s", "--solver", type=str, default='impost', choices=('impost', 'astar', 'apf'), help="Solver")
    args = parser.parse_args()
    # print('Read targets data from ', args.targets)
    # with open(args.targets, 'r') as f:
    #     targets_data = load(f)

    print('Read solution data from ', args.result)
    with open(args.result, 'r') as f:
        data = load(f)

    env = Environment()
    for target_path in data[1:]:
        speed = target_path["items"][0]["length"] / target_path["items"][0]["duration"] * 3600
        env.add_ts(targets.LinearTarget((target_path["items"][0]["x"], target_path["items"][0]["y"]),
                                        (
                                            speed * cos(
                                                radians(target_path["items"][0]["begin_angle"])),
                                            speed * sin(
                                                radians(target_path["items"][0]["begin_angle"]))
                                        ),
                                        2))

    path = []
    time = 0
    first = True
    for segment in data[0]["items"]:
        speed = segment["length"] / segment["duration"]
        if segment["curve"] == 0:
            path.append((segment["x"], segment["y"], time / 3600, speed * 3600))
            time += segment["duration"]
            path.append((segment["x"] + segment["duration"] * speed * cos(radians(segment["begin_angle"])),
                         segment["y"] + segment["duration"] * speed * sin(radians(segment["begin_angle"])),
                         time / 3600,
                         speed * 3600))
        else:
            if first:
                path.append((segment["x"], segment["y"], time / 3600, speed * 3600))
                first = False
            r = 1 / segment['curve']
            beta = segment['begin_angle']
            Xc, Yc = r * cos(radians(beta + 90)), r * sin(radians(beta + 90))
            phi = atan2(-Yc, -Xc)
            Xc += segment["x"]
            Yc += segment["y"]
            dl = 0.001
            curve = segment["curve"]
            r = abs(r)
            dl = segment["length"] / 50
            for length in np.linspace(0, segment["length"], 10):
                time += dl / speed
                angle = length * curve
                path.append((Xc + r * cos(phi + angle),
                             Yc + r * sin(phi + angle),
                             time / 3600,
                             speed * 3600))

    # print("--- %s seconds ---" % (time.time() - start_time))

    fig, axtop, ax1, ax2 = graphics.plot_result(path, env)

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    if args.savefig:
        plt.savefig(args.figdir + '/' + args.taskname + '.png')
    if args.video:
        graphics.anim_save(path, env, save='{}_{}.mp4'.format(args.taskname, args.solver))
