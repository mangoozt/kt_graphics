import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
import numpy as np


def cc(arg, alpha=.6):
    """
    Shorthand to convert 'named' colors to rgba format with opacity.
    """
    return mcolors.to_rgba(arg, alpha=alpha)


def plot_on_axes(map_ax, v_ax, path, env, timecircles=True):
    """
    Shows path and velocity plots
    :param v_ax: axes for velocity plot
    :param map_ax: axes for map
    :param timecircles: Draw circles every 30 min
    :param path: path in format (x,y,t,v)
    :param env: Environment
    :type env: .environment.Environment
    :return:
    """
    from matplotlib.collections import LineCollection

    path = np.array(path)
    v = path[:, 3]
    t = path[:, 2]
    points = np.array([path[:, 1], path[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # map_ax.scatter(*route.goals[-1][1::-1], marker='*', s=80, color='r')
    map_ax.scatter(*path[0, :2], marker='o')
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, v.max())
    lc = LineCollection(segments, cmap='rainbow', norm=norm)
    # Set the values used for colormapping
    lc.set_array(v)
    lc.set_linewidth(2)
    line = map_ax.add_collection(lc)

    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.colorbar import colorbar
    ax1_divider = make_axes_locatable(v_ax)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = colorbar(line, cax=cax1)
    cb1.ax.get_xaxis().set_visible(False)
    cb1.ax.get_yaxis().set_visible(False)

    # Find point for ticks
    tickstep = .5
    xytt = np.ndarray((int(t[-1] // tickstep) + 1, 3))
    k = 0
    for i in range(0, len(path) - 1):
        while t[i] <= k * tickstep < t[i + 1]:
            w = path[i + 1, :2] - path[i, :2]
            w = w / np.linalg.norm(w)
            xytt[k, :2] = path[i, :2] + w * v[i] * (k * tickstep - t[i])
            xytt[k, 2] = v[i]
            k = k + 1

    map_ax.scatter(xytt[:, 1], xytt[:, 0], cmap='rainbow', c=xytt[:, 2], norm=norm, s=7)
    map_ax.scatter(path[:, 1], path[:, 0], c='k', marker='x')

    # plot targets
    t = np.linspace(0, path[-1, 2])
    tt = np.arange(0, path[-1, 2], tickstep)
    color = iter(cm.get_cmap("tab10")(np.linspace(0, 1, len(env.targets))))

    norm = plt.Normalize(0, path[-1, 3])
    for ts in env.targets:
        xy = (np.outer(ts.init_p, np.ones_like(t)) + np.outer(ts.init_v, t))
        xyt = (np.outer(ts.init_p, np.ones_like(tt)) + np.outer(ts.init_v, tt))
        c = next(color)
        if timecircles:
            for i, p in enumerate(np.flip(xyt).transpose()):
                map_ax.add_artist(
                    plt.Circle(p, ts.safe_r, edgecolor=c, facecolor='none', alpha=1))
        else:
            map_ax.add_artist(plt.Circle(*np.flip(ts.init_p), ts.safe_r, edgecolor=c, facecolor='none'))
        map_ax.plot(*np.flip(xy), color=c)
        map_ax.scatter(*np.flip(xyt), color=c, marker='o', s=5)
        map_ax.scatter(*np.flip(ts.init_p), color=c, marker='p', s=5)
    '''
        # Draw route
        def seg_intersect(a1, l1, a2, l2):
            dap = np.dot([[0, 1], [-1, 0]], l1)
            denom = np.dot(dap, l2)
            num = np.dot(dap, a1 - a2)
            return (num / denom.astype(float)) * l2 + a2
    
        center = []
        left = []
        right = []
        center.append(np.array(route.goals[0])[1::-1])
        w1 = np.array(route.goals[1])[1::-1] - center[0]
        t = np.dot([[0, 1], [-1, 0]], w1 / np.linalg.norm(w1) * route.width)
        left.append(center[-1] + t)
        right.append(center[-1] - t)
        s2 = np.array(route.goals[1])[1::-1]
        for i in range(1, len(route.goals) - 1):
            s3 = np.array(route.goals[i + 1])[1::-1]
            w2 = s3 - s2
            t = np.dot([[0, 1], [-1, 0]], w2 / np.linalg.norm(w2) * route.width)
            left.append(seg_intersect(left[-1], w1, s2 + t, w2))
            right.append(seg_intersect(right[-1], w1, s2 - t, w2))
            center.append(s2)
            w1 = w2
            s2 = s3
        center.append(s2)
        left.append(s2 + t)
        right.append(s2 - t)
    
        map_ax.plot(*zip(*left), color=cc('k'))
        map_ax.plot(*zip(*center), color=cc('k'), linestyle='-.')
        map_ax.plot(*zip(*right), color=cc('k'))
    '''
    map_ax.set_aspect('equal', 'box')

    v_ax.step(list(path[:, 2]), v, where='post', alpha=0.8)
    v_ax.plot(list(path[:, 2]), v, 'kx')
    v_ax.ticklabel_format(useOffset=False)
    v_ax.set_ylim(bottom=0)
    v_ax.grid(which='both')
    v_ax.set_xlabel("t, hr")
    v_ax.set_ylabel(r"$V_{OS}, knots$")


def plot_result(path, env, timecircles=True, title=None):
    """
    Plots one result
    :param path: path in format List[(x,y,t,v),...]
    :param env: Environment
    :param route: Route
    :param timecircles: Draw circles every 30min
    :param title: Title of the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_on_axes(ax1, ax2, path, env, timecircles)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.set_size_inches(11, 5)
    plt.show()


def plot_result_two(paths, envs, route, timecircles=True, title=None, show=True):
    """
    Plots two results side-by-side
    :param paths: List with two paths in format List[(x,y,t,v),...]
    :param envs:  List with two  Environments
    :param route:  List with two routes
    :param timecircles: Draw circles every 30min
    :param title: Title of the figure
    :param show: Show or return figure and axes in format (fig, axtop, ax1, ax2)
    :return: None||(fig, axtop, ax1, ax2)
    """
    fig = plt.gcf()
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[2, 2],
                           height_ratios=[1, 3],
                           figure=fig)
    axtop = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    plot_on_axes(ax1, axtop, paths[0], envs[0], route, timecircles)
    plot_on_axes(ax2, axtop, paths[1], envs[1], route, timecircles)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    if show:
        plt.show()
    else:
        return fig, axtop, ax1, ax2


def plot_on_axes_t(map_ax, v_ax, path, env, time, timecircles=True):
    """
    Shows path and velocity plots
    :param v_ax: axes for velocity plot
    :param map_ax: axes for map
    :param timecircles: Draw circles every 30 min
    :param path: path in format (x,y,t,v)
    :param env: Environment
    :type env: .environment.Environment
    :return:
    """
    from matplotlib.collections import LineCollection

    # Draw route
    def seg_intersect(a1, l1, a2, l2):
        dap = np.dot([[0, 1], [-1, 0]], l1)
        denom = np.dot(dap, l2)
        num = np.dot(dap, a1 - a2)
        return (num / denom.astype(float)) * l2 + a2

    center = []
    left = []
    right = []
    center.append(np.array(route.goals[0])[1::-1])
    w1 = np.array(route.goals[1])[1::-1] - center[0]
    t = np.dot([[0, 1], [-1, 0]], w1 / np.linalg.norm(w1) * route.width)
    left.append(center[-1] + t)
    right.append(center[-1] - t)
    s2 = np.array(route.goals[1])[1::-1]
    for i in range(1, len(route.goals) - 1):
        s3 = np.array(route.goals[i + 1])[1::-1]
        w2 = s3 - s2
        t = np.dot([[0, 1], [-1, 0]], w2 / np.linalg.norm(w2) * route.width)
        left.append(seg_intersect(left[-1], w1, s2 + t, w2))
        right.append(seg_intersect(right[-1], w1, s2 - t, w2))
        center.append(s2)
        w1 = w2
        s2 = s3
    center.append(s2)
    left.append(s2 + t)
    right.append(s2 - t)

    map_ax.plot(*zip(*left), color=cc('k'))
    map_ax.plot(*zip(*center), color=cc('k'), linestyle='-.')
    map_ax.plot(*zip(*right), color=cc('k'))

    sizes = (map_ax.get_xlim(), map_ax.get_ylim())

    path = np.array(path)
    v = path[:, 3]
    t = path[:, 2]
    if time != 0:
        n = np.max(np.where(t < time))
        p = np.append(path[:n + 1, (1, 0)],
                      [path[n, (1, 0)] + (path[n + 1, (1, 0)] - path[n, (1, 0)]) / (t[n + 1] - t[n]) * (time - t[n])],
                      axis=0)

        points = np.array([p[:, 0], p[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        map_ax.scatter(*route.goals[-1][1::-1], marker='*', s=80, color='r')
        map_ax.scatter(*path[0, :2], marker='o')
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, v.max())
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        # Set the values used for colormapping
        lc.set_array(v)
        lc.set_linewidth(2)
        line = map_ax.add_collection(lc)

        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mpl_toolkits.axes_grid1.colorbar import colorbar
        ax1_divider = make_axes_locatable(v_ax)
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cb1 = colorbar(line, cax=cax1)
        cb1.ax.get_xaxis().set_visible(False)
        cb1.ax.get_yaxis().set_visible(False)

        # Find point for ticks
        tickstep = .5
        xyv = np.ndarray((int(time // tickstep) + 1, 3))
        k = 0
        for i in range(0, n+1):
            while k * tickstep <= time and t[i] <= k * tickstep < t[i + 1]:
                w = path[i + 1, :2] - path[i, :2]
                w = w / np.linalg.norm(w)
                xyv[k, :2] = path[i, :2] + w * v[i] * (k * tickstep - t[i])
                xyv[k, 2] = v[i]
                k = k + 1

        map_ax.scatter(xyv[:, 1], xyv[:, 0], cmap='rainbow', c=xyv[:, 2], norm=norm, s=7)
        map_ax.scatter(path[:n + 1, 1], path[:n + 1, 0], c='k', marker='x')

        sizes = (map_ax.get_xlim(), map_ax.get_ylim())

        # plot targets
        t = np.linspace(0, time)
        tt = np.arange(0, time, tickstep)
        color = iter(cm.get_cmap("tab10")(np.linspace(0, 1, len(env.targets))))

        norm = plt.Normalize(0, path[-1, 3])
        for ts in env.targets:
            xy = (np.outer(ts.init_p, np.ones_like(t)) + np.outer(ts.init_v, t))
            xyt = (np.outer(ts.init_p, np.ones_like(tt)) + np.outer(ts.init_v, tt))
            c = next(color)
            if timecircles:
                for i, p in enumerate(np.flip(xyt).transpose()):
                    map_ax.add_artist(
                        plt.Circle(p, ts.safe_r, edgecolor=c, facecolor='none', alpha=1))
            else:
                map_ax.add_artist(plt.Circle(*np.flip(ts.init_p), ts.safe_r, edgecolor=c, facecolor='none'))
            map_ax.add_artist(plt.Circle((xy[1, -1], xy[0, -1]), ts.safe_r, edgecolor=c, facecolor='none', alpha=.5))
            map_ax.plot(*np.flip(xy), color=c)
            map_ax.scatter(*np.flip(xyt), color=c, marker='o', s=5)
            map_ax.scatter(*np.flip(ts.init_p), color=c, marker='p', s=5)

        v_ax.step(list(np.append(path[:n + 1, 2], [time])), list(np.append(v[:n + 1], [v[n]])), where='post', alpha=0.8)
        v_ax.axvline(x=time, color='k', linewidth=1)
        v_ax.plot(list(path[:, 2]), v, 'kx')
        v_ax.ticklabel_format(useOffset=False)
        v_ax.set_ylim(bottom=0)
        v_ax.grid(which='both')
        v_ax.set_xlabel("t, hr")
        v_ax.set_ylabel(r"$V_{OS}, knots$")

    map_ax.set_xlim(*sizes[0])
    map_ax.set_ylim(*sizes[1])

    map_ax.set_aspect('equal', 'box')


def anim_save(path, env, timecircles=True, title=None, show=True, save=None, timescale=1 / 360, dt=1):
    if (save is not None):
        from moviepy.video.io.bindings import mplfig_to_npimage
        import moviepy.editor as mpy
        print('Prepare animation')
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        gs = fig.add_gridspec(5, 1)
        v_ax = fig.add_subplot(gs[0, :])
        map_ax = fig.add_subplot(gs[1:, :])

        def make_frame_mpl(t):
            map_ax.clear()
            v_ax.clear()
            plot_on_axes_t(map_ax, v_ax, path, env, route, time=t / timescale / 3600)
            # map_ax.autoscale_view()
            npimage = mplfig_to_npimage(fig)
            return npimage

        animation = mpy.VideoClip(make_frame_mpl, duration=path[-1, 2] * 3600 * timescale)

        animation.write_videofile(save, fps=15, threads=8, preset='ultrafast', audio=False, logger='bar')
