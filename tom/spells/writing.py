from __future__ import annotations

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from matplotlib import cm

from tom.util.math import count_graphs
from tom.util.math import generate_unique_combinations

cmap = cm.get_cmap("viridis")


logger = structlog.get_logger()


def draw_shape(n, radius=1, start_angle=None):
    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]
    print(len(small_angle))
    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    for i in np.arange(len(x)):
        for j in np.arange(len(x))[i + 1 :]:
            plt.plot([x[i], x[j]], [y[i], y[j]], c="k")
    plt.title(f"{n} sides: {count_graphs(n)} possibilities")
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.axis("scaled")
    plt.show()


def draw_spell(level, rang, dtype, area, n=12, start_angle=None, radius=1, title=None):
    ranges = [0, 5, 10, 30, 60, 90, 100, 150, 300, 500, 1000, 5280]
    levels = list(np.arange(0, 12))
    area_types = [
        "None",
        "ST",
        "MT",
        "Wall",
        "Sphere",
        "Cyl",
        "Cone",
        "Line",
        "Cube",
        "Square",
        "Circle",
    ]
    dtypes = [
        "bludgeoning",
        "cold",
        "poison",
        "thunder",
        "acid",
        "necrotic",
        "force",
        "radiant",
        "psychic",
        "lightning",
        "fire",
        "slashing",
    ]

    i_range = ranges.index(rang)
    i_levels = levels.index(level)
    i_area = area_types.index(area)
    i_dtype = dtypes.index(dtype)

    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    atributes = [i_dtype, i_levels, i_range, i_area]
    labels = [
        f"damage type: {dtype}",
        f"level: {level}",
        f"range: {rang}ft",
        f"area_type: {area}",
    ]
    for k in np.arange(len(atributes)):
        # print(k)
        i = atributes[k]
        j = (atributes[k] + k + 1) % 12
        # print(i,j)
        plt.plot([x[i], x[j]], [y[i], y[j]], label=labels[k])
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.title(title)
    plt.axis("scaled")
    plt.axis("off")
    plt.legend()
    plt.show()


# draw_spell(level = 3,rang = 150,dtype = 'fire',area = 'Sphere',title = "Fireball")
# draw_spell(level = 0,rang = 60,dtype = 'radiant',
#           area = 'None',title = "Sacred Flame")


def draw_explain(k, array, n=12, radius=1, start_angle=None, title=None, show=True):
    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    for it in np.arange(n):
        i = it
        j = (it + k) % n
        try:
            plt.plot([x[i], x[j]], [y[i], y[j]], label=array[it], c="k")
        except:
            plt.plot([x[i], x[j]], [y[i], y[j]], label="Unassigned", c="k")
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.title(title)
    plt.axis("scaled")
    if show == True:
        plt.legend()
        plt.show()


def draw_edge_groups(n, radius=1, start_angle=None):
    if start_angle == None:
        start_angle = np.pi / n
    small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

    x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    for i in np.arange(int(math.ceil(len(x) / 2))):
        c = cmap(i / (math.ceil(len(x) / 2)))
        print(i)
        # for j in np.arange(len(x))[i+1:]:
        #   plt.plot([x[i],x[j]],[y[i],y[j]],c = 'k')
        for j in np.arange(len(x)):
            if j == 0:
                plt.plot(
                    [x[j], x[(j + i) % n]],
                    [y[j], y[(j + i) % n]],
                    "-.",
                    color="k",
                    label=i,
                )
            else:
                plt.plot([x[j], x[(j + i) % n]], [y[j], y[(j + i) % n]], "--", color=c)

    plt.legend()
    plt.show()


def calculate_all_possible_shapes(n, k):
    # n = #of sides
    # k = #of sides activated
    existing_sets = []
    all_possible = []

    max_join = n - k
    zeros = np.zeros(k)
    zeros[0] = max_join
    all_possible.append(list(zeros))
    existing_sets.append(set(all_possible[-1]))
    if k == 1:
        return all_possible
    elif k == n:
        return [0]
    while zeros[0] != zeros[-1]:
        print(zeros)
        for i in np.arange(len(zeros) - 1):
            print(i)
            if zeros[i + 1] < zeros[i]:
                zeros[i] -= 1
                zeros[i + 1] += 1

                if set(zeros) not in existing_sets:
                    all_possible.append(list(zeros))

                    existing_sets.append(set(all_possible[-1]))
                    print(zeros)
    return all_possible


def draw_spell_search(
    search_name,
    output_name="output.csv",
    start_angle=None,
    radius=1,
    color=False,
    breakdown=False,
    shape="Straight",
    s=0,
    base="Polygon",
    save=False,
):
    data = pd.read_csv(output_name)
    data_trans = data.T
    if search_name == "Random":
        search_name = random.choice(data["name"])

    indices = [i for i in np.arange(len(data)) if data_trans[i]["name"] == search_name]
    final = data_trans[indices[0]]
    for i in indices:
        d = data_trans[i]

        if d["cantrip_scaling"] <= final["cantrip_scaling"]:
            final = d

    ranges = sorted(set(data["range"]))

    areas = set(data["area_types"].astype(str))
    areas = sorted(list(set(data["area_types"])), key=lambda x: str(x))

    schools = sorted(set(data["school"]))
    levels = np.arange(10)
    dtypes = sorted(set(data["damage"]))

    lens = [len(areas), len(levels), len(ranges), len(dtypes), len(schools)]
    n = max(lens) + 2

    if start_angle == None:
        start_angle = np.pi / n

    # small_angle = [start_angle + i * 2*np.pi/n for i in np.arange(1,n+1)]

    # x,y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    print(final)
    atributes = []
    atribute_labels = []

    atributes.append(list(levels).index(final["level"]))
    atribute_labels.append(f"level: {final['level']}")

    atributes.append(list(schools).index(final["school"]))
    atribute_labels.append(f"School: {final['school']}")

    atributes.append(list(dtypes).index(final["damage"]))
    atribute_labels.append(f"Damage Type: {final['damage']}")

    atributes.append(list(areas).index(final["area_types"]))
    atribute_labels.append(f" Area Type: {final['area_types']}")

    atributes.append(list(ranges).index(final["range"]))
    atribute_labels.append(f" Range: {final['range']}")

    N = 2 * len(atributes) + 1
    # print("Attributes: " ,atributes)
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)

    if breakdown == True:
        for i in np.arange(int(N / 2)):
            ax1 = int(N / 2)
            plt.subplot(1, ax1, i + 1)
            plt.title(atribute_labels[i])
            j = atributes[i] + 1

            if shape == "Straight":
                decode_shape(
                    non_repeating[j],
                    n=N,
                    k=i + 1,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Circle":
                decode_shape_circular(
                    non_repeating[j],
                    k=i + 1,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Non_Centred":
                decode_shape_circular(
                    non_repeating[j],
                    k=i + 1,
                    s=s,
                    color=cmap(i / math.floor(N / 2)),
                    centered=False,
                    base=base,
                )
            else:
                print("Shape Undefined")
        plt.show()
    for i in np.arange(math.floor(N / 2)):
        plt.title(search_name)
        j = atributes[i] + 1
        # print("nr:",non_repeating[j])
        if shape == "Straight":
            decode_shape(
                non_repeating[j],
                n=N,
                k=i + 1,
                color=cmap(i / math.floor(N / 2)),
                label=atribute_labels[i],
                base=base,
            )
        elif shape == "Circle":
            decode_shape_circular(
                non_repeating[j],
                k=i + 1,
                color=cmap(i / math.floor(N / 2)),
                label=atribute_labels[i],
                base=base,
            )
        elif shape == "Non_Centred":
            decode_shape_circular(
                non_repeating[j],
                k=i + 1,
                s=s,
                color=cmap(i / math.floor(N / 2)),
                label=atribute_labels[i],
                centered=False,
                base=base,
            )
        else:
            print("Shape Undefined")
        # decode_shape(non_repeating[j+1],n = N, k = i+1,color = cmap(i/math.floor(N/2)),label = atribute_labels[i])
        # plt.legend(loc='upper left')
    if save == True:
        plt.savefig(f"Diagrams/{search_name}_{shape}_{base}.png", transparent=True)
    plt.show()


def decode_shape(
    in_array, n, k=1, radius=1, start_angle=None, color="k", label=None, base="Polygon"
):
    if start_angle == None:
        start_angle = np.pi / n

    if base == "Polygon":
        small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

        x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    elif base == "Line":
        x = np.arange(0, n + 1)
        y = np.zeros((1, n))

    elif base == "Quadratic":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [x_**2 for x_ in x]
    elif base == "Quadratic2":
        x = [0]
        while len(x) < n:
            if -x[-1] in x:
                x.append(-x[-1] + 1)
            else:
                x.append(-x[-1])
        y = [x_**2 for x_ in x]
    elif base == "SemiCircular":
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "QuarterCircular":
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "CubicFunction":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [0.1 * x_**3 + -0.75 * x_ for x_ in x]

    labelled = False
    for i in range(n):
        if in_array[i] == 1:
            if labelled == False:
                plt.plot(
                    [x[i], x[(i + k) % n]],
                    [y[i], y[(i + k) % n]],
                    "-",
                    color=color,
                    label=label,
                )
                labelled = True
            else:
                plt.plot(
                    [x[i], x[(i + k) % n]], [y[i], y[(i + k) % n]], "-", color=color
                )
        elif in_array[i] == 2:
            offset = 0.01
            plt.plot(
                [x[i], x[(i + k) % n]],
                [y[i], y[(i + k) % n]],
                "-",
                color=color,
                label=label,
            )
            plt.plot(
                [x[i] + offset, x[(i + k) % n] + offset],
                [y[i] + offset, y[(i + k) % n] + offset],
                "-",
                color=color,
            )
        else:
            plt.plot(
                [x[i], x[(i + k) % n]],
                [y[i], y[(i + k) % n]],
                ":",
                linewidth=0.5,
                color=color,
            )
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")
    plt.axis("scaled")
    plt.axis("off")
    # plt.show()


def draw_feature_key(
    N=11,
    output_name="output.csv",
    key="level",
    k=1,
    s=1,
    start_angle=None,
    radius=1,
    color=None,
    shape="Straight",
    base="Polygon",
):
    data = pd.read_csv(output_name)
    print(data.keys())
    if key == "area_types":
        features = set(data["area_types"].astype(str))
        features = sorted(list(set(data["area_types"])), key=lambda x: str(x))
    else:
        features = sorted(set(data[key]))
    print(len(features))
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)

    for i in np.arange(len(features)):
        plt.subplot(math.ceil(len(features) / 4), 4, i + 1)
        # print(non_repeating[i])
        if color == None:
            plt.title(features[i])
            if shape == "Straight":
                decode_shape(
                    non_repeating[i + 1],
                    n=N,
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Circle":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Non_Centre":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    s=s,
                    color=cmap(i / math.floor(N / 2)),
                    centered=False,
                    base=base,
                )
            else:
                print("Shape not supported")
        else:
            if shape == "Straight":
                decode_shape(non_repeating[i + 1], n=N, k=k, color="k", base=base)
            elif shape == "Circle":
                decode_shape_circular(non_repeating[i + 1], k=k, color="k", base=base)
            elif shape == "Non_Centred":
                decode_shape_circular(
                    non_repeating[i + 1], k=k, s=s, color="k", centered=False, base=base
                )
            else:
                print("Shape not supported")
            plt.title(features[i])
    plt.show()


def draw_centre_circle(P, Q, thetas=None):
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]
    a = (x1 + x2) / 2
    b = (y1 + y2) / 2
    r = np.sqrt((a - x1) ** 2 + (b - y1) ** 2)

    if thetas == "Full":
        theta = np.linspace(0, 2 * np.pi, 150)
    else:
        theta0 = math.atan2(y1 - b, x1 - a)
        theta1 = math.atan2(y2 - b, x2 - a)
        if y2 < y1:
            theta0, theta1 = theta1 + np.pi, theta0 + np.pi

        theta = np.linspace(theta0, theta1, 150)
    X2 = r * np.cos(theta) + a
    Y2 = r * np.sin(theta) + b

    return (X2, Y2, a, b)


def draw_non_centre_circle(P, Q, b, radius=0, thetas=None):
    x1 = P[0]
    y1 = P[1]
    x2 = Q[0]
    y2 = Q[1]

    b2 = -b

    delta = x1**2 - x2**2 + y1**2 - y2**2
    a = (delta - 2 * (y1 - y2) * b) / (2 * (x1 - x2))
    a2 = (delta - 2 * (y1 - y2) * b2) / (2 * (x1 - x2))

    r = np.sqrt((x1 - a) ** 2 + (y1 - b) ** 2)
    r2 = np.sqrt((x1 - a2) ** 2 + (y1 - b2) ** 2)
    if r2 <= r:
        a = a2
        b = b2
        r = r2

    if thetas == "Full":
        theta1 = np.linspace(0, 2 * np.pi, 150)

    else:
        theta0 = math.atan2(y1 - b, x1 - a)
        theta1 = math.atan2(y2 - b, x2 - a)

        theta02 = theta0
        theta12 = theta1

        while theta1 < theta0:
            theta0 -= 2 * np.pi

        while theta02 < theta12:
            theta12 -= 2 * np.pi

        arc1 = r * (theta1 - theta0)
        arc2 = r * (theta02 - theta12)

        # print(f'Arc1: {arc1}\n Arc2: {arc2} \n {(arc1+arc2)/(2*np.pi*r)}')
        if arc1 < arc2 or np.sqrt(b**2) < 1:
            theta = np.linspace(theta1, theta0, 150)
        else:
            theta = np.linspace(theta02, theta12, 150)

    X = r * np.cos(theta) + a
    Y = r * np.sin(theta) + b
    return (X, Y, a, b)


def decode_shape_circular(
    in_array,
    k=1,
    radius=1,
    start_angle=None,
    label=None,
    color="k",
    centered=True,
    s=0,
    base="Polygon",
):
    if np.sqrt(s**2) < 1 and "Quadratic" in base:
        raise Exception("Exception: Quadratic shapes requires s >= 1")
    n = len(in_array)
    # print(in_array)
    if start_angle == None:
        start_angle = np.pi / n

    if base == "Polygon":
        small_angle = [start_angle + i * 2 * np.pi / n for i in np.arange(1, n + 1)]

        x, y = (radius * np.sin(small_angle), radius * np.cos(small_angle))
    elif base == "Line":
        x = np.arange(0, n)
        y = np.zeros((1, n))[0]
    elif base == "Quadratic":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [x_**2 for x_ in x]
    elif base == "Quadratic2":
        x = [0]
        while len(x) < n:
            if -x[-1] in x:
                x.append(-x[-1] + 1)
            else:
                x.append(-x[-1])
        y = [x_**2 for x_ in x]

    elif base == "SemiCircular":
        theta0 = 0
        theta1 = -np.pi
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "QuarterCircular":
        theta0 = 0
        theta1 = -np.pi / 2
        theta = np.linspace(theta0, theta1, n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    elif base == "CubicFunction":
        x = np.arange(-math.floor(n / 2), math.ceil(n / 2))
        y = [0.1 * x_**3 + -0.75 * x_ for x_ in x]
        # print(x,y)
    # small_angle = [start_angle + i * 2*np.pi/n for i in np.arange(1,n+1)]

    # x,y = (radius * np.sin(small_angle), radius * np.cos(small_angle))

    labelled = label == None

    for i in range(n):
        P = [x[i], y[i]]
        Q = [x[(i + k) % n], y[(i + k) % n]]
        if centered == True:
            X, Y, a, b = draw_centre_circle(P, Q)
        else:
            X, Y, a, b = draw_non_centre_circle(P, Q, radius=radius, b=s)
        # plt.plot(a,b,"x",color = color)
        if in_array[i] == 1:
            if labelled == False:
                plt.plot(X, Y, label=label, color=color)
                labelled = True
            else:
                plt.plot(X, Y, color=color)

        else:
            # P = [x[i],y[i]]
            # Q = [x[(i+k)%n],y[(i+k)%n]]
            # X,Y,a,b = draw_centre_circle(P,Q)
            plt.plot(X, Y, ":", color=color, linewidth=0.3)
            # plt.plot(a,b,".")
    plt.scatter(x, y, s=70, facecolors="none", edgecolors="k")

    plt.axis("scaled")
    plt.axis("off")


def save_feature_key(
    N=11,
    output_name="output.csv",
    key="level",
    k=1,
    s=1,
    start_angle=None,
    radius=1,
    color=None,
    shape="Straight",
    base="Polygon",
):
    data = pd.read_csv(output_name)
    print(data.keys())
    if key == "area_types":
        features = set(data["area_types"].astype(str))
        features = sorted(list(set(data["area_types"])), key=lambda x: str(x))
        features = [
            str(f)
            .replace("C", "Cube")
            .replace("N", "Cone")
            .replace("L", "Line")
            .replace("R", "Circle")
            .replace("S", "Sphere")
            .replace("MT", "Multiple-Targets")
            .replace("W", "Wall")
            .replace("Y", "Cylinder")
            .replace("SphereT", "Single-Target")
            .replace("nan", "None")
            .replace("Q", "Square")
            for f in features
        ]
        # print(features)
        # input()
    else:
        features = sorted(set(data[key]))
    print(len(features))
    if not os.path.exists(f"keys/{key}_{shape}_{base}"):
        os.mkdir(f"keys/{key}_{shape}_{base}")
    if os.path.isfile(f"Uniques/{N}.npy"):
        non_repeating = np.load(f"Uniques/{N}.npy")
    else:
        non_repeating = generate_unique_combinations(N)
        non_repeating = np.array(non_repeating)
        np.save(f"Uniques/{N}.npy", non_repeating)

    for i in np.arange(len(features)):
        # plt.subplot(math.ceil(len(features)/4),4,i+1)
        # print(non_repeating[i])
        if color == None:
            plt.title(features[i])
            if shape == "Straight":
                decode_shape(
                    non_repeating[i + 1],
                    n=N,
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Circle":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    color=cmap(i / math.floor(N / 2)),
                    base=base,
                )
            elif shape == "Non_Centre":
                decode_shape_circular(
                    non_repeating[i + 1],
                    k=k,
                    s=s,
                    color=cmap(i / math.floor(N / 2)),
                    centered=False,
                    base=base,
                )
            else:
                print("Shape not supported")
        else:
            if shape == "Straight":
                decode_shape(non_repeating[i + 1], n=N, k=k, color="k", base=base)
            elif shape == "Circle":
                decode_shape_circular(non_repeating[i + 1], k=k, color="k", base=base)
            elif shape == "Non_Centred":
                decode_shape_circular(
                    non_repeating[i + 1], k=k, s=s, color="k", centered=False, base=base
                )
            else:
                print("Shape not supported")
            plt.title(f"{features[i]}   {non_repeating[i+1]}")
        print(str(features[i]).replace("/", "-"))
        plt.savefig(
            f'keys/{key}_{shape}_{base}/{str(features[i]).replace("/","-")}.png',
            transparent=True,
        )
        plt.close()
